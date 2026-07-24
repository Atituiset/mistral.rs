use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use candle_core::{DType, Device, Result, Tensor};
use tracing::info;

use crate::topology::RemoteAwareDevice;
use crate::TryIntoDType;

use super::mappers::{DeviceMapper, LayerDeviceMapper};

/// Persistent TCP connections to remote workers, keyed by address string.
pub struct RemoteConnectionPool {
    connections: HashMap<String, Mutex<TcpStream>>,
}

impl RemoteConnectionPool {
    pub fn new(mappings: &[RemoteAwareDevice]) -> Result<Self> {
        let mut unique_addrs = HashMap::new();
        for dev in mappings {
            if let RemoteAwareDevice::Remote { addr } = dev {
                if !unique_addrs.contains_key(addr) {
                    let stream = TcpStream::connect(addr.as_str()).map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "Failed to connect to remote worker at {addr}: {e}"
                        ))
                    })?;
                    stream
                        .set_nodelay(true)
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    stream
                        .set_read_timeout(Some(Duration::from_secs(300)))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    stream
                        .set_write_timeout(Some(Duration::from_secs(30)))
                        .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                    unique_addrs.insert(addr.clone(), Mutex::new(stream));
                }
            }
        }
        info!("Initialized {} remote connection(s)", unique_addrs.len());
        Ok(Self {
            connections: unique_addrs,
        })
    }

    /// Send a command + tensor payload to the remote worker and receive the response tensor.
    /// Protocol:
    ///   Send: [1B cmd][4B LE layer_start][4B LE layer_end][8B LE tensor_len][tensor_bytes]
    ///   Recv: [8B LE response_len][response_bytes]
    pub fn roundtrip(
        &self,
        addr: &str,
        cmd: u8,
        layer_start: u32,
        layer_end: u32,
        payload: &[u8],
    ) -> Result<Vec<u8>> {
        let mut guard = self
            .connections
            .get(addr)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("No connection for remote worker at {addr}"))
            })?
            .lock()
            .unwrap();

        let try_op = |stream: &mut TcpStream| -> Result<Vec<u8>> {
            // Write header: [1B cmd][4B LE start][4B LE end][4B LE past_kv][4B reserved][8B LE len]
            tracing::debug!(target: "remote", "Sending cmd={cmd} start={layer_start} end={layer_end} len={}", payload.len());
            stream
                .write_all(&[cmd])
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            stream
                .write_all(&layer_start.to_le_bytes())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            stream
                .write_all(&layer_end.to_le_bytes())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let past_kv: u32 = 0; // KV cache offset, 0 for first forward
            stream
                .write_all(&past_kv.to_le_bytes())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let reserved: u32 = 0;
            stream
                .write_all(&reserved.to_le_bytes())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let len = payload.len() as u64;
            stream
                .write_all(&len.to_le_bytes())
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            stream
                .write_all(payload)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            stream
                .flush()
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;

            // Read response
            let mut len_buf = [0u8; 8];
            stream
                .read_exact(&mut len_buf)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            let resp_len = u64::from_le_bytes(len_buf) as usize;
            let mut resp = vec![0u8; resp_len];
            stream
                .read_exact(&mut resp)
                .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
            Ok(resp)
        };

        match try_op(&mut guard) {
            Ok(resp) => Ok(resp),
            Err(e) => {
                // Attempt reconnection on failure
                info!("Remote connection to {addr} lost, reconnecting...");
                match TcpStream::connect(addr.clone()) {
                    Ok(new_stream) => {
                        let _ = new_stream.set_nodelay(true);
                        let _ = new_stream.set_read_timeout(Some(Duration::from_secs(300)));
                        *guard = new_stream;
                    }
                    Err(conn_err) => {
                        return Err(candle_core::Error::Msg(format!(
                            "Failed to reconnect to {addr}: {conn_err}"
                        )));
                    }
                }
                // Retry once after reconnection
                try_op(&mut guard)
            }
        }
    }
}

/// Serialize a tensor to a byte vector: [8B LE num_elements][8B LE ndims][dim0..dimN as u64 LE][f32 data]
pub fn serialize_tensor(t: &Tensor) -> Result<Vec<u8>> {
    let t = t.to_device(&Device::Cpu)?.to_dtype(DType::F32)?;
    let data = t.flatten_all()?.to_vec1::<f32>()?;
    let shape = t.shape().dims();

    let mut buf = Vec::with_capacity(16 + shape.len() * 8 + data.len() * 4);
    buf.extend_from_slice(&(data.len() as u64).to_le_bytes());
    buf.extend_from_slice(&(shape.len() as u64).to_le_bytes());
    for &d in shape {
        buf.extend_from_slice(&(d as u64).to_le_bytes());
    }
    // SAFETY: f32 slice to u8 bytes
    let byte_data: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    buf.extend_from_slice(byte_data);
    Ok(buf)
}

/// Deserialize bytes back to a tensor on the target device.
pub fn deserialize_tensor(data: &[u8], device: &Device) -> Result<Tensor> {
    let n_elements = u64::from_le_bytes(data[0..8].try_into().unwrap()) as usize;
    let n_dims = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    let mut shape = Vec::with_capacity(n_dims);
    for i in 0..n_dims {
        shape.push(u64::from_le_bytes(
            data[16 + i * 8..24 + i * 8].try_into().unwrap(),
        ) as usize);
    }
    let data_offset = 16 + n_dims * 8;
    let expected_bytes = n_elements * 4;
    if data.len() < data_offset + expected_bytes {
        return Err(candle_core::Error::Msg(format!(
            "Deserialize tensor: expected {} bytes, got {}",
            data_offset + expected_bytes,
            data.len()
        )));
    }
    let f32_data: &[f32] = unsafe {
        std::slice::from_raw_parts(
            data[data_offset..].as_ptr() as *const f32,
            n_elements,
        )
    };
    Tensor::from_vec(f32_data.to_vec(), shape.as_slice(), device)
}

/// Mapper that delegates to `LayerDeviceMapper` for local layers and forwards
/// hidden states to remote workers for remote layers.
pub struct RemoteLayerMapper {
    local_mapper: LayerDeviceMapper,
    connection_pool: RemoteConnectionPool,
    /// Per-layer device assignment: Local(dev) or Remote{addr}
    layer_specs: Vec<RemoteAwareDevice>,
    /// For each remote block: (addr, first_layer, last_layer)
    remote_blocks: Vec<(String, usize, usize)>,
    /// Map from layer index to remote block index
    layer_to_block: Vec<Option<usize>>,
}

impl RemoteLayerMapper {
    pub fn new(
        local_mapper: LayerDeviceMapper,
        connection_pool: RemoteConnectionPool,
        layer_specs: Vec<RemoteAwareDevice>,
    ) -> Self {
        let n = layer_specs.len();

        // Identify contiguous remote blocks
        let mut remote_blocks: Vec<(String, usize, usize)> = Vec::new();
        let mut layer_to_block: Vec<Option<usize>> = vec![None; n];
        let mut i = 0;
        while i < n {
            if let RemoteAwareDevice::Remote { addr } = &layer_specs[i] {
                let block_idx = remote_blocks.len();
                let start = i;
                let addr = addr.clone();
                while i < n {
                    match &layer_specs[i] {
                        RemoteAwareDevice::Remote { addr: a } if a == &addr => {
                            layer_to_block[i] = Some(block_idx);
                            i += 1;
                        }
                        _ => break,
                    }
                }
                remote_blocks.push((addr, start, i - 1));
            } else {
                i += 1;
            }
        }

        for (addr, start, end) in &remote_blocks {
            info!("Remote block: layers {start}-{end} → {addr}");
        }

        Self {
            local_mapper,
            connection_pool,
            layer_specs,
            remote_blocks,
            layer_to_block,
        }
    }
}

impl std::fmt::Debug for RemoteLayerMapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RemoteLayerMapper")
            .field("layer_specs", &self.layer_specs)
            .field("remote_blocks", &self.remote_blocks)
            .finish()
    }
}

impl DeviceMapper for RemoteLayerMapper {
    fn map(&self, input: Tensor, layer: usize) -> Result<Tensor> {
        match self.layer_specs.get(layer) {
            Some(RemoteAwareDevice::Local(_)) => self.local_mapper.map(input, layer),
            Some(RemoteAwareDevice::Remote { addr }) => {
                let block_idx = self.layer_to_block[layer].unwrap();
                let (addr, start, end) = &self.remote_blocks[block_idx];
                let payload = serialize_tensor(&input)?;
                let resp = self
                    .connection_pool
                    .roundtrip(addr, 0x00, *start as u32, *end as u32, &payload)?;
                deserialize_tensor(&resp, &Device::Cpu)
            }
            None => {
                // Layer beyond known range, pass through
                Ok(input)
            }
        }
    }

    fn set_device(
        &self,
        layer: usize,
        varbuilder: mistralrs_quant::ShardedVarBuilder,
        loading_isq: bool,
    ) -> mistralrs_quant::ShardedVarBuilder {
        match self.layer_specs.get(layer) {
            Some(RemoteAwareDevice::Local(_)) => {
                self.local_mapper
                    .set_device(layer, varbuilder, loading_isq)
            }
            _ => varbuilder.set_device(Device::Cpu),
        }
    }

    fn device_for(&self, layer: usize, loading_isq: bool) -> Option<&Device> {
        if loading_isq {
            return self.local_mapper.device_for(layer, loading_isq);
        }
        match self.layer_specs.get(layer) {
            Some(RemoteAwareDevice::Local(dev)) => Some(dev),
            Some(RemoteAwareDevice::Remote { .. }) => {
                // Return None: caller should skip weight loading for this layer
                None
            }
            None => self.local_mapper.device_for(layer, loading_isq),
        }
    }

    fn get_unique_devices(&self) -> Vec<Device> {
        self.local_mapper.get_unique_devices()
    }

    fn cast_nm_device(&self, x: &Tensor, loading_isq: bool) -> Result<Tensor> {
        self.local_mapper.cast_nm_device(x, loading_isq)
    }

    fn set_nm_device(
        &self,
        varbuilder: mistralrs_quant::ShardedVarBuilder,
        loading_isq: bool,
    ) -> mistralrs_quant::ShardedVarBuilder {
        self.local_mapper.set_nm_device(varbuilder, loading_isq)
    }

    fn num_device_mapping_layers(&self) -> usize {
        self.layer_specs.len()
    }

    fn get_comm_for(&self, layer_idx: usize) -> Result<Arc<mistralrs_quant::Comm>> {
        match self.layer_specs.get(layer_idx) {
            Some(RemoteAwareDevice::Local(_)) => self.local_mapper.get_comm_for(layer_idx),
            _ => Err(candle_core::Error::Msg(
                "No NCCL comm for remote layer".to_string(),
            )),
        }
    }

    fn get_min_dtype(&self, dtype: &dyn TryIntoDType) -> Result<DType> {
        self.local_mapper.get_min_dtype(dtype)
    }

    fn is_layer_remote(&self, layer: usize) -> bool {
        self.layer_specs
            .get(layer)
            .is_some_and(|d| d.is_remote())
    }
}
