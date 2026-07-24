//! Remote layer worker: loads a GGUF model, listens on TCP, and processes
//! hidden states through an assigned layer range.
//!
//! Wire protocol (requests from local host):
//!   [1B cmd=0x00] [4B LE layer_start] [4B LE layer_end]
//!   [4B LE seq_len] [4B LE start_offset]
//!   [8B LE payload_len] [payload bytes]
//!
//! Response:
//!   [8B LE response_len] [response bytes]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::Arc;

use candle_core::{DType, Device, Tensor};
use mistralrs_core::{
    device_map::{DeviceMapper, LayerDeviceMapper},
    pipeline::gguf::GGUFArchitecture,
    DeviceMapSetting, IsqOrganization,
};

use anyhow::Context;

const CMD_COMPUTE: u8 = 0x00;
const CMD_RESET: u8 = 0x01;

pub fn run_remote_worker(
    model_dir: &str,
    model_file: &str,
    listen_addr: &str,
    start_layer: usize,
    end_layer: usize,
    device: Device,
) -> anyhow::Result<()> {
    println!("Loading GGUF model from {}/{}...", model_dir, model_file);
    println!("Layer range: {}-{}", start_layer, end_layer);
    println!("Listening on {}", listen_addr);

    // We use the full GGUF pipeline to load the model, but configure it
    // so only our owned layers are loaded. We use a custom topology mapper.
    // For now: load the full model via mistralrs-core, but only use our layers.

    // Create a topology-like device assignment: our layers on `device`, rest as remote
    // We need a layer count first. Parse the GGUF metadata.

    let listener = TcpListener::bind(listen_addr)
        .context(format!("Failed to bind to {listen_addr}"))?;

    // For now, we accept one connection and process requests sequentially.
    // In a real deployment, we'd use connection pooling.
    println!("Waiting for connection...");
    let (mut stream, addr) = listener.accept()?;
    println!("Connected from {addr}");

    stream
        .set_nodelay(true)
        .context("Failed to set TCP_NODELAY")?;

    // Load the model. For the initial implementation, we use the openai-harmony approach
    // of loading the model lazily. We'll just use mistralrs-core's Runner API.
    // Actually, for simplicity, let's create a minimal model loader.

    // TODO: Use full model loading with a custom topology that marks our layers as local
    // and all others as remote. The remote layers will be skipped (Option::None).
    // For now, we run a standalone model instance.
    println!("Model loading would happen here. Starting event loop...");

    loop {
        // Read command header
        let mut cmd_buf = [0u8; 1];
        if stream.read_exact(&mut cmd_buf).is_err() {
            println!("Connection closed");
            break;
        }

        match cmd_buf[0] {
            CMD_COMPUTE => {
                let mut header = [0u8; 16];
                stream
                    .read_exact(&mut header)
                    .context("Failed to read header")?;

                let _layer_start = u32::from_le_bytes(header[0..4].try_into().unwrap());
                let _layer_end = u32::from_le_bytes(header[4..8].try_into().unwrap());
                let _seq_len = u32::from_le_bytes(header[8..12].try_into().unwrap());
                let _start_offset = u32::from_le_bytes(header[12..16].try_into().unwrap());

                let mut len_buf = [0u8; 8];
                stream
                    .read_exact(&mut len_buf)
                    .context("Failed to read payload length")?;
                let payload_len = u64::from_le_bytes(len_buf) as usize;

                let mut payload = vec![0u8; payload_len];
                stream
                    .read_exact(&mut payload)
                    .context("Failed to read payload")?;

                // For now, echo back a dummy response
                // Real implementation: deserialize → forward through layers → serialize → send back
                let dummy: Vec<u8> = payload[..std::cmp::min(16, payload_len)].to_vec();
                let resp_len = dummy.len() as u64;
                stream
                    .write_all(&resp_len.to_le_bytes())
                    .context("Failed to write response length")?;
                stream
                    .write_all(&dummy)
                    .context("Failed to write response")?;
                stream.flush().context("Failed to flush")?;
            }
            CMD_RESET => {
                println!("Reset command received");
                // Reset KV cache (no-op for now)
            }
            _ => {
                println!("Unknown command: {}", cmd_buf[0]);
                break;
            }
        }
    }

    Ok(())
}
