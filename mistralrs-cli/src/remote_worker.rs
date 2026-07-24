//! Remote layer worker for cross-machine heterogeneous inference.
//!
//! Loads a GGUF model with a topology that marks only the assigned layer range
//! as local (CPU); all other layers are skipped. Listens on TCP for hidden state
//! requests and runs `forward_from_layer()` on the loaded model.
//!
//! Wire protocol:
//!   Request:  [1B cmd] [4B LE start] [4B LE end] [4B LE past_kv] [4B reserved] [8B LE len] [bytes]
//!   Response: [8B LE len] [bytes]

use std::io::{Read, Write};
use std::net::TcpListener;

use anyhow::Context;

const CMD_COMPUTE: u8 = 0x00;
const CMD_RESET: u8 = 0x01;

pub fn run_remote_worker(
    _model_dir: &str,
    _model_file: &str,
    listen_addr: &str,
    start_layer: usize,
    end_layer: usize,
) -> anyhow::Result<()> {
    println!(
        "Remote worker: owns layers {}-{}, listening on {}",
        start_layer, end_layer, listen_addr
    );

    // TODO: Load the GGUF model using GGUFPipeline with a custom topology
    // that marks layers start_layer..=end_layer as Local(Cpu) and all others
    // as Remote. This causes the model loading code to skip remote layers.
    //
    // Once loaded, the worker calls model.forward_from_layer() for each
    // incoming hidden state tensor.
    //
    // For now, this skeleton proves the protocol integration compiles.

    let listener =
        TcpListener::bind(listen_addr).context(format!("Failed to bind to {listen_addr}"))?;
    println!("Waiting for connection...");
    let (mut stream, addr) = listener.accept()?;
    println!("Connected from {addr}");
    stream.set_nodelay(true)?;

    loop {
        let mut cmd = [0u8; 1];
        if stream.read_exact(&mut cmd).is_err() {
            break;
        }

        let mut header = [0u8; 16];
        stream.read_exact(&mut header)?;
        let _layer_start = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
        let _layer_end = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;

        let mut len_buf = [0u8; 8];
        stream.read_exact(&mut len_buf)?;
        let payload_len = u64::from_le_bytes(len_buf) as usize;

        let mut payload = vec![0u8; payload_len];
        stream.read_exact(&mut payload)?;

        match cmd[0] {
            CMD_COMPUTE => {
                // Placeholder: echo payload back
                // Real: deserialize → forward_from_layer → serialize → send
                stream.write_all(&(payload_len as u64).to_le_bytes())?;
                stream.write_all(&payload)?;
                stream.flush()?;
            }
            CMD_RESET => {
                println!("KV cache reset");
            }
            _ => break,
        }
    }

    Ok(())
}
