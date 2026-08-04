//! Remote layer worker for cross-machine heterogeneous inference.
//!
//! Loads a GGUF model (all layers on CPU), listens on TCP, and processes
//! hidden-state forward requests for an assigned layer range.
//!
//! Wire protocol:
//!   Request:  [1B cmd] [4B LE start] [4B LE end] [4B LE past_kv] [4B reserved] [8B LE len] [bytes]
//!   Response: [8B LE len] [bytes]

use std::io::{Read, Write};
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use candle_core::Device;
use mistralrs_core::{
    deserialize_tensor, serialize_tensor, AdapterPaths, DeviceMapMetadata, DeviceMapSetting,
    GGUFLoaderBuilder, GGUFSpecificConfig, Loader, LocalModelPaths, ModelDType, ModelPaths,
    Pipeline,
};

const CMD_COMPUTE: u8 = 0x00;
const CMD_RESET: u8 = 0x01;

pub fn run_remote_worker(
    model_dir: &str,
    model_file: &str,
    listen_addr: &str,
    start_layer: usize,
    end_layer: usize,
) -> anyhow::Result<()> {
    let model_path = PathBuf::from(model_dir).join(model_file);
    if !model_path.exists() {
        anyhow::bail!("Model file not found: {}", model_path.display());
    }

    let layer_range = if start_layer == 0 && end_layer == 0 {
        None // Load all layers
    } else {
        Some((start_layer, end_layer))
    };

    let loader_builder = GGUFLoaderBuilder::new(
        None,
        None,
        model_dir.to_string(),
        vec![model_file.to_string()],
        GGUFSpecificConfig {
            layer_range,
            ..Default::default()
        },
        false,
        None,
    );
    let loader: Box<dyn Loader> = loader_builder.build();

    let paths = LocalModelPaths {
        tokenizer_filename: PathBuf::new(),
        config_filename: PathBuf::new(),
        template_filename: None,
        filenames: vec![model_path],
        adapter_paths: AdapterPaths::None,
        gen_conf: None,
        preprocessor_config: None,
        processor_config: None,
        chat_template_json_filename: None,
    };
    let paths: Box<dyn ModelPaths> = Box::new(paths);

    let device = Device::Cpu;
    let dtype = ModelDType::F32;
    let mapper = DeviceMapSetting::Map(DeviceMapMetadata::dummy());

    let pipeline: Arc<tokio::sync::Mutex<dyn Pipeline + Send + Sync>> = loader.load_model_from_path(
        &paths,
        &dtype,
        &device,
        true,
        mapper,
        None,
        None,
    )?;

    println!("Model loaded successfully");

    let listener =
        TcpListener::bind(listen_addr).context(format!("Failed to bind to {listen_addr}"))?;
    println!("Listening on {listen_addr}");

    // Accept connections and spawn a thread for each one.
    // The pipeline (model) is shared via Arc; each handler uses try_lock().
    for stream in listener.incoming() {
        let mut stream = match stream {
            Ok(s) => s,
            Err(e) => {
                eprintln!("accept error: {e}");
                continue;
            }
        };
        let addr = match stream.peer_addr() {
            Ok(a) => a,
            Err(_) => continue,
        };
        println!("Connected from {addr}");
        stream.set_nodelay(true)?;

        let pipeline = Arc::clone(&pipeline);

        std::thread::spawn(move || {
            loop {
                let mut cmd = [0u8; 1];
                if stream.read_exact(&mut cmd).is_err() {
                    break;
                }

                let mut header = [0u8; 16];
                if stream.read_exact(&mut header).is_err() {
                    break;
                }
                let layer_start = u32::from_le_bytes(header[0..4].try_into().unwrap()) as usize;
                let layer_end = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
                let past_kv = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;

                let mut len_buf = [0u8; 8];
                if stream.read_exact(&mut len_buf).is_err() {
                    break;
                }
                let payload_len = u64::from_le_bytes(len_buf) as usize;

                let mut payload = vec![0u8; payload_len];
                if stream.read_exact(&mut payload).is_err() {
                    break;
                }

                match cmd[0] {
                    CMD_COMPUTE => {
                        let hidden = match deserialize_tensor(&payload, &Device::Cpu) {
                            Ok(t) => t,
                            Err(e) => {
                                eprintln!("[{addr}] deserialize error: {e}");
                                break;
                            }
                        };

                        let result = {
                            let pipeline = match pipeline.try_lock() {
                                Ok(p) => p,
                                Err(_) => {
                                    eprintln!("[{addr}] pipeline lock failed");
                                    break;
                                }
                            };
                            pipeline.forward_from_layer(&hidden, layer_start, layer_end, past_kv)
                        };

                        match result {
                            Ok(output) => {
                                let resp = match serialize_tensor(&output) {
                                    Ok(r) => r,
                                    Err(e) => {
                                        eprintln!("[{addr}] serialize error: {e}");
                                        break;
                                    }
                                };
                                if stream.write_all(&(resp.len() as u64).to_le_bytes()).is_err() {
                                    break;
                                }
                                if stream.write_all(&resp).is_err() {
                                    break;
                                }
                                if stream.flush().is_err() {
                                    break;
                                }
                            }
                            Err(e) => {
                                eprintln!("[{addr}] forward_from_layer error: {e}");
                                break;
                            }
                        }
                    }
                    CMD_RESET => {
                        match pipeline.try_lock() {
                            Ok(pipeline) => {
                                if let Err(e) = pipeline.reset_kv_cache() {
                                    eprintln!("[{addr}] reset_kv_cache error: {e}");
                                }
                            }
                            Err(_) => {
                                eprintln!("[{addr}] pipeline lock failed for reset");
                            }
                        }
                        println!("[{addr}] KV cache reset");
                    }
                    _ => break,
                }
            }
            println!("Connection from {addr} closed");
        });
    }
    #[allow(unreachable_code)]
    Ok(())
}
