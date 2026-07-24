#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::attention::{AttentionMask, SdpaParams};
use crate::device_map::{DeviceMappedMask, DeviceMapper};
use crate::gguf::Content;
use crate::layers::{CausalMaskConfig, CausalMasker, QRmsNorm, RotaryEmbedding, Sdpa};
use crate::layers_masker::PastKvLenCache;
use crate::paged_attention::{AttentionImplementation, PagedAttention};
use crate::pipeline::text_models_inputs_processor::PagedAttentionInputMetadata;
use crate::pipeline::{extract_logits, EitherCache, KvCache, NormalCache};
use crate::utils::gguf_metadata::ContentMetadata;
use crate::utils::model_config as ModelConfig;
use crate::utils::progress::{new_multi_progress, NiceProgressBar};
use candle_core::quantized::QMatMul;
use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{Embedding, Module};
use mistralrs_quant::{GgufMatMul, QuantMethod, QuantMethodConfig};

// Default fallback for models that don't specify context_length
const DEFAULT_MAX_SEQ_LEN: u32 = 4096;

struct Mlp {
    feed_forward_w1: Arc<dyn QuantMethod>,
    feed_forward_w2: Arc<dyn QuantMethod>,
    feed_forward_w3: Arc<dyn QuantMethod>,
}

// candle's QTensor::indexed_moe_forward is cuda-only; other devices take the
// dequantize-and-gather fallback
fn experts_moe_forward(
    w: &QMatMul,
    xs: &candle_core::Tensor,
    indices: &candle_core::Tensor,
) -> candle_core::Result<candle_core::Tensor> {
    if xs.device().is_cuda() {
        w.indexed_moe_forward(xs, indices)
    } else {
        mistralrs_quant::cpu_indexed_moe_forward(w, xs, indices)
    }
}

impl Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let w1 = self.feed_forward_w1.forward(xs)?;
        let w3 = self.feed_forward_w3.forward(xs)?;
        let y = crate::ops::mul_and_act(&w1, &w3, crate::layers::Activation::Silu)?;
        self.feed_forward_w2.forward(&y)
    }
}

struct FusedMoe {
    gate: QMatMul,
    gate_experts: QMatMul,
    up_experts: QMatMul,
    down_experts: QMatMul,
    norm_topk_prob: bool,
    num_experts_per_tok: usize,
}

impl FusedMoe {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, hidden_dim) = xs.dims3()?;
        let xs = xs.reshape(((), hidden_dim))?;
        let original_dtype = xs.dtype();
        let (num_tokens, hidden_dim) = xs.dims2()?;
        let router_logits = self.gate.forward(&xs.to_dtype(DType::F32)?)?;
        let topk = crate::ops::moe_router_topk(
            &router_logits,
            crate::ops::MoeRouterTopKConfig {
                top_k: self.num_experts_per_tok,
                score_function: crate::ops::MoeRouterScoreFunction::Softmax,
                selected_weight: crate::ops::MoeRouterSelectedWeight::Score,
                renormalize: self.norm_topk_prob,
                norm_min: 0.0,
                output_scale: 1.0,
                logit_clip: None,
            },
            None,
            None,
        )?;
        let (scores, indices) = (topk.values, topk.indices);

        let ys = {
            let xs = xs.reshape((num_tokens, 1, hidden_dim))?;
            let gate = experts_moe_forward(&self.gate_experts, &xs, &indices)?;
            let up = experts_moe_forward(&self.up_experts, &xs, &indices)?;
            let activated = crate::ops::mul_and_act(&gate, &up, crate::layers::Activation::Silu)?;
            experts_moe_forward(&self.down_experts, &activated, &indices)?
        };
        ys.broadcast_mul(&scores.unsqueeze(D::Minus1)?)?
            .sum(D::Minus2)?
            .reshape((batch, seq_len, hidden_dim))?
            .to_dtype(original_dtype)
    }
}

enum MoeOrMlp {
    FusedMoe(FusedMoe),
    Mlp(Mlp),
}

impl MoeOrMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            Self::Mlp(m) => m.forward(xs),
            Self::FusedMoe(m) => m.forward(xs),
        }
    }
}

// SSM (Gated DeltaNet) layer weights for Qwen3.5/3.6 MoE hybrid architecture.
// Each SSM layer uses a causal conv1d for local mixing followed by a
// gated delta-rule recurrence that maintains a square state matrix per head.
struct SsmWeights {
    // fused QKV input projection [d_model, 2*inner_size] — after conv+silu
    // it is split into q [n_kv_heads*state_size], k [n_kv_heads*state_size],
    // v [n_heads*state_size]
    attn_qkv: Arc<dyn QuantMethod>,
    // output gating projection [d_model, inner_size]
    attn_gate: Arc<dyn QuantMethod>,
    // causal depthwise conv1d weight [kernel, 2*inner_size] stored as [kernel, channels]
    ssm_conv1d: Tensor,
    // per-head transition scalar (precomputed as -exp(A_log)), [n_heads]
    ssm_a: Tensor,
    // input→alpha projection [d_model, n_heads]
    ssm_alpha: Arc<dyn QuantMethod>,
    // input→beta projection [d_model, n_heads]
    ssm_beta: Arc<dyn QuantMethod>,
    // delta-time bias [n_heads]
    ssm_dt: Tensor,
    // output projection [inner_size, d_model]
    ssm_out: Arc<dyn QuantMethod>,
    // group-norm weight for gated output [state_size]
    ssm_norm: Tensor,
    n_heads: usize,       // num_v_heads = time_step_rank
    n_kv_heads: usize,    // num_k_heads = group_count
    state_size: usize,    // head_v_dim = head_k_dim
    inner_size: usize,
}

impl SsmWeights {
    fn forward(
        &self,
        x: &Tensor,
        conv_state: &Mutex<Option<Tensor>>,
        ssm_state: &Mutex<Option<Tensor>>,
    ) -> Result<Tensor> {
        // x: [batch, seq_len, d_model]
        let (_batch, seq_len, _d_model) = x.dims3()?;
        let _device = x.device();

        // 1. fused QKV projection → [batch, seq_len, 2*inner_size]
        let qkv = self.attn_qkv.forward(x)?;

        // 2. causal conv1d depthwise over channels
        let qkv = causal_conv1d_depthwise(&qkv, &self.ssm_conv1d, conv_state)?;

        // 3. SiLU activation
        let qkv = candle_nn::ops::silu(&qkv)?;

        // 4. split into q, k, v
        // q: [batch, seq_len, n_kv_heads * state_size]
        // k: [batch, seq_len, n_kv_heads * state_size]
        // v: [batch, seq_len, n_heads * state_size]
        let q_size = self.n_kv_heads * self.state_size;
        let k_size = self.n_kv_heads * self.state_size;
        let v_size = self.n_heads * self.state_size;
        let q = qkv.narrow(D::Minus1, 0, q_size)?;
        let k = qkv.narrow(D::Minus1, q_size, k_size)?;
        let v = qkv.narrow(D::Minus1, q_size + k_size, v_size)?;

        // reshape to [batch, seq_len, n_heads, state_size]
        let q = q.reshape((_batch, seq_len, self.n_kv_heads, self.state_size))?;
        let k = k.reshape((_batch, seq_len, self.n_kv_heads, self.state_size))?;
        let v = v.reshape((_batch, seq_len, self.n_heads, self.state_size))?;

        // 5. L2 normalize q and k
        let q = l2_norm(&q, 1e-6)?;
        let k = l2_norm(&k, 1e-6)?;

        // 6. repeat q/k from n_kv_heads to n_heads if they differ
        let (q, k) = if self.n_kv_heads != self.n_heads {
            let repeat = self.n_heads / self.n_kv_heads;
            let q = q
                .unsqueeze(D::Minus2)?
                .broadcast_as((
                    _batch,
                    seq_len,
                    self.n_kv_heads,
                    repeat,
                    self.state_size,
                ))?
                .reshape((_batch, seq_len, self.n_heads, self.state_size))?;
            let k = k
                .unsqueeze(D::Minus2)?
                .broadcast_as((
                    _batch,
                    seq_len,
                    self.n_kv_heads,
                    repeat,
                    self.state_size,
                ))?
                .reshape((_batch, seq_len, self.n_heads, self.state_size))?;
            (q, k)
        } else {
            (q, k)
        };

        // 7. alpha projection → gate (data-dependent decay)
        let alpha = self.ssm_alpha.forward(x)?; // [batch, seq_len, n_heads]
        let alpha = alpha.broadcast_add(&self.ssm_dt.reshape((1, 1, self.n_heads))?)?;
        let alpha = softplus(&alpha)?;
        let gate = alpha.broadcast_mul(&self.ssm_a.reshape((1, 1, self.n_heads))?)?;
        // gate = -exp(A_log) * softplus(alpha+dt), values in (-inf, 0]
        let alpha_decay = gate.exp()?; // exp(gate) ∈ (0, 1]

        // 8. beta projection (data-dependent gating)
        let beta = self.ssm_beta.forward(x)?; // [batch, seq_len, n_heads]
        let beta = candle_nn::ops::sigmoid(&beta)?;

        // 9. Gated DeltaNet recurrence
        let out = gated_delta_net_recurrence(
            &q,
            &k,
            &v,
            &alpha_decay,
            &beta,
            ssm_state,
            self.state_size,
            self.n_heads,
        )?;

        // 10. gated normalization: norm(output) * silu(gate_proj)
        let gate_proj = self.attn_gate.forward(x)?; // [batch, seq_len, inner_size]
        let gate_proj = gate_proj
            .reshape((_batch, seq_len, self.n_heads, self.state_size))?;
        let out = norm_gated(&out, &gate_proj, &self.ssm_norm)?;

        // 11. reshape to [batch, seq_len, inner_size] and output projection
        let out = out.reshape((_batch, seq_len, self.inner_size))?;
        let out = self.ssm_out.forward(&out)?; // → [batch, seq_len, d_model]
        Ok(out)
    }
}

/// Causal depthwise conv1d: each channel gets its own kernel applied over time.
/// conv_weight: [kernel_size, channels]
/// conv_state: stores last (kernel-1) frames as [kernel-1, channels]
fn causal_conv1d_depthwise(
    x: &Tensor,
    conv_weight: &Tensor,
    conv_state: &Mutex<Option<Tensor>>,
) -> Result<Tensor> {
    let (batch, seq_len, channels) = x.dims3()?;
    let kernel_size = conv_weight.dims()[0];
    let device = x.device();
    if kernel_size == 0 {
        return Ok(x.clone());
    }
    let mut state = conv_state.lock().unwrap();
    let pad_size = kernel_size - 1;
    if seq_len > 1 {
        // multi-token: do full conv1d via matrix ops
        // For simplicity, process each position sequentially for correctness
        let mut outputs = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let x_t = x.narrow(1, t, 1)?;
            let (_, _seq1, _) = x_t.dims3()?;
            let x_t = x_t.reshape((batch, channels))?;
            let mut accum = x_t.broadcast_mul(&conv_weight.narrow(0, 0, 1)?.reshape((1, channels))?)?;
            if let Some(ref prev) = *state {
                // prev: [pad_size, channels]
                // conv_weight[1..]: [pad_size, channels]
                for k in 1..kernel_size {
                    if k - 1 < pad_size {
                        let prev_t = prev.narrow(0, pad_size - (k - 1) - 1, 1)?;
                        let w_k = conv_weight.narrow(0, k, 1)?.reshape((1, channels))?;
                        accum = (accum + prev_t.broadcast_mul(&w_k)?)?;
                    }
                }
            }
            outputs.push(accum.reshape((batch, 1, channels))?);
            // update state: shift in x_t
            let x_t = x_t.reshape((1, channels))?;
            let new_state = if let Some(ref prev) = *state {
                Tensor::cat(&[prev.narrow(0, 1, pad_size - 1)?, x_t], 0)?
            } else {
                if pad_size > 1 {
                    let zeros = Tensor::zeros((pad_size - 1, channels), x_t.dtype(), &device)?;
                    Tensor::cat(&[zeros, x_t], 0)?
                } else {
                    x_t
                }
            };
            *state = Some(new_state);
        }
        Tensor::cat(&outputs.iter().map(|t| t.as_ref()).collect::<Vec<_>>(), 1)
    } else {
        // single token: use conv_state
        let x_t = x.reshape((batch, channels))?;
        let mut accum = x_t.broadcast_mul(&conv_weight.narrow(0, 0, 1)?.reshape((1, channels))?)?;
        if let Some(ref prev) = *state {
            for k in 1..kernel_size {
                let prev_t = prev.narrow(0, pad_size - k, 1)?;
                let w_k = conv_weight.narrow(0, k, 1)?.reshape((1, channels))?;
                accum = (accum + prev_t.broadcast_mul(&w_k)?)?;
            }
        }
        // update state
        let x_t_1d = x_t.reshape((1, channels))?;
        let new_state = if let Some(ref prev) = *state {
            if pad_size > 1 {
                Tensor::cat(&[prev.narrow(0, 1, pad_size - 1)?, x_t_1d], 0)?
            } else {
                x_t_1d
            }
        } else if pad_size > 1 {
            let zeros = Tensor::zeros((pad_size - 1, channels), x_t.dtype(), &device)?;
            Tensor::cat(&[zeros, x_t_1d], 0)?
        } else {
            x_t_1d
        };
        *state = Some(new_state);
        Ok(accum.reshape((batch, 1, channels))?)
    }
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    (Tensor::ones_like(x)? + x.exp()?)?.log()
}

// L2-normalize along the last dimension
fn l2_norm(x: &Tensor, eps: f64) -> Result<Tensor> {
    let norm = x.sqr()?.sum_keepdim(D::Minus1)?;
    let norm = norm.sqrt()?;
    let eps_tensor = Tensor::new(eps as f32, x.device())?.to_dtype(x.dtype())?;
    let norm = norm.broadcast_maximum(&eps_tensor)?;
    x.broadcast_div(&norm)
}

// Gated DeltaNet recurrence: per-head state update via outer product.
// q, k, v: [batch, seq_len, n_heads, state_size]
// alpha_decay, beta: [batch, seq_len, n_heads]
// ssm_state: [n_heads, state_size, state_size] (persistent across tokens)
fn gated_delta_net_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    alpha_decay: &Tensor,
    beta: &Tensor,
    ssm_state: &Mutex<Option<Tensor>>,
    state_size: usize,
    n_heads: usize,
) -> Result<Tensor> {
    let (batch, seq_len, _n_heads, _state_size) = q.dims4()?;
    let device = q.device();
    let mut state = ssm_state.lock().unwrap();

    if state.is_none() {
        *state = Some(Tensor::zeros(
            (n_heads, state_size, state_size),
            q.dtype(),
            &device,
        )?);
    }
    let mut s = state.clone().unwrap(); // [n_heads, state_size, state_size]

    let mut outputs = Vec::with_capacity(seq_len);
    for t in 0..seq_len {
        let q_t = q.narrow(1, t, 1)?.reshape((batch, n_heads, state_size))?; // [batch, n_heads, D]
        let k_t = k.narrow(1, t, 1)?.reshape((batch, n_heads, state_size))?;
        let v_t = v.narrow(1, t, 1)?.reshape((batch, n_heads, state_size))?;
        let alpha_t = alpha_decay.narrow(1, t, 1)?.reshape((batch, n_heads))?; // [batch, n_heads]
        let beta_t = beta.narrow(1, t, 1)?.reshape((batch, n_heads))?;

        // kv = S^T @ k_t  → [batch, n_heads, D]
        let mut kv_vecs = Vec::with_capacity(batch as usize);
        for b in 0..batch as usize {
            let s_h = s.clone(); // [n_heads, D, D]
            let k_bh = k_t.narrow(0, b, 1)?.reshape((n_heads, state_size))?; // [n_heads, D]
            // S^T @ k  for each head: (D,D)^T @ (D,) = (D,D) @ (D,) = (D,)
            let s_t = s_h.transpose(1, 2)?; // [n_heads, D, D]
            let kv = s_t
                .broadcast_matmul(&k_bh.unsqueeze(D::Minus1)?)? // [n_heads, D, 1]
                .reshape((1, n_heads, state_size))?;
            kv_vecs.push(kv);
        }
        let kv = Tensor::cat(&kv_vecs.iter().map(|t| t.as_ref()).collect::<Vec<_>>(), 0)?;

        // delta = beta_t * (v_t - kv)  → [batch, n_heads, D]
        let diff = v_t.sub(&kv)?;
        let delta = beta_t.unsqueeze(D::Minus1)?.broadcast_mul(&diff)?;

        // S = alpha_t * S + k_t ⊗ delta  (outer product update)
        // k_t ⊗ delta: for each head, outer(k_h, delta_h) → [D, D]
        let mut new_s_parts = Vec::with_capacity(batch as usize);
        for b in 0..batch as usize {
            let alpha_bh = alpha_t.narrow(0, b, 1)?.reshape((n_heads, 1, 1))?; // [n_heads,1,1]
            let k_bh = k_t.narrow(0, b, 1)?.reshape((n_heads, state_size))?; // [n_heads, D]
            let delta_bh = delta.narrow(0, b, 1)?.reshape((n_heads, state_size))?; // [n_heads, D]

            let outer = k_bh
                .unsqueeze(D::Minus1)?
                .broadcast_matmul(&delta_bh.unsqueeze(D::Minus2)?)?; // [n_heads, D, D]
            let s_new_b = s
                .broadcast_mul(&alpha_bh)?
                .add(&outer)?; // [n_heads, D, D]
            new_s_parts.push(s_new_b);
        }
        // For batched inference, use the last batch's state (single-sequence assumption)
        s = new_s_parts.last().unwrap().clone();

        // out_t = S_new @ q_t  → [batch, n_heads, D]
        let mut out_vecs = Vec::with_capacity(batch as usize);
        for b in 0..batch as usize {
            let s_b = s.clone(); // [n_heads, D, D]
            let q_bh = q_t.narrow(0, b, 1)?.reshape((n_heads, state_size))?;
            let out_b = s_b
                .broadcast_matmul(&q_bh.unsqueeze(D::Minus1)?)? // [n_heads, D, 1]
                .reshape((1, n_heads, state_size))?;
            out_vecs.push(out_b);
        }
        let out_t = Tensor::cat(
            &out_vecs.iter().map(|t| t.as_ref()).collect::<Vec<_>>(),
            0,
        )?;
        outputs.push(out_t.unsqueeze(1)?); // [batch, 1, n_heads, D]
    }

    *state = Some(s);
    Tensor::cat(
        &outputs.iter().map(|t| t.as_ref()).collect::<Vec<_>>(),
        1,
    )
}

// Gated normalization: group_norm(output) * silu(gate_proj)
fn norm_gated(output: &Tensor, gate: &Tensor, norm_weight: &Tensor) -> Result<Tensor> {
    // output, gate: [batch, seq_len, n_heads, state_size]
    // norm_weight: [state_size]
    let state_size = output.dims()[3];
    // RMS-norm over the last dimension
    let rms = output
        .sqr()?
        .mean_keepdim(D::Minus1)?
        .sqrt()?;
    let rms = rms.clamp(1e-6, f64::INFINITY)?;
    let normed = output.broadcast_div(&rms)?;
    let normed = normed.broadcast_mul(&norm_weight.reshape((1, 1, 1, state_size))?)?;
    let gate_act = candle_nn::ops::silu(gate)?;
    normed.broadcast_mul(&gate_act)
}

enum LayerWeights {
    Attention(AttentionWeights),
    Ssm {
        ssm: SsmWeights,
        conv_state: Mutex<Option<Tensor>>,
        ssm_state: Mutex<Option<Tensor>>,
        attention_norm: QRmsNorm,
        mlp: MoeOrMlp,
        ffn_norm: QRmsNorm,
    },
}

struct AttentionWeights {
    attention_wq: Arc<dyn QuantMethod>,
    attention_wk: Arc<dyn QuantMethod>,
    attention_wv: Arc<dyn QuantMethod>,
    attention_wo: Arc<dyn QuantMethod>,
    attention_norm: QRmsNorm,
    q_norm: QRmsNorm,
    k_norm: QRmsNorm,
    mlp: MoeOrMlp,
    ffn_norm: QRmsNorm,
    n_head: usize,
    n_kv_head: usize,
    head_dim: usize,
    rotary: Arc<RotaryEmbedding>,
    paged_attn: Option<PagedAttention>,
    sdpa_params: SdpaParams,
    dtype: DType,
}

impl AttentionWeights {
    fn forward_attn(
        &self,
        x: &Tensor,
        mask: &AttentionMask,
        start_offsets: &[usize],
        kv_cache: &mut KvCache,
        metadata: Option<((Tensor, Tensor), &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let (q, k, v) = if seq_len != 1 {
            let q = q
                .reshape((b_sz, seq_len, self.n_head, self.head_dim))?
                .transpose(1, 2)?;
            let k = k
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            let v = v
                .reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?
                .transpose(1, 2)?;
            (q, k, v)
        } else {
            let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
            let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            let v = v.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
            (q, k, v)
        };

        let positions =
            crate::pipeline::text_positions_tensor(start_offsets, q.dim(2)?, q.device())?;
        let (q, k) = self.rotary.forward_qk_norm(
            &q,
            &k,
            self.q_norm.weight(),
            self.k_norm.weight(),
            self.q_norm.eps(),
            self.k_norm.eps(),
            &positions,
        )?;

        let (q, k, v) = (
            q.to_dtype(self.dtype)?,
            k.to_dtype(self.dtype)?,
            v.to_dtype(self.dtype)?,
        );

        let y = match &self.paged_attn {
            Some(paged_attn) => {
                let ((key_cache, value_cache), input_metadata) = metadata.unwrap();
                paged_attn.forward(
                    &q,
                    &k,
                    &v,
                    mask,
                    Some(key_cache),
                    Some(value_cache),
                    input_metadata,
                    &self.sdpa_params,
                    None,
                )?
            }
            None => {
                let (k, v) = kv_cache.append(&k, &v)?;

                Sdpa.run_attention(&q, &k, &v, mask, None, &self.sdpa_params)?
            }
        };

        let y = if mask.is_custom() {
            y.transpose(1, 2)?.reshape((b_sz, seq_len, ()))?
        } else {
            y.reshape((b_sz, seq_len, ()))?
        };

        let y = self.attention_wo.forward(&y.to_dtype(x.dtype())?)?;
        Ok(y)
    }
}

pub struct ModelWeights {
    tok_embeddings: Embedding,
    layers: Vec<Option<LayerWeights>>,
    norm: QRmsNorm,
    output: Arc<dyn QuantMethod>,
    pub device: Device,
    pub cache: EitherCache,
    pub max_seq_len: usize,
    mapper: Option<Box<dyn DeviceMapper + Send + Sync>>,
    dtype: DType,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct QwenMoEConfig {
    pub moe_intermediate_size: usize,
    pub num_experts: Option<usize>,
    pub mlp_only_layers: Option<Vec<usize>>,
    pub decoder_sparse_step: Option<usize>,
    pub norm_topk_prob: bool,
    pub num_experts_per_tok: usize,
    pub ssm_inner_size: Option<usize>,
    pub ssm_state_size: Option<usize>,
    pub ssm_conv_kernel: Option<usize>,
    pub ssm_time_step_rank: Option<usize>,
    pub ssm_group_count: Option<usize>,
    pub full_attention_interval: Option<usize>,
}

pub(crate) struct PropsGGUF {
    pub head_count: usize,
    pub head_count_kv: usize,
    pub block_count: usize,
    pub embedding_length: usize,
    pub rms_norm_eps: f32,
    pub max_seq_len: usize,
    pub rope_freq_base: f32,
    pub key_length: usize,
    pub value_length: usize,
    pub moe_cfg: QwenMoEConfig,
    pub ssm_state_size: usize,
    pub ssm_time_step_rank: usize,
    pub ssm_inner_size: usize,
    pub ssm_group_count: usize,
    pub full_attention_interval: usize,
}

fn verify_qwen3_arch(
    metadata: &HashMap<String, candle_core::quantized::gguf_file::Value>,
) -> Result<String> {
    use crate::utils::gguf_metadata::TryValueInto;
    let actual_arch: String = metadata
        .get("general.architecture")
        .cloned()
        .try_value_into()?;

    if actual_arch != "qwen3" && actual_arch != "qwen3moe" && actual_arch != "qwen35moe" {
        candle_core::bail!("Expected `qwen3` architecture, got `{actual_arch}`.");
    }
    Ok(actual_arch)
}

impl TryFrom<ContentMetadata<'_>> for PropsGGUF {
    type Error = anyhow::Error;

    fn try_from(c: ContentMetadata) -> std::result::Result<Self, Self::Error> {
        let _ = verify_qwen3_arch(c.metadata)?;

        let required = [
            "attention.head_count",
            "attention.head_count_kv",
            "block_count",
            "embedding_length",
            "attention.layer_norm_rms_epsilon",
        ];
        c.has_required_keys(&required)?;

        let embed_len = c.get_value::<u32>("embedding_length")? as usize;
        let head_count = c.get_value::<u32>("attention.head_count")? as usize;

        // NOTE: Values are not aligned with GGUFv3 types
        // TODO: Normalize value types to spec

        let ssm_inner_size = c
            .get_value::<u32>("ssm.inner_size")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(0);
        let ssm_state_size = c
            .get_value::<u32>("ssm.state_size")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(0);
        let ssm_conv_kernel = c
            .get_value::<u32>("ssm.conv_kernel")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(0);
        let ssm_time_step_rank = c
            .get_value::<u32>("ssm.time_step_rank")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(0);
        let ssm_group_count = c
            .get_value::<u32>("ssm.group_count")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(0);
        let full_attention_interval = c
            .get_value::<u32>("full_attention_interval")
            .ok()
            .map(|x| x as usize)
            .unwrap_or(1);

        let moe_cfg = QwenMoEConfig {
            moe_intermediate_size: c.get_value::<u32>("expert_feed_forward_length")? as usize,
            num_experts: Some(c.get_value::<u32>("expert_count")? as usize),
            mlp_only_layers: Some(vec![]),
            decoder_sparse_step: Some(1),
            norm_topk_prob: true,
            num_experts_per_tok: c.get_value::<u32>("expert_used_count")? as usize,
            ssm_inner_size: if ssm_inner_size > 0 {
                Some(ssm_inner_size)
            } else {
                None
            },
            ssm_state_size: if ssm_state_size > 0 {
                Some(ssm_state_size)
            } else {
                None
            },
            ssm_conv_kernel: if ssm_conv_kernel > 0 {
                Some(ssm_conv_kernel)
            } else {
                None
            },
            ssm_time_step_rank: if ssm_time_step_rank > 0 {
                Some(ssm_time_step_rank)
            } else {
                None
            },
            ssm_group_count: if ssm_group_count > 0 {
                Some(ssm_group_count)
            } else {
                None
            },
            full_attention_interval: if full_attention_interval > 1 {
                Some(full_attention_interval)
            } else {
                None
            },
        };

        let props = Self {
            head_count,
            head_count_kv: c.get_value::<u32>("attention.head_count_kv")? as usize,
            block_count: c.get_value::<u32>("block_count")? as usize,
            embedding_length: embed_len,
            rms_norm_eps: c.get_value("attention.layer_norm_rms_epsilon")?,
            max_seq_len: c
                .get_value::<u64>("context_length")
                .ok()
                .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize,
            rope_freq_base: c.get_value("rope.freq_base").ok().unwrap_or(10_000_f32),
            key_length: c
                .get_value::<u32>("attention.key_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            value_length: c
                .get_value::<u32>("attention.value_length")
                .ok()
                .map(|x| x as usize)
                .unwrap_or(embed_len / head_count),
            moe_cfg,
            ssm_state_size,
            ssm_time_step_rank,
            ssm_inner_size,
            ssm_group_count,
            full_attention_interval,
        };

        Ok(props)
    }
}

impl ModelConfig::FromGGUF for ModelWeights {
    fn from_gguf<R: std::io::Seek + std::io::Read>(
        mut ct: Content<'_, R>,
        device: &Device,
        mapper: Box<dyn DeviceMapper + Send + Sync>,
        attention_mechanism: AttentionImplementation,
        dtype: DType,
    ) -> Result<Self> {
        // Parameter extraction from metadata.
        let meta = ct.get_metadata();
        let actual_arch = verify_qwen3_arch(meta)?;

        let metadata = ContentMetadata {
            path_prefix: &actual_arch,
            metadata: meta,
        };
        let PropsGGUF {
            head_count,
            head_count_kv,
            block_count,
            embedding_length,
            rms_norm_eps,
            max_seq_len,
            rope_freq_base,
            key_length,
            value_length,
            moe_cfg,
            ssm_state_size,
            ssm_time_step_rank,
            ssm_inner_size,
            ssm_group_count,
            full_attention_interval,
        } = PropsGGUF::try_from(metadata).or_else(|err| candle_core::bail!("{err}"))?;

        let qtok_embeddings = ct.tensor("token_embd.weight", device)?;
        let tok_embeddings = qtok_embeddings.dequantize(device)?;
        let norm = QRmsNorm::new(ct.tensor("output_norm.weight", device)?, rms_norm_eps)?;
        let output = if !ct.has_tensor("output.weight") {
            ct.tensor("token_embd.weight", device)?
        } else {
            ct.tensor("output.weight", device)?
        };
        let mut layers = Vec::with_capacity(block_count);

        let head_dim = key_length;
        if key_length != value_length {
            candle_core::bail!(
                "Expected key_length == value_length, got {key_length} != {value_length}"
            );
        }

        let mut ropes = HashMap::new();
        for layer_idx in 0..block_count {
            let layer_dev = mapper.device_for(layer_idx, false).unwrap_or(device);
            let rope_dev: &Device = if layer_dev.location() == device.location() {
                device
            } else {
                layer_dev
            };
            ropes.insert(
                rope_dev.location(),
                Arc::new(RotaryEmbedding::new(
                    rope_freq_base,
                    head_dim,
                    max_seq_len,
                    rope_dev,
                    true,
                    DType::F32,
                )?),
            );
        }

        for layer_idx in NiceProgressBar::<_, 'b'>(
            0..block_count,
            "Loading repeating layers",
            &new_multi_progress(),
        ) {
            let prefix = format!("blk.{layer_idx}");
            if mapper.is_layer_remote(layer_idx) {
                layers.push(None);
                continue;
            }
            let device = mapper.device_for(layer_idx, false).unwrap_or(device);

            let is_ssm = full_attention_interval > 0
                && ssm_inner_size > 0
                && ct.has_tensor(&format!("{prefix}.ssm_conv1d.weight"));

            let layer = if is_ssm {
                let inner_size = ssm_inner_size;
                let state_size = ssm_state_size;
                let n_heads = ssm_time_step_rank;
                let n_kv_heads = ssm_group_count;

                let attn_qkv = ct.tensor(&format!("{prefix}.attn_qkv.weight"), device)?;
                let attn_gate = ct.tensor(&format!("{prefix}.attn_gate.weight"), device)?;
                let ssm_conv1d = ct.tensor(&format!("{prefix}.ssm_conv1d.weight"), device)?
                    .dequantize(device)?;
                let ssm_a = ct.tensor(&format!("{prefix}.ssm_a"), device)?
                    .dequantize(device)?;
                let ssm_dt = ct.tensor(&format!("{prefix}.ssm_dt.bias"), device)?
                    .dequantize(device)?;
                let ssm_alpha = ct.tensor(&format!("{prefix}.ssm_alpha.weight"), device)?;
                let ssm_beta = ct.tensor(&format!("{prefix}.ssm_beta.weight"), device)?;
                let ssm_out = ct.tensor(&format!("{prefix}.ssm_out.weight"), device)?;
                let ssm_norm = ct.tensor(&format!("{prefix}.ssm_norm.weight"), device)?
                    .dequantize(device)?;

                // SSM layers: MoE FFN with shared expert
                let ssm_mlp = {
                    let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                    let gate_experts =
                        ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?;
                    let up_experts =
                        ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;
                    let down_experts =
                        ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
                    MoeOrMlp::FusedMoe(FusedMoe {
                        gate: QMatMul::from_qtensor(gate)?,
                        gate_experts: QMatMul::from_qtensor(gate_experts)?,
                        up_experts: QMatMul::from_qtensor(up_experts)?,
                        down_experts: QMatMul::from_qtensor(down_experts)?,
                        norm_topk_prob: moe_cfg.norm_topk_prob,
                        num_experts_per_tok: moe_cfg.num_experts_per_tok,
                    })
                };
                // Qwen3.5/3.6 uses post_attention_norm instead of ffn_norm
                let ssm_attn_norm =
                    ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
                let ssm_ffn_norm = if ct.has_tensor(&format!("{prefix}.post_attention_norm.weight"))
                {
                    ct.tensor(&format!("{prefix}.post_attention_norm.weight"), device)?
                } else {
                    ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?
                };

                let ssm = SsmWeights {
                    attn_qkv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attn_qkv),
                        b: None,
                    })?),
                    attn_gate: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attn_gate),
                        b: None,
                    })?),
                    ssm_conv1d,
                    ssm_a,
                    ssm_alpha: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(ssm_alpha),
                        b: None,
                    })?),
                    ssm_beta: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(ssm_beta),
                        b: None,
                    })?),
                    ssm_dt,
                    ssm_out: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(ssm_out),
                        b: None,
                    })?),
                    ssm_norm,
                    n_heads,
                    n_kv_heads,
                    state_size,
                    inner_size,
                };
                LayerWeights::Ssm {
                    ssm,
                    conv_state: Mutex::new(None),
                    ssm_state: Mutex::new(None),
                    attention_norm: QRmsNorm::new(ssm_attn_norm, rms_norm_eps)?,
                    mlp: ssm_mlp,
                    ffn_norm: QRmsNorm::new(ssm_ffn_norm, rms_norm_eps)?,
                }
            } else {
                let rotary = ropes
                    .get(&device.location())
                    .expect("No RoPE for device location!")
                    .clone();

                let attention_wq = ct.tensor(&format!("{prefix}.attn_q.weight"), device)?;
                let attention_wk = ct.tensor(&format!("{prefix}.attn_k.weight"), device)?;
                let attention_wv = ct.tensor(&format!("{prefix}.attn_v.weight"), device)?;
                let attention_wo = ct.tensor(&format!("{prefix}.attn_output.weight"), device)?;

                let mlp = if !moe_cfg
                    .mlp_only_layers
                    .as_ref()
                    .unwrap()
                    .contains(&layer_idx)
                    && (moe_cfg.num_experts.unwrap() > 0
                        && (layer_idx + 1) % moe_cfg.decoder_sparse_step.unwrap() == 0)
                {
                    let gate = ct.tensor(&format!("{prefix}.ffn_gate_inp.weight"), device)?;
                    let gate_experts =
                        ct.tensor(&format!("{prefix}.ffn_gate_exps.weight"), device)?;
                    let up_experts =
                        ct.tensor(&format!("{prefix}.ffn_up_exps.weight"), device)?;
                    let down_experts =
                        ct.tensor(&format!("{prefix}.ffn_down_exps.weight"), device)?;
                    let moe = FusedMoe {
                        gate: QMatMul::from_qtensor(gate)?,
                        gate_experts: QMatMul::from_qtensor(gate_experts)?,
                        up_experts: QMatMul::from_qtensor(up_experts)?,
                        down_experts: QMatMul::from_qtensor(down_experts)?,
                        norm_topk_prob: moe_cfg.norm_topk_prob,
                        num_experts_per_tok: moe_cfg.num_experts_per_tok,
                    };
                    MoeOrMlp::FusedMoe(moe)
                } else {
                    let feed_forward_w1 =
                        ct.tensor(&format!("{prefix}.ffn_gate.weight"), device)?;
                    let feed_forward_w2 =
                        ct.tensor(&format!("{prefix}.ffn_down.weight"), device)?;
                    let feed_forward_w3 =
                        ct.tensor(&format!("{prefix}.ffn_up.weight"), device)?;
                    let mlp = Mlp {
                        feed_forward_w1: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(feed_forward_w1),
                            b: None,
                        })?),
                        feed_forward_w2: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(feed_forward_w2),
                            b: None,
                        })?),
                        feed_forward_w3: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                            q_weight: Arc::new(feed_forward_w3),
                            b: None,
                        })?),
                    };
                    MoeOrMlp::Mlp(mlp)
                };

                // Qwen3 always has q_norm and k_norm
                let q_norm = QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.attn_q_norm.weight"), device)?,
                    rms_norm_eps,
                )?;
                let k_norm = QRmsNorm::new(
                    ct.tensor(&format!("{prefix}.attn_k_norm.weight"), device)?,
                    rms_norm_eps,
                )?;

                let attention_norm = ct.tensor(&format!("{prefix}.attn_norm.weight"), device)?;
                let ffn_norm = ct.tensor(&format!("{prefix}.ffn_norm.weight"), device)?;
                let paged_attn = match &attention_mechanism {
                    AttentionImplementation::Eager => None,
                    AttentionImplementation::PagedAttention => {
                        Some(PagedAttention::new(head_dim, device, None)?)
                    }
                };
                LayerWeights::Attention(AttentionWeights {
                    attention_wq: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wq),
                        b: None,
                    })?),
                    attention_wk: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wk),
                        b: None,
                    })?),
                    attention_wv: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wv),
                        b: None,
                    })?),
                    attention_wo: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                        q_weight: Arc::new(attention_wo),
                        b: None,
                    })?),
                    attention_norm: QRmsNorm::new(attention_norm, rms_norm_eps)?,
                    q_norm,
                    k_norm,
                    mlp,
                    ffn_norm: QRmsNorm::new(ffn_norm, rms_norm_eps)?,
                    n_head: head_count,
                    n_kv_head: head_count_kv,
                    head_dim,
                    rotary: rotary.clone(),
                    paged_attn,
                    sdpa_params: SdpaParams {
                        n_kv_groups: head_count / head_count_kv,
                        softcap: None,
                        softmax_scale: 1.0 / (head_dim as f32).sqrt(),
                        sliding_window: None,
                        sinks: None,
                    },
                    dtype,
                })
            };
            layers.push(Some(layer))
        }
        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            layers,
            norm,
            output: Arc::new(GgufMatMul::new(QuantMethodConfig::Gguf {
                q_weight: Arc::new(output),
                b: None,
            })?),
            device: device.clone(),
            cache: EitherCache::Normal(NormalCache::new(block_count, max_seq_len)),
            max_seq_len,
            mapper: Some(mapper),
            dtype,
        })
    }
}

impl ModelWeights {
    pub fn forward(
        &self,
        x: &Tensor,
        start_offsets: &[usize],
        context_lens: Vec<(usize, usize)>,
        metadata: Option<(Vec<(Tensor, Tensor)>, &PagedAttentionInputMetadata)>,
    ) -> Result<Tensor> {
        let mut layer_in = self.tok_embeddings.forward(x)?;
        let cache = &mut self.cache.normal().0;
        let mask = CausalMasker.make_causal_mask(
            x,
            metadata
                .as_ref()
                .map(|(_, _)| &start_offsets as &dyn PastKvLenCache)
                .unwrap_or(cache as &dyn PastKvLenCache),
            self.dtype,
            &CausalMaskConfig::default(),
        )?;
        let mask = if metadata
            .as_ref()
            .map(|(_, meta)| meta.is_first_prompt_chunk)
            .unwrap_or(true)
        {
            mask
        } else {
            AttentionMask::None
        };
        let mask = if let Some(ref mapper) = self.mapper {
            DeviceMappedMask::new(mask, &**mapper)?
        } else {
            DeviceMappedMask::from_single(mask)
        };
        if let Some(ref mapper) = self.mapper {
            let past_kv = start_offsets.first().copied().unwrap_or(0) as u32;
            mapper.set_past_kv(past_kv);
        }
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(ref mapper) = self.mapper {
                layer_in = mapper.map(layer_in, i)?;
            }
            let layer = match layer {
                Some(l) => l,
                None => continue,
            };
            let x = layer_in;
            let residual = &x;
            let attn = match layer {
                LayerWeights::Attention(attn) => {
                    let x = attn.attention_norm.forward(&x)?;
                    attn.forward_attn(
                        &x,
                        &mask.get(x.device()),
                        start_offsets,
                        &mut cache[i],
                        metadata
                            .as_ref()
                            .map(|(kv_cache, metadata)| (kv_cache[i].clone(), *metadata)),
                    )?
                }
                LayerWeights::Ssm {
                    ssm,
                    ref conv_state,
                    ref ssm_state,
                    ref attention_norm,
                    ..
                } => {
                    let x = attention_norm.forward(&x)?;
                    ssm.forward(&x, conv_state, ssm_state)?
                }
            };
            let x = (attn + residual)?;

            // FFN
            let residual = &x;
            let x = match layer {
                LayerWeights::Attention(attn) => {
                    let x = attn.ffn_norm.forward(&x)?;
                    let x = attn.mlp.forward(&x)?;
                    (x + residual)?
                }
                LayerWeights::Ssm {
                    ref ffn_norm,
                    ref mlp,
                    ..
                } => {
                    let x = ffn_norm.forward(&x)?;
                    let x = mlp.forward(&x)?;
                    (x + residual)?
                }
            };
            layer_in = x;
        }
        let x = self.norm.forward(&layer_in)?;
        let x = extract_logits(&x, context_lens)?;
        self.output.forward(&x.contiguous()?)
    }

    #[allow(clippy::too_many_arguments, dead_code)]
    pub fn forward_from_layer(
        &self,
        hidden: &Tensor,
        start_layer: usize,
        end_layer: usize,
        past_kv_len: usize,
        cache: &mut [KvCache],
    ) -> Result<Tensor> {
        let mut layer_in = hidden.to_device(&self.device)?;
        let seq_len = hidden.dims()[1];
        let kv_offsets = [past_kv_len];
        let kv_ref: &[usize] = &kv_offsets;
        let kv_len_cache: &dyn PastKvLenCache = &kv_ref;
        let mask = if seq_len > 1 {
            let dummy_ids = Tensor::zeros((1, seq_len), DType::U32, &self.device)?;
            let attention_mask = CausalMasker.make_causal_mask(
                &dummy_ids,
                kv_len_cache,
                self.dtype,
                &CausalMaskConfig::default(),
            )?;
            DeviceMappedMask::from_single(attention_mask)
        } else {
            DeviceMappedMask::from_single(AttentionMask::None)
        };
        for (i, layer) in self.layers.iter().enumerate() {
            if i < start_layer || i > end_layer {
                continue;
            }
            let layer = match layer {
                Some(l) => l,
                None => continue,
            };
            let x = layer_in;
            let residual = &x;
            let attn = match layer {
                LayerWeights::Attention(attn) => {
                    let x = attn.attention_norm.forward(&x)?;
                    attn.forward_attn(
                        &x,
                        &mask.get(x.device()),
                        &[past_kv_len],
                        &mut cache[i],
                        None,
                    )?
                }
                LayerWeights::Ssm {
                    ssm,
                    ref conv_state,
                    ref ssm_state,
                    ref attention_norm,
                    ..
                } => {
                    let x = attention_norm.forward(&x)?;
                    ssm.forward(&x, conv_state, ssm_state)?
                }
            };
            let x = (attn + residual)?;
            let residual = &x;
            layer_in = match layer {
                LayerWeights::Attention(attn) => {
                    let x = attn.ffn_norm.forward(&x)?;
                    let x = attn.mlp.forward(&x)?;
                    (x + residual)?
                }
                LayerWeights::Ssm {
                    ref ffn_norm,
                    ref mlp,
                    ..
                } => {
                    let x = ffn_norm.forward(&x)?;
                    let x = mlp.forward(&x)?;
                    (x + residual)?
                }
            };
        }
        layer_in.to_device(&Device::Cpu)
    }
}
