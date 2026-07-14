# 4. Lensing architectures that aren't transformers

> The workspace is *clearest* in the models `jlens` refuses to load. That is not a
> coincidence, and it is the most interesting thing we found.

## The models that matter are hybrids

The standing objection to this whole research programme is architectural. Butlin & Long:
transformers have *"no obviously separable input processors."* Erik Hoel: LLMs *"flatly
lack"* modularity and reentrant dynamics. The deflationary reading of the J-space is that
it is an artifact of transformer topology — a shared additive residual stream that you
then rediscover and name after a theory of consciousness.

So look at where the phenomenon actually lives.

**Qwen3.5-27B** — the model where the flagship ASCII-face readout **emerges** (rank 2 for
"nose", versus 164 at Qwen3-14B):

```
layer_types: {'linear_attention': 48, 'full_attention': 16}   # 64 layers
full_attention_interval: 4
linear_conv_kernel_dim: 4
```

**Only 16 of 64 layers are softmax attention.** The other 48 are gated linear attention
with a convolution kernel — an SSM-class mechanism.

**Nemotron-H** — pushing further:

```
{'NemotronHMamba2Mixer': 21, 'NemotronHMLP': 17, 'NemotronHAttention': 4}
```

21 Mamba-2 mixers. Four attention layers.

If a J-space appears in these, then "global workspace" is not a property of softmax
attention — and the architectural objection is simply wrong. **You cannot test that with
a library that refuses to load them.**

## Layouts: it's five fields, not an incompatibility

```
ValueError: could not locate the text decoder inside NemotronHForCausalLM
            (tried 6 known layouts); pass layout= explicitly
```

That is a *naming* mismatch. Nemotron-H calls its embedding `embeddings` and its final
norm `norm_f`:

```python
from jlens_lab import from_hf, describe

describe(hf)      # {'Mamba2Mixer': 21, 'MLP': 17, 'Attention': 4}
model = from_hf(hf, tok)     # drop-in; registry first, jlens auto-detect as fallback
```

`describe()` is worth calling on anything before you lens it. "Transformer" is doing less
work than you think in most recent models.

## The wall: SSM Jacobians are not affordable on the naive path

Measured on **Nemotron-H-4B**, A100-80GB, without fused kernels:

```
dim_batch= 8  seq= 64    peak 73.1 GB   OK     <- ONE layer. Of a 4B model.
dim_batch=16  seq=128    OOM
3 layers @ 8/64          OOM (needed 24 GiB more on top of 62 GiB)
```

A 4B **transformer** needs a few GB for the same work.

Why: without NVIDIA's fused kernels, Mamba-2 falls back to a naive scan, and the backward
graph holds the **fully unrolled recurrence** across the sequence. `jacobian_for_prompt`
then calls `autograd.grad` once per `dim_batch` chunk of `d_model` against that graph.

```
[transformers] The fast path is not available because one of
(selective_state_update, causal_conv1d_fn, causal_conv1d_update) is None.
Falling back to the naive implementation.
```

**Install the fused kernels. They are not an optimisation, they are the only way this
runs.** With them, the same computation on the same GPU:

```
                         naive        fused
dim_batch= 8 seq= 64     73.1 GB  ->   9.0 GB     8x
dim_batch=16 seq=128     OOM      ->   9.5 GB
dim_batch=32 seq=128     OOM      ->  11.1 GB
```

From "impossible on any GPU that exists" to "comfortable on a mid-range card."

And the lens works. Nemotron-H-4B (21 Mamba-2 mixers, 4 attention layers), prompt
*"The capital of France is"*:

```
L10 top5: ['.\",', '.`', 'Berlin', 'mars', '.\"\n\n']
L20 top5: ['mars', 'officielle', 'administratives', 'ONU', 'Marte']
L30 top5: ['Paris', 'France', 'Paris', 'Marseille', 'Louvre']
```

A coherent semantic cluster at layer 30 -- the capital, the country, another French city,
a Paris landmark -- in a model that is overwhelmingly **not** a transformer. The J-space is
not a fact about softmax attention.

> I initially assumed the *opposite* — that the naive path was safer, because fused
> kernels might not support the backward the Jacobian needs. Wrong on both counts: `jlens`
> needs only a **first-order** gradient, and the fused kernels ship a proper custom
> backward. That wrong assumption is what OOM'd a 119GB machine.

```python
# CUDA *devel* base -- these compile at install.
# mamba-ssm compiles against torch.version.cuda, so the base image CUDA and the torch
# wheel CUDA must MATCH. Default `pip install torch` pulls cu130; against a 12.8 base it
# dies with "detected CUDA version (12.8) mismatches ... PyTorch (13.0)".
image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "build-essential", "g++")          # base ships clang++; torch refuses it
    .env({"CC": "gcc", "CXX": "g++", "TORCH_CUDA_ARCH_LIST": "8.0"})
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install("causal-conv1d>=1.4.0", "mamba-ssm>=2.2.2",
                 extra_options="--no-build-isolation")
)
```

And **assert the fast path is live** before you spend anything:

```python
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from causal_conv1d import causal_conv1d_fn      # ImportError here == a 73GB/layer run
```

## Run it somewhere it can die alone

On a unified-memory machine (DGX Spark / GB10), **a GPU OOM is a SYSTEM OOM.** It does not
raise `torch.OutOfMemoryError` — it starves the host. Ours went unreachable: kernel alive,
`sshd` dead, other users' jobs destroyed, physical power cycle required.

The same failure on a rented A100 is a clean `torch.OutOfMemoryError` telling you it tried
to allocate 24.00 GiB, in a container that dies by itself, for about a dollar. That is not
merely safer — it is *diagnosable*.

Budget memory before you run:

| model class | why it is expensive |
|---|---|
| dense transformer | one backward per `dim_batch` chunk of `d_model`, plus a `d_model²` accumulator per layer |
| **Mamba / SSM** | **the recurrent scan's autograd graph is unrolled over the sequence** — start at `dim_batch=8`, `max_seq_len=64`, and only scale up on measurements |
| MoE | all experts materialise in bf16 regardless of active params (`Nemotron-3-Nano-30B-A3B` is ~60GB, not ~6GB) |
