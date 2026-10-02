#!/usr/bin/env python3
"""
Export Mamba2 intermediate tensors for the mamba2-explainer visualizer.

Usage:
    pip install transformers torch numpy
    python scripts/export_mamba2_tensors.py --model AntonV/mamba2-130m-hf \
        --out tools/mamba2-explainer/static/data

Note: 'state-spaces/mamba2-130m' uses the original mamba-ssm checkpoint format
and cannot be loaded with Mamba2ForCausalLM from transformers. Use a community
HF-converted repo such as AntonV/mamba2-130m-hf instead.

Output structure:
    static/data/manifest.json          -- list of all exports + outputs section
    static/data/model_meta.json        -- model config + per-layer/head decay A
    static/data/<prompt_id>_l<layer>_s<step>.npz  -- per-step tensors
    static/data/<prompt_id>_output.npz  -- per-step next-token distribution (top-50)

Per (layer, step) npz tensors:
    hidden_in   {d_model}             input to the layer (embeddings for layer 0)
    hidden_out  {d_model}             layer output after residual add
    proj_out    {d_in_proj}           W_in projection output, split z | xBC | dt
    conv_state  {conv_dim, d_conv}    r_l conv window after this token,
                                      columns [x_{t-d_conv+1} .. x_t], zero-padded at start
    ssm_state   {n_head, 16, 16}      s_l after this token, strided subsample of the
                                      full {n_head, head_dim, d_state} recurrent state

Per prompt output npz (one per prompt; the distribution is a property of the
whole stack -- final RMS norm -> lm_head -> logits -> softmax -- so it is not
duplicated into the per-layer files):
    probs       {T,50} f32            top-50 softmax probabilities per step
    ids         {T,50} f32            top-50 token ids per step (exact below 2^24)
    entropy     {T} f32               entropy of full softmax distribution per step
    hidden      {T,768} f32           final-RMS-normed hidden state per step
    embeddings  {T,768} f32           layer-0 input (token embedding) per step
    actual_rank {T} int32             rank of the actual next prompt token (1..50, 0 if absent)
    actual_prob {T} f32               softmax prob of the actual next prompt token (0 if absent)

manifest.json "outputs" section (keyed by prompt id) carries the display
strings: per step the top-20 token strings, the actual next token string and
its rank, so the frontend never has to decode token ids itself.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away",
    "The quick brown fox jumps over",
    "In machine learning, gradient descent",
    "The Mamba2 model uses state space",
]

# ---------------------------------------------------------------------------
# Tensor capture
# ---------------------------------------------------------------------------

def _f32(t: torch.Tensor) -> np.ndarray:
    # clone: the cache buffers are updated in place, a numpy view would alias them
    return t.detach().cpu().float().clone().numpy()


# Display decimation of the recurrent state: the raw s_l is {n_head, head_dim,
# d_state} (786 KB per layer-step in f32); a strided subsample keeps the panel
# honest about the structure while staying inside the data budget.
SSM_SUB_HEAD_DIM = 16
SSM_SUB_D_STATE = 16


# ---------------------------------------------------------------------------
# Minimal Mamba2 forward with captures
# ---------------------------------------------------------------------------

# state-spaces/mamba2-*-hf repos do not bundle a tokenizer -- they use GPT-NeoX.
TOKENIZER_FALLBACK = "EleutherAI/gpt-neox-20b"


def load_model(model_name: str):
    try:
        from transformers import AutoTokenizer, Mamba2ForCausalLM
    except ImportError:
        print("ERROR: transformers >= 4.39 with Mamba2 support required.")
        print("  pip install 'transformers>=4.39' torch")
        sys.exit(1)

    print(f"Loading {model_name} ...")

    # state-spaces/mamba2-* repos do not bundle a tokenizer.
    # Try the model repo; on failure use the known-good GPT-NeoX tokenizer.
    tokenizer = None
    for tok_name in (model_name, TOKENIZER_FALLBACK):
        for fast in (True, False):
            try:
                tokenizer = AutoTokenizer.from_pretrained(tok_name, use_fast=fast)
                if tok_name != model_name:
                    print(f"  Tokenizer loaded from fallback: {tok_name}")
                break
            except Exception as exc:
                last_exc = exc
    if tokenizer is None:
        print(f"ERROR: Could not load tokenizer: {last_exc}")
        print("  pip install sentencepiece")
        sys.exit(1)

    model = Mamba2ForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        ignore_mismatched_sizes=False,
    )
    model.eval()
    return tokenizer, model


def run_forward_token_by_token(model, tokenizer, prompt: str):
    """Run one decode step per token and capture per-layer tensors.

    Decoding token by token is what makes the recurrent state observable:
    after each step the cache holds the real conv window r_l, exactly like
    llama.cpp's recurrent memory does during generation.

    Returns: (tokens, captures, n_layers, outputs)
        tokens: list of token strings
        captures: [layer][step] -> dict of np arrays
        outputs: dict of per-step output-distribution arrays and strings:
            probs       list of {50} f32, top-50 softmax probs
            ids         list of {50} f32, top-50 token ids
            top_tokens  list of lists of 20 strings, display strings for top-20
            entropy     list of scalars, entropy of the full softmax dist
            hidden      list of {d_model} f32, final-RMS-normed hidden
            embeddings  list of {d_model} f32, layer-0 input embedding
            actual_next list of strings, the prompt token that followed (or '')
            actual_rank list of ints, its 1-based rank in the top-50 (0 if absent)
            actual_prob list of floats, its softmax prob (0 if absent)
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # {1, T}
    T = input_ids.shape[1]

    layers = model.backbone.layers
    n_layers = len(layers)

    captures: list[list[dict]] = [
        [{} for _ in range(T)] for _ in range(n_layers)
    ]

    # Per-prompt output buffers (top-50 tokens + entropy + final-normed hidden)
    # One npz per prompt, not per-layer: distribution is a property of the whole stack.
    outputs = {
        "probs": [],       # list of {50} f32
        "ids": [],         # list of {50} f32
        "top_tokens": [],  # list of lists of 20 display strings
        "entropy": [],     # list of scalars
        "hidden": [],      # list of {d_model} f32, final-RMS-normed
        "embeddings": [],  # list of {d_model} f32, layer-0 input
        "actual_next": [], # list of strings ('' at the final step)
        "actual_rank": [], # list of ints (1-based, 0 if not in top-50)
        "actual_prob": [], # list of floats
    }

    cache = None
    with torch.no_grad():
        for t in range(T):
            out = model(
                input_ids[:, t : t + 1],
                cache_params=cache,
                use_cache=True,
                output_hidden_states=True,
            )
            cache = out.cache_params

            for lidx, layer in enumerate(layers):
                c = captures[lidx][t]

                # hidden_states[0] is the embedding output, hidden_states[l + 1]
                # the output of layer l, so hidden_states[l] is layer l's input
                h_in = out.hidden_states[lidx][0, -1]
                c["hidden_in"] = _f32(h_in)
                c["hidden_out"] = _f32(out.hidden_states[lidx + 1][0, -1])
                c["proj_out"] = _f32(layer.mixer.in_proj(h_in.unsqueeze(0))[0])

                layer_cache = cache.layers[lidx]
                c["conv_state"] = _f32(layer_cache.conv_states[0][0])

                # s_l after this token: {n_head, head_dim, d_state}, strided
                # subsample on the last two axes for display
                s_full = layer_cache.recurrent_states[0][0]
                step_h = max(1, s_full.shape[1] // SSM_SUB_HEAD_DIM)
                step_d = max(1, s_full.shape[2] // SSM_SUB_D_STATE)
                c["ssm_state"] = _f32(s_full[:, ::step_h, ::step_d])

            # --- capture next-token distribution from the LAST layer's logits ---
            # out.logits has shape {1, 1, V} for the single generated token at position t
            logits = out.logits[0, -1]  # {V}
            probs = torch.softmax(logits, dim=-1)  # {V}
            topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)  # {50}

            # Entropy of the full distribution
            log_probs = torch.log(probs + 1e-30)  # {V}
            entropy = -(probs * log_probs).sum().item()  # scalar

            # Actual next-token rank+prob (if a prompt token follows)
            actual_rank = 0
            actual_prob = 0.0
            actual_next_str = ""
            if t < T - 1:
                next_token_id = input_ids[0, t + 1].item()
                # find rank of next_token_id in topk_indices
                matches = (topk_indices == next_token_id).nonzero(as_tuple=False)
                if matches.shape[0] > 0:
                    actual_rank = matches[0].item() + 1  # 1-indexed
                    actual_prob = topk_probs[matches[0][0]].item()

            # Data-flow tail: final layer output -> final RMS norm -> lm_head.
            # hidden_states[-1] is pre-norm; norm_f is what lm_head consumes.
            final_normed = model.backbone.norm_f(out.hidden_states[-1][0, -1])  # {d_model}

            outputs["probs"].append(topk_probs.detach().cpu().float().numpy())
            outputs["ids"].append(topk_indices.detach().cpu().float().numpy())
            outputs["entropy"].append(entropy)
            outputs["hidden"].append(_f32(final_normed))
            outputs["embeddings"].append(_f32(out.hidden_states[0][0, -1]))
            outputs["actual_rank"].append(actual_rank)
            outputs["actual_prob"].append(actual_prob)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Display strings for the top-20 of each step + the actual next token
    for t in range(T):
        ids_t = outputs["ids"][t].astype(np.int64)
        outputs["top_tokens"].append(tokenizer.convert_ids_to_tokens(ids_t[:20].tolist()))
        outputs["actual_next"].append(tokens[t + 1] if t < T - 1 else "")

    return tokens, captures, n_layers, outputs


# ---------------------------------------------------------------------------
# Serialize to npz + manifest
# ---------------------------------------------------------------------------

def export_prompt(
    model,
    tokenizer,
    prompt: str,
    prompt_id: str,
    out_dir: Path,
    manifest: list,
    outputs_section: dict,
):
    print(f"  Prompt {prompt_id}: {prompt[:50]!r}")
    tokens, captures, n_layers, outputs = run_forward_token_by_token(model, tokenizer, prompt)
    T = len(tokens)

    # Write one output npz per prompt (top-50 distribution, not per-layer)
    output_npz_path = out_dir / f"{prompt_id}_output.npz"
    np.savez_compressed(str(output_npz_path),
                        probs=np.stack(outputs["probs"]),      # {T,50} f32
                        ids=np.stack(outputs["ids"]),           # {T,50} f32
                        entropy=np.array(outputs["entropy"], dtype=np.float32),   # {T} f32
                        hidden=np.stack(outputs["hidden"]),     # {T,768} f32
                        embeddings=np.stack(outputs["embeddings"]),  # {T,768} f32
                        actual_rank=np.array(outputs["actual_rank"], dtype=np.int32),   # {T} int32
                        actual_prob=np.array(outputs["actual_prob"], dtype=np.float32)  # {T} f32
                       )

    for layer_idx in range(n_layers):
        for step_idx in range(T):
            c = captures[layer_idx][step_idx]
            if not c:
                continue

            npz_name = f"{prompt_id}_l{layer_idx:02d}_s{step_idx:03d}.npz"
            npz_path = out_dir / npz_name

            np.savez_compressed(str(npz_path), **c)

            tensor_shapes = {k: list(v.shape) for k, v in c.items()}

            manifest.append({
                "prompt_id": prompt_id,
                "prompt_text": prompt,
                "layer": layer_idx,
                "step": step_idx,
                "token": tokens[step_idx],
                "file": npz_name,
                "shapes": tensor_shapes,
            })

    # Outputs section: display strings + rank of the actually-following token
    outputs_section[prompt_id] = {
        "output_file": f"{prompt_id}_output.npz",
        "steps": [
            {
                "step": t,
                "token": tokens[t],
                "top_tokens": outputs["top_tokens"][t],
                "actual_next": outputs["actual_next"][t],
                "actual_rank": int(outputs["actual_rank"][t]),
            }
            for t in range(T)
        ],
    }

    return T, n_layers


def export_model_meta(model, tokenizer, out_dir: Path):
    """Write model config metadata."""
    cfg = model.config
    meta = {
        "model_name": cfg._name_or_path if hasattr(cfg, "_name_or_path") else "mamba2",
        "n_layers": cfg.num_hidden_layers,
        "d_model": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
    }

    # Try to get Mamba2-specific dims
    for attr in ["state_size", "intermediate_size", "num_heads", "n_groups",
                 "conv_kernel_size", "expand", "head_dim"]:
        v = getattr(cfg, attr, None)
        if v is not None:
            meta[attr] = v

    # d_conv lives under a different config key; d_inner is not on the config
    # at all -- take both from the first mixer so the frontend can split proj_out
    if "conv_kernel_size" not in meta:
        meta["conv_kernel_size"] = getattr(cfg, "conv_kernel", 4)
    mixer0 = model.backbone.layers[0].mixer
    meta.setdefault("intermediate_size", mixer0.intermediate_size)

    # Real decay per layer/head for the SSMStatePanel decay bars: A = -exp(A_log)
    # A_log is a {n_heads} parameter on every layer's mixer.
    meta["A"] = [
        (-layer.mixer.A_log.detach().float().exp()).tolist()
        for layer in model.backbone.layers
    ]

    # Document the s_l decimation so panels can label the subsampled view
    meta["ssm_state_full"] = [meta.get("num_heads", 0), meta.get("head_dim", 0), meta.get("state_size", 0)]
    meta["ssm_state_sub"] = [meta.get("num_heads", 0), SSM_SUB_HEAD_DIM, SSM_SUB_D_STATE]

    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Model meta: {meta['model_name']}, A per layer/head: "
          f"{len(meta['A'])}x{len(meta['A'][0]) if meta['A'] else 0}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Export Mamba2 tensors for visualizer")
    parser.add_argument(
        "--model",
        default="AntonV/mamba2-130m-hf",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--out",
        default="tools/mamba2-explainer/static/data",
        help="Output directory for npz files and manifest",
    )
    parser.add_argument(
        "--prompts",
        type=int,
        default=5,
        help="Number of prompts to export (1-5)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=-1,
        help="Max layers to export (-1 = all)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer, model = load_model(args.model)

    export_model_meta(model, tokenizer, out_dir)

    manifest = []
    outputs_section = {}
    n_prompts = min(args.prompts, len(PROMPTS))

    for i, prompt in enumerate(PROMPTS[:n_prompts]):
        prompt_id = f"p{i:02d}"
        T, n_layers = export_prompt(model, tokenizer, prompt, prompt_id, out_dir, manifest, outputs_section)
        print(f"    -> {T} tokens, {n_layers} layers, "
              f"{len([m for m in manifest if m['prompt_id'] == prompt_id])} npz files")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"prompts": PROMPTS[:n_prompts], "entries": manifest, "outputs": outputs_section}, f, indent=2)

    print(f"\nDone. {len(manifest)} tensors exported to {out_dir}/")

    # Report total size
    total_bytes = sum(
        (out_dir / m["file"]).stat().st_size
        for m in manifest
        if (out_dir / m["file"]).exists()
    )
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
