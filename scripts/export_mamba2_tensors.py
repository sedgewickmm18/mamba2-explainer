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
    static/data/manifest.json          -- list of all exports
    static/data/<prompt_id>_l<layer>_s<step>.npz  -- per-step tensors

Per (layer, step) npz tensors:
    hidden_in   {d_model}             input to the layer (embeddings for layer 0)
    hidden_out  {d_model}             layer output after residual add
    proj_out    {d_in_proj}           W_in projection output, split z | xBC | dt
    conv_state  {conv_dim, d_conv}    r_l conv window after this token,
                                      columns [x_{t-d_conv+1} .. x_t], zero-padded at start
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
        if tokenizer is not None:
            break

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

    Returns: (tokens, captures, n_layers)
        tokens: list of token strings
        captures: [layer][step] -> dict of np arrays
    """
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids  # {1, T}
    T = input_ids.shape[1]

    layers = model.backbone.layers
    n_layers = len(layers)

    captures: list[list[dict]] = [
        [{} for _ in range(T)] for _ in range(n_layers)
    ]

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

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    return tokens, captures, n_layers


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
):
    print(f"  Prompt {prompt_id}: {prompt[:50]!r}")
    tokens, captures, n_layers = run_forward_token_by_token(model, tokenizer, prompt)
    T = len(tokens)

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

    with open(out_dir / "model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Model meta: {meta}")


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
    n_prompts = min(args.prompts, len(PROMPTS))

    for i, prompt in enumerate(PROMPTS[:n_prompts]):
        prompt_id = f"p{i:02d}"
        T, n_layers = export_prompt(model, tokenizer, prompt, prompt_id, out_dir, manifest)
        print(f"    -> {T} tokens, {n_layers} layers, "
              f"{len([m for m in manifest if m['prompt_id'] == prompt_id])} npz files")

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({"prompts": PROMPTS[:n_prompts], "entries": manifest}, f, indent=2)

    print(f"\nDone. {len(manifest)} tensors exported to {out_dir}/")
    print(f"Manifest: {manifest_path}")

    # Report total size
    total_bytes = sum(
        (out_dir / m["file"]).stat().st_size
        for m in manifest
        if (out_dir / m["file"]).exists()
    )
    print(f"Total size: {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
