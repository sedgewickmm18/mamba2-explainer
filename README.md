# Mamba2 Explainer

Interactive browser visualization of the Mamba2 SSM architecture, inspired by
[Transformer Explainer](https://poloclub.github.io/transformer-explainer/).

Shows the full Mamba2 data flow token by token using pre-computed tensors from
a real mamba2-130m model. No live inference required - fully static, deployable
to GitHub Pages.

---

## What it shows

Seven panels covering the complete Mamba2 forward pass — from the input
embedding to the next-token distribution:

| Panel | What you see |
|---|---|
| Input Embedding | Token sequence, real \|\|embedding\|\| per token |
| W_in Projection | zxBCdt output, color-coded split: z (purple) / xBC (teal) / dt (orange) |
| Conv History r_l | Sliding window buffer (d_conv-1=3 slots), constant size |
| SSM State s_l | Real S_before / update / S_after heatmaps (subsampled s_l), real exp(A·dt) decay per head, KV cache comparison |
| Scan / SSD | Decode recurrence with real dt slice vs prefill SSD path, causal decay mask L |
| SwiGLU + W_out | Gate, skip connection D*x, grouped norm, residual add |
| Output Distribution | Top-20 of the exported top-50 next-token candidates, actual next token ringed, entropy readout |

The data-flow tail of the model is explicit in the export and the UI:
`hidden_out (layer 23) → final RMSNorm → lm_head → logits → softmax →
next-token distribution` — panel 7 is always visible and dims with a note
while a non-final layer is selected, since the distribution is a property of
the whole stack, not of one layer.

The topbar has a free layer-navigation slider (0..23) plus prev/next arrows,
and a "Model diagram" link opening a mermaid-rendered architecture overlay
(authored from the `arch:llama.cpp:mamba2-ssm` FalkorDB graph, statically
embedded — no runtime graph access).

Includes a 16-step guided tour (auto-opens on first visit).

---

## Quick start (development)

```sh
# Install deps
cd tools/mamba2-explainer
npm install

# Generate tensor data (requires Python + torch + transformers)
cd ../..
pip install "transformers>=4.39" torch numpy
python scripts/export_mamba2_tensors.py \
    --model AntonV/mamba2-130m-hf \
    --out tools/mamba2-explainer/static/data

# Run dev server
cd tools/mamba2-explainer
npm run dev
```

Open http://localhost:5173

---

## Building for production

```sh
cd tools/mamba2-explainer
BASE_PATH=/your/subpath npm run build
# Output in dist/
```

For GitHub Pages deployment, `BASE_PATH` should match the repo path
(e.g. `/llama.cpp/mamba2-explainer`). The CI workflow sets this automatically.

---

## Re-generating tensors

To update the tensor data with a different model or additional prompts:

```sh
python scripts/export_mamba2_tensors.py \
    --model AntonV/mamba2-130m-hf \
    --out tools/mamba2-explainer/static/data \
    --prompts 5
```

Note: `state-spaces/mamba2-130m` uses the original mamba-ssm checkpoint format
and is not loadable via `Mamba2ForCausalLM`. Use `AntonV/mamba2-130m-hf` or
another community HF-converted repo.

Edit `PROMPTS` in [`scripts/export_mamba2_tensors.py`](../../scripts/export_mamba2_tensors.py)
to use different example sentences.

---

## Architecture

```
tools/mamba2-explainer/
  src/
    routes/
      +page.svelte          -- top-level layout, topbar (layer slider), data-flow strip, panel stage
      +layout.ts            -- static prerender config
    lib/
      data/
        tensorLoader.ts     -- manifest fetch, npz parser, Svelte store, output-distribution loader
        modelDiagram.ts     -- MAMBA2_MERMAID flowchart + modal open store
      components/
        EmbeddingPanel.svelte
        ProjectionPanel.svelte
        ConvPanel.svelte
        SSMStatePanel.svelte
        ScanPanel.svelte
        GateOutputPanel.svelte
        OutputDistributionPanel.svelte
        ModelDiagramModal.svelte
        GuidedTour.svelte
        Heatmap.svelte      -- shared reusable heatmap
    app.html                -- HTML shell
    app.css                 -- global design tokens (CSS vars)
  static/
    data/                   -- manifest.json, model_meta.json + *.npz tensor files
  svelte.config.js          -- adapter-static, base path
  vite.config.ts
  package.json
```

Reference architecture document: [`docs/architecture/mamba2-ssm-dataflow.md`](../../docs/architecture/mamba2-ssm-dataflow.md)
