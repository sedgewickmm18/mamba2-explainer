<script lang="ts">
/**
 * +page.svelte - Top-level page layout.
 * Top bar: prompt selector, layer selector, token step prev/next.
 * Main canvas: horizontally scrollable SVG stage with all panels.
 */
import '../app.css';
import { onMount } from 'svelte';
import {
    appState,
    maxStep,
    maxLayer,
    initLoader,
    selectPrompt,
    selectLayer,
    stepForward,
    stepBack,
    goToStep,
} from '$lib/data/tensorLoader';

import EmbeddingPanel from '$lib/components/EmbeddingPanel.svelte';
import ProjectionPanel from '$lib/components/ProjectionPanel.svelte';
import ConvPanel from '$lib/components/ConvPanel.svelte';
import SSMStatePanel from '$lib/components/SSMStatePanel.svelte';
import ScanPanel from '$lib/components/ScanPanel.svelte';
import GateOutputPanel from '$lib/components/GateOutputPanel.svelte';
import OutputDistributionPanel from '$lib/components/OutputDistributionPanel.svelte';
import GuidedTour from '$lib/components/GuidedTour.svelte';
import ModelDiagramModal from '$lib/components/ModelDiagramModal.svelte';
import { openModelDiagram } from '$lib/data/modelDiagram';

onMount(() => {
    initLoader();
});

$: manifest  = $appState.manifest;
$: meta      = $appState.meta;
$: prompts   = manifest?.prompts ?? [];
$: promptIdx = $appState.promptIndex;
$: layerIdx  = $appState.layerIndex;
$: stepIdx   = $appState.stepIndex;
$: tokens    = $appState.tokens;
$: loading   = $appState.loading;
$: error     = $appState.error;
$: nLayers   = meta?.n_layers ?? 24;

// Slider label follows the store; dragging updates the label only, and the
// per-layer npz fetch happens on release (change) to avoid fetch storms.
let sliderLayer = 0;
$: if (layerIdx !== sliderLayer) sliderLayer = layerIdx;

function onPromptChange(e: Event) {
    const idx = parseInt((e.target as HTMLSelectElement).value, 10);
    selectPrompt(idx);
}

function onLayerLeft() {
    if (layerIdx > 0) selectLayer(layerIdx - 1);
}
function onLayerRight() {
    if (layerIdx < $maxLayer) selectLayer(layerIdx + 1);
}
function onLayerSlide(e: Event) {
    sliderLayer = parseInt((e.target as HTMLInputElement).value, 10);
}
function onLayerCommit() {
    if (sliderLayer !== layerIdx) selectLayer(sliderLayer);
}

function scrollToDistribution() {
    const el = document.getElementById('panel-distribution');
    if (el) el.scrollIntoView({ behavior: 'smooth', inline: 'center', block: 'nearest' });
}
</script>

<div class="app-root">
    <!-- Top bar -->
    <header class="topbar">
        <span class="topbar-title">Mamba2 Explainer</span>

        <!-- Model diagram link -->
        <div class="topbar-group">
            <button class="topbar-select" on:click={openModelDiagram} aria-label="Open model diagram">
                📊 Model diagram
            </button>
        </div>

        <!-- Prompt selector -->
        <label class="topbar-group" for="prompt-select">
            <span class="topbar-label">Prompt</span>
            <select
                id="prompt-select"
                class="topbar-select"
                value={promptIdx}
                on:change={onPromptChange}
            >
                {#each prompts as p, i}
                    <option value={i}>{p.length > 40 ? p.slice(0, 40) + '...' : p}</option>
                {/each}
                {#if prompts.length === 0}
                    <option value={0}>-- no data loaded --</option>
                {/if}
            </select>
        </label>

        <!-- Layer selector -->
        <div class="topbar-group">
            <span class="topbar-label">Layer</span>
            <button class="nav-btn" on:click={onLayerLeft} disabled={layerIdx <= 0} aria-label="Previous layer">&#8249;</button>
            <div class="layer-slider-wrapper">
                <input
                    type="range"
                    min={0}
                    max={$maxLayer}
                    value={sliderLayer}
                    on:input={onLayerSlide}
                    on:change={onLayerCommit}
                    class="layer-slider"
                    aria-label="Layer selector"
                />
                <span class="layer-value">{sliderLayer} / {nLayers - 1}</span>
            </div>
            <button class="nav-btn" on:click={onLayerRight} disabled={layerIdx >= $maxLayer} aria-label="Next layer">&#8250;</button>
        </div>

        <!-- Token step -->
        <div class="topbar-group">
            <span class="topbar-label">Token step</span>
            <button class="nav-btn" on:click={stepBack} disabled={stepIdx <= 0} aria-label="Previous token">&#8249;</button>
            <span class="nav-val">{stepIdx} / {$maxStep}</span>
            <button class="nav-btn" on:click={stepForward} disabled={stepIdx >= $maxStep} aria-label="Next token">&#8250;</button>
        </div>

        <!-- Token chips -->
        <div class="token-chips" aria-label="Token sequence">
            {#each tokens as tok, i}
                <button
                    class="token-chip"
                    class:active={i === stepIdx}
                    on:click={() => goToStep(i)}
                    title="Step {i}: {tok}"
                >{tok}</button>
            {/each}
            {#if tokens.length === 0}
                <span class="topbar-label">No tokens loaded</span>
            {/if}
        </div>

        <!-- Loading indicator -->
        {#if loading}
            <span class="loading-dot" aria-live="polite">...</span>
        {/if}
    </header>

    <!-- Error banner -->
    {#if error}
        <div class="error-banner" role="alert">
            <strong>Error:</strong> {error}
            <br/>
            <small>Run <code>python scripts/export_mamba2_tensors.py</code> to generate tensor data, then refresh.</small>
        </div>
    {/if}

    <!-- No data notice -->
    {#if !loading && !error && (!manifest || manifest.entries.length === 0)}
        <div class="notice-banner">
            <strong>No tensor data found.</strong>
            Generate data with:
            <code>python scripts/export_mamba2_tensors.py --out tools/mamba2-explainer/static/data</code>
            then rebuild.
        </div>
    {/if}

    <!-- Data-flow strip above stage: Embedding -> Layer l-1 -> Layer l -> Layer l+1 -> Final Norm -> lm_head -> Distribution -->
    <div class="data-flow-strip">
        <div class="flow-breadcrumb">
            <button type="button"
                class="flow-step"
                on:click={() => selectLayer(0)}
                title="Jump to the embedding (input of layer 0)">Embedding</button>
            <svg class="flow-arrow" viewBox="0 0 8 8"><path d="M2 1 L6 4 L2 7"/></svg>
            {#if layerIdx > 0}
                <button type="button"
                    class="flow-step"
                    on:click={() => selectLayer(layerIdx - 1)}
                    title="hidden_out of this layer feeds hidden_in of layer {layerIdx}">Layer {layerIdx - 1}</button>
                <svg class="flow-arrow" viewBox="0 0 8 8"><path d="M2 1 L6 4 L2 7"/></svg>
            {/if}
            <span class="flow-step active" role="note" title="Currently inspected layer">Layer {layerIdx}</span>
            <svg class="flow-arrow" viewBox="0 0 8 8"><path d="M2 1 L6 4 L2 7"/></svg>
            {#if layerIdx < $maxLayer}
                <button type="button"
                    class="flow-step"
                    on:click={() => selectLayer(layerIdx + 1)}
                    title="hidden_out of layer {layerIdx} feeds hidden_in of this layer">Layer {layerIdx + 1}</button>
                <svg class="flow-arrow" viewBox="0 0 8 8"><path d="M2 1 L6 4 L2 7"/></svg>
            {/if}
            <button type="button"
                class="flow-step"
                on:click={() => selectLayer($maxLayer)}
                title="Final RMSNorm consumes hidden_out of layer {$maxLayer}">Final Norm</button>
            <svg class="flow-arrow" viewBox="0 0 8 8"><path d="M2 1 L6 4 L2 7"/></svg>
            <button type="button"
                class="flow-step"
                on:click={scrollToDistribution}
                title="lm_head + softmax produce panel 7's next-token distribution">lm_head → Distribution</button>
        </div>
    </div>

    <!-- Main stage: horizontally scrollable panels -->
    <main class="stage-scroll" id="main-stage">
        <div class="stage-inner">
            <!-- Panel 1: Input embedding -->
            <section class="stage-panel" id="panel-embedding" data-panel="embedding">
                <div class="panel-title">1. Input Embedding</div>
                <EmbeddingPanel />
            </section>

            <div class="stage-arrow">&#8594;</div>

            <!-- Panel 2: W_in projection -->
            <section class="stage-panel" id="panel-projection" data-panel="projection">
                <div class="panel-title">2. W<sub>in</sub> Projection</div>
                <ProjectionPanel />
            </section>

            <div class="stage-arrow">&#8594;</div>

            <!-- Panel 3: Conv history -->
            <section class="stage-panel" id="panel-conv" data-panel="conv">
                <div class="panel-title">3. Conv History r<sub>l</sub></div>
                <ConvPanel />
            </section>

            <div class="stage-arrow">&#8594;</div>

            <!-- Panel 4: SSM state -->
            <section class="stage-panel panel-wide" id="panel-ssm" data-panel="ssm">
                <div class="panel-title">4. SSM State s<sub>l</sub></div>
                <SSMStatePanel />
            </section>

            <div class="stage-arrow">&#8594;</div>

            <!-- Panel 5: Scan / SSD -->
            <section class="stage-panel" id="panel-scan" data-panel="scan">
                <div class="panel-title">5. Scan / SSD</div>
                <ScanPanel />
            </section>

            <div class="stage-arrow">&#8594;</div>

            <!-- Panel 6: Gate and output -->
            <section class="stage-panel" id="panel-gate" data-panel="gate">
                <div class="panel-title">6. SwiGLU + W<sub>out</sub></div>
                <GateOutputPanel />
            </section>

            <div class="stage-arrow">&#8594;</div>

            <!-- Panel 7: Output Distribution (always visible) -->
            <section class="stage-panel" id="panel-distribution" data-panel="output-distribution">
                <div class="panel-title">7. Output Distribution</div>
                <OutputDistributionPanel />
            </section>
        </div>
    </main>

    <!-- Guided tour overlay -->
    <GuidedTour />

    <!-- Model diagram modal overlay -->
    <ModelDiagramModal />
</div>

<style>
.app-root {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
    background: var(--bg);
}

/* Top bar */
.topbar {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 0.6rem 1.5rem;
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    flex-wrap: wrap;
    position: sticky;
    top: 0;
    z-index: 100;
}

.topbar-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-teal);
    white-space: nowrap;
    letter-spacing: 0.03em;
}

.topbar-group {
    display: flex;
    align-items: center;
    gap: 0.4rem;
}

.topbar-label {
    font-size: 0.78rem;
    color: var(--text-dim);
    white-space: nowrap;
}

.topbar-select {
    background: var(--bg-card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 0.25rem 0.5rem;
    font-size: 0.82rem;
    max-width: 220px;
}

.nav-btn {
    background: var(--bg-card);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 4px;
    width: 26px;
    height: 26px;
    font-size: 1rem;
    line-height: 1;
    transition: background var(--transition);
}
.nav-btn:hover:not(:disabled) {
    background: var(--accent-blue);
    color: #fff;
}
.nav-btn:disabled {
    opacity: 0.35;
    cursor: not-allowed;
}

.nav-val {
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text);
    min-width: 52px;
    text-align: center;
}

/* Token chips */
.token-chips {
    display: flex;
    gap: 4px;
    flex-wrap: nowrap;
    overflow-x: auto;
    max-width: 500px;
    padding-bottom: 2px;
}
.token-chip {
    background: var(--bg-card);
    color: var(--text-dim);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 7px;
    font-family: var(--font-mono);
    font-size: 0.72rem;
    white-space: nowrap;
    transition: all var(--transition);
}
.token-chip.active {
    background: var(--accent-teal);
    color: #000;
    border-color: var(--accent-teal);
    font-weight: 700;
}
.token-chip:hover {
    border-color: var(--accent-blue);
}

/* Loading dot */
.loading-dot {
    font-family: var(--font-mono);
    color: var(--accent-orange);
    font-size: 1.2rem;
    animation: blink 1s step-start infinite;
}
@keyframes blink { 0%, 100% { opacity: 1 } 50% { opacity: 0.2 } }

/* Banners */
.error-banner {
    background: #3b1414;
    border: 1px solid #7f2020;
    color: #fca5a5;
    padding: 0.75rem 1.5rem;
    font-size: 0.875rem;
}
.notice-banner {
    background: #1e2d1e;
    border: 1px solid #3a5f3a;
    color: #86efac;
    padding: 0.75rem 1.5rem;
    font-size: 0.875rem;
}

/* Stage */
.stage-scroll {
    flex: 1;
    overflow-x: auto;
    padding: 1.5rem;
}
.stage-inner {
    display: flex;
    align-items: flex-start;
    gap: 0;
    min-width: max-content;
}

.stage-panel {
    background: var(--bg-panel);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: var(--panel-pad);
    min-width: 280px;
    max-width: 360px;
    flex-shrink: 0;
}
.stage-panel.panel-wide {
    min-width: 420px;
    max-width: 540px;
}

.panel-title {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 0.75rem;
}

.stage-arrow {
    align-self: center;
    padding: 0 0.75rem;
    color: var(--border);
    font-size: 1.4rem;
    flex-shrink: 0;
}

/* Layer slider */
.layer-slider-wrapper {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex: 1;
    max-width: 300px;
}

.layer-slider {
    flex: 1;
    cursor: pointer;
}

.layer-value {
    min-width: 52px;
    text-align: center;
    font-family: var(--font-mono);
    font-size: 0.8rem;
    color: var(--text);
}

/* Data-flow strip */
.data-flow-strip {
    background: var(--bg-panel);
    border-bottom: 1px solid var(--border);
    border-top: 1px solid var(--border);
    padding: 0.5rem 1rem;
    margin: 0.5rem 0;
}

.flow-breadcrumb {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    flex-wrap: wrap;
    font-size: 0.75rem;
    color: var(--text-dim);
}

.flow-step {
    font: inherit;
    cursor: pointer;
    display: inline-flex;
    align-items: center;
    gap: 2px;
    padding: 2px 6px;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 4px;
    cursor: pointer;
}

.flow-step.active {
    background: var(--accent-teal);
    color: #000;
    border-color: var(--accent-teal);
}

.flow-step:hover:not(.active) {
    background: var(--bg-card);
}

.flow-arrow {
    width: 10px;
    height: 10px;
    flex-shrink: 0;
}

.flow-arrow path {
    fill: none;
    stroke: var(--text-dim);
    stroke-width: 1.4;
}
</style>
