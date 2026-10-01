<script lang="ts">
/**
 * SSMStatePanel.svelte - the centrepiece visualization.
 *
 * Shows the SSM state matrix S as d_state x head_dim heatmap per head.
 * Three sub-panels: S_before, update (dt*(B outer x)), S_after.
 * Context-length counter vs transformer KV cache size comparison.
 * Decay visualization: exp(A*dt) per head.
 *
 * Equation visualized:
 *   S_t = exp(A * dt) * S_{t-1}  +  dt * outer(B, x)
 *   y_t = C @ S_t
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';
import { appState } from '$lib/data/tensorLoader';

let svgMain: SVGSVGElement;
let svgDecay: SVGSVGElement;

const W_MAIN = 480;
const H_MAIN = 240;
const W_DECAY = 480;
const H_DECAY = 60;
const PAD = { top: 28, right: 10, bottom: 24, left: 10 };

// Synthetic model params when not available from tensors
const D_STATE = 64;
const N_HEADS = 24;
const HEAD_DIM = 64;

let selectedHead = 0;

$: stepIdx  = $appState.stepIndex;
$: meta     = $appState.meta;
$: nHeads   = meta?.num_heads ?? N_HEADS;
$: dState   = meta?.state_size ?? D_STATE;
$: headDim  = meta?.head_dim ?? HEAD_DIM;
$: nLayers  = meta?.n_layers ?? 24;
$: dModel   = meta?.d_model ?? 768;

// KV cache size a transformer would need for the same context length
$: transformerKVBytes = stepIdx * dModel * 2 * 2 * nLayers;  // float16, key+value, all layers
$: ssmStateBytes = dState * headDim * nHeads * 4;  // float32 per layer
$: ratio = transformerKVBytes > 0 ? (transformerKVBytes / ssmStateBytes).toFixed(1) : '--';

// Synthetic state matrices (until real data is loaded)
$: { if ($appState.tensors.size > 0) renderAll(); }

function syntheticState(step: number, head: number, ds: number, hd: number): Float32Array {
    const arr = new Float32Array(ds * hd);
    for (let i = 0; i < ds; i++) {
        for (let j = 0; j < hd; j++) {
            // Deterministic synthetic values that look plausible
            arr[i * hd + j] = Math.sin(step * 0.3 + head * 0.7 + i * 0.13 + j * 0.09) * 0.5;
        }
    }
    return arr;
}

function syntheticDecay(head: number, nH: number): Float32Array {
    // Simulated exp(A*dt) values: one scalar per head in [0.1, 0.99]
    const arr = new Float32Array(nH);
    for (let h = 0; h < nH; h++) {
        arr[h] = 0.1 + 0.88 * ((Math.sin(h * 0.41 + 0.2) + 1) / 2);
    }
    return arr;
}

function renderHeatmap(
    g: d3.Selection<SVGGElement, unknown, null, undefined>,
    data: Float32Array,
    rows: number,
    cols: number,
    x0: number,
    y0: number,
    w: number,
    h: number,
    absMax: number
) {
    const cScale = d3.scaleSequential(d3.interpolateRdBu).domain([absMax, -absMax]);
    const cellW = w / cols;
    const cellH = h / rows;
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = data[r * cols + c] ?? 0;
            g.append('rect')
                .attr('x', x0 + c * cellW)
                .attr('y', y0 + r * cellH)
                .attr('width', cellW - 0.3)
                .attr('height', cellH - 0.3)
                .attr('fill', cScale(val));
        }
    }
}

function renderAll() {
    renderMain();
    renderDecay();
}

function renderMain() {
    if (!svgMain) return;
    const svg = d3.select(svgMain);
    svg.selectAll('*').remove();

    const ds = dState;
    const hd = headDim;
    const plotW = W_MAIN - PAD.left - PAD.right;
    const plotH = H_MAIN - PAD.top - PAD.bottom;

    const panelW = Math.floor((plotW - 16) / 3);
    const panelH = plotH;

    const sBefore = syntheticState(Math.max(0, stepIdx - 1), selectedHead, ds, hd);
    const sAfter  = syntheticState(stepIdx, selectedHead, ds, hd);
    const update  = new Float32Array(ds * hd);
    for (let i = 0; i < ds * hd; i++) update[i] = sAfter[i] - sBefore[i];

    const allVals = [...sBefore, ...sAfter, ...update];
    const absMax = Math.max(0.01, Math.max(...allVals.map(Math.abs)));

    const g = svg.append('g');

    const panels = [
        { label: 'S_{t-1}', data: sBefore, x: PAD.left, color: '#60a5fa' },
        { label: 'update (dt*Bx)', data: update, x: PAD.left + panelW + 8, color: '#fb923c' },
        { label: 'S_t (new)', data: sAfter, x: PAD.left + 2 * (panelW + 8), color: '#4ade80' },
    ];

    for (const p of panels) {
        // Label
        svg.append('text')
            .attr('x', p.x + panelW / 2)
            .attr('y', PAD.top - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', p.color)
            .attr('font-size', 10)
            .text(p.label);

        renderHeatmap(g, p.data, ds, hd, p.x, PAD.top, panelW, panelH, absMax);
    }

    // Arrows between panels
    for (let i = 0; i < panels.length - 1; i++) {
        const x = panels[i].x + panelW + 2;
        const y = PAD.top + panelH / 2;
        svg.append('text')
            .attr('x', x + 3).attr('y', y + 4)
            .attr('fill', '#8b90a8').attr('font-size', 12)
            .text('+');
    }

    // Bottom: dimension annotation
    svg.append('text')
        .attr('x', PAD.left + panelW / 2).attr('y', H_MAIN - 6)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 8)
        .text(`${ds} x ${hd}`);
}

function renderDecay() {
    if (!svgDecay) return;
    const svg = d3.select(svgDecay);
    svg.selectAll('*').remove();

    const nH = nHeads;
    const decays = syntheticDecay(selectedHead, nH);
    const barW = (W_DECAY - 20) / nH;

    svg.append('text')
        .attr('x', 10).attr('y', 12)
        .attr('fill', '#8b90a8').attr('font-size', 9)
        .text('exp(A*dt) per head -- decay factor (darker = more forgetting)');

    const g = svg.append('g').attr('transform', 'translate(10,18)');
    const cScale = d3.scaleSequential((t) => d3.interpolateBlues(1 - t)).domain([0, 1]);

    for (let h = 0; h < nH; h++) {
        const v = decays[h];
        g.append('rect')
            .attr('x', h * barW).attr('y', 0)
            .attr('width', barW - 0.5).attr('height', 28)
            .attr('fill', cScale(v))
            .attr('stroke', h === selectedHead ? '#fb923c' : 'none')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('click', () => { selectedHead = h; renderAll(); })
            .append('title').text(`head ${h}: exp(A*dt) = ${v.toFixed(3)}`);
    }
    g.append('text')
        .attr('x', selectedHead * barW + barW / 2).attr('y', 40)
        .attr('text-anchor', 'middle').attr('fill', '#fb923c').attr('font-size', 8)
        .text(`head ${selectedHead}`);
}

onMount(renderAll);
afterUpdate(renderAll);
</script>

<div class="ssm-wrap">
    <!-- Context length counter -->
    <div class="counters">
        <div class="counter">
            <span class="counter-label">Context length</span>
            <span class="counter-val">{stepIdx + 1} tokens</span>
        </div>
        <div class="counter highlight">
            <span class="counter-label">Transformer KV cache</span>
            <span class="counter-val">{(transformerKVBytes / 1024).toFixed(0)} KB</span>
        </div>
        <div class="counter">
            <span class="counter-label">SSM state (this layer)</span>
            <span class="counter-val">{(ssmStateBytes / 1024).toFixed(1)} KB</span>
        </div>
        <div class="counter ratio">
            <span class="counter-label">Ratio KV/SSM</span>
            <span class="counter-val">{ratio}x</span>
        </div>
    </div>

    <!-- Head selector -->
    <div class="head-sel">
        <span class="head-label">Head</span>
        {#each Array.from({ length: nHeads }) as _, h}
            <button
                class="head-btn"
                class:active={h === selectedHead}
                on:click={() => { selectedHead = h; renderAll(); }}
            >{h}</button>
        {/each}
    </div>

    <!-- Main heatmap: S_before / update / S_after -->
    <svg bind:this={svgMain} width={W_MAIN} height={H_MAIN} />

    <!-- Decay bars per head -->
    <svg bind:this={svgDecay} width={W_DECAY} height={H_DECAY} />

    <div class="ssm-note">
        S has {dState * headDim * nHeads * 4} bytes. Transformer KV grows +{dModel * 2 * 2} bytes/token.
    </div>
</div>

<style>
.ssm-wrap { display: inline-block; }

.counters {
    display: flex;
    gap: 1rem;
    margin-bottom: 0.6rem;
    flex-wrap: wrap;
}
.counter {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
    font-size: 0.78rem;
}
.counter.highlight { border-color: #60a5fa; }
.counter.ratio { border-color: #4ade80; }
.counter-label { color: var(--text-dim); display: block; font-size: 0.7rem; }
.counter-val { color: var(--text); font-family: var(--font-mono); font-size: 0.9rem; }

.head-sel {
    display: flex;
    align-items: center;
    gap: 3px;
    margin-bottom: 6px;
    flex-wrap: wrap;
}
.head-label {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-right: 4px;
}
.head-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-dim);
    border-radius: 3px;
    width: 22px;
    height: 22px;
    font-size: 0.65rem;
    line-height: 1;
    transition: all var(--transition);
}
.head-btn.active {
    background: #fb923c;
    color: #000;
    border-color: #fb923c;
    font-weight: 700;
}

.ssm-note {
    font-size: 0.68rem;
    color: var(--text-dim);
    margin-top: 4px;
}
</style>
