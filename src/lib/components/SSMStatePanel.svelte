<script lang="ts">
/**
 * SSMStatePanel.svelte - the centrepiece visualization.
 *
 * Shows the recurrent SSM state s_l as a d_state x head_dim heatmap per head,
 * entirely from real exported tensors:
 *   - S_t     from ssm_state in the current step's npz ({n_head,16,16}
 *             strided subsample of the full {n_head,64,128} state)
 *   - S_{t-1} from the previous step's npz within the same layer
 *             (zeros at step 0: the state starts empty)
 *   - update  = S_t - S_{t-1}, computed client-side
 *
 * Decay visualization: exp(A * dt) per head with the real per-head A from
 * model_meta.json (A = -exp(A_log)) and dt taken from the dt slice of the
 * current step's proj_out.
 *
 * Equation visualized:
 *   S_t = exp(A * dt) * S_{t-1}  +  dt * outer(B, x)
 *   y_t = C @ S_t
 */
import { onMount } from 'svelte';
import * as d3 from 'd3';
import { appState, loadStepTensors, type ModelMeta } from '$lib/data/tensorLoader';

let svgMain: SVGSVGElement;
let svgDecay: SVGSVGElement;

const W_MAIN = 480;
const H_MAIN = 240;
const W_DECAY = 480;
const H_DECAY = 60;
const PAD = { top: 28, right: 10, bottom: 24, left: 10 };

let selectedHead = 0;

$: stepIdx  = $appState.stepIndex;
$: layerIdx = $appState.layerIndex;
$: meta     = $appState.meta;
$: tensors  = $appState.tensors;
$: nHeads   = meta?.num_heads ?? 24;
$: subD     = meta?.ssm_state_sub?.[1] ?? 16;   // subsampled head_dim
$: subW     = meta?.ssm_state_sub?.[2] ?? 16;   // subsampled d_state
$: fullH    = meta?.head_dim ?? 64;              // full head_dim
$: fullD    = meta?.state_size ?? 128;           // full d_state
$: dModel   = meta?.d_model ?? 768;
$: nLayers  = meta?.n_layers ?? 24;

// KV cache a transformer would need for the same context length
$: transformerKVBytes = (stepIdx + 1) * dModel * 2 * 2 * nLayers;  // f16, K+V, all layers
$: ssmStateBytes = fullD * fullH * nHeads * 4;                     // f32, this layer
$: ratio = transformerKVBytes > 0 ? (transformerKVBytes / ssmStateBytes).toFixed(1) : '--';

// Real decay per head: exp(A * dt) with A from model_meta and dt from proj_out
$: decayPerHead = computeDecay(tensors, meta, layerIdx);

// Previous-step s_l cache: S_before(t) = S_after(t-1) in the same layer
const prevStateCache = new Map<string, Float32Array>();
let renderToken = 0;

function cellCount(): number {
    return subD * subW;
}

async function loadPrevState(promptIdx: number, layerIdx: number, step: number): Promise<Float32Array | null> {
    if (step <= 0) return new Float32Array(cellCount()); // state starts at zero
    const key = `${promptIdx}:${layerIdx}:${step - 1}`;
    const hit = prevStateCache.get(key);
    if (hit) return hit;
    const prev = await loadStepTensors(promptIdx, layerIdx, step - 1);
    const st = prev?.get('ssm_state') ?? null;
    if (st) prevStateCache.set(key, st);
    return st;
}

function dtSlice(proj: Float32Array | null, m: ModelMeta | null): Float32Array | null {
    if (!proj || !m) return null;
    const dInner =
        m.intermediate_size ??
        (m.head_dim && m.num_heads ? m.head_dim * m.num_heads : null);
    const dState = m.state_size ?? null;
    const nGroups = m.n_groups ?? null;
    const nH = m.num_heads ?? null;
    if (!dInner || !dState || !nGroups || !nH) return null;
    const dtStart = dInner + (dInner + 2 * nGroups * dState);
    if (dtStart + nH > proj.length) return null;
    return proj.subarray(dtStart, dtStart + nH);
}

function computeDecay(tens: Map<string, Float32Array>, m: ModelMeta | null, layer: number): Float32Array | null {
    const A = m?.A?.[layer];
    const dt = dtSlice(tens.get('proj_out') ?? null, m);
    if (!A || !dt || dt.length !== A.length) return null;
    const arr = new Float32Array(A.length);
    for (let h = 0; h < A.length; h++) {
        arr[h] = Math.exp(A[h] * dt[h]);
    }
    return arr;
}

async function renderPipeline(
    step: number,
    layer: number,
    state: Float32Array | null,
    head: number
) {
    const token = ++renderToken;
    if (!state) {
        if (svgMain) {
            const svg = d3.select(svgMain);
            svg.selectAll('*').remove();
            svg.append('text')
                .attr('x', W_MAIN / 2).attr('y', H_MAIN / 2)
                .attr('text-anchor', 'middle')
                .attr('fill', '#8b90a8').attr('font-size', 11)
                .text('awaiting ssm_state data ...');
        }
        return;
    }
    const perHead = cellCount();
    const after = state.subarray(head * perHead, (head + 1) * perHead);
    const before = await loadPrevState($appState.promptIndex, layer, step);
    if (token !== renderToken) return; // a newer step/layer won the race
    renderMain(before, after, head);
}

function renderMain(before: Float32Array | null, after: Float32Array, head: number) {
    if (!svgMain || !after) return;
    const svg = d3.select(svgMain);
    svg.selectAll('*').remove();

    const ds = subW; // subsampled d_state (rows)
    const hd = subD; // subsampled head_dim (cols)
    const plotW = W_MAIN - PAD.left - PAD.right;
    const plotH = H_MAIN - PAD.top - PAD.bottom;
    const panelW = Math.floor((plotW - 16) / 3);

    const sBefore = before ?? new Float32Array(ds * hd);
    const sAfter = after;
    const update = new Float32Array(ds * hd);
    for (let i = 0; i < ds * hd; i++) update[i] = sAfter[i] - sBefore[i];

    const absMax = Math.max(
        0.01,
        Math.abs(d3.min([...sBefore, ...sAfter, ...update]) ?? 0.01),
        Math.abs(d3.max([...sBefore, ...sAfter, ...update]) ?? 0.01)
    );

    const g = svg.append('g');

    const panels = [
        { label: 'S_{t-1}', data: sBefore, x: PAD.left, color: '#60a5fa' },
        { label: 'update (dt*Bx)', data: update, x: PAD.left + panelW + 8, color: '#fb923c' },
        { label: 'S_t (new)', data: sAfter, x: PAD.left + 2 * (panelW + 8), color: '#4ade80' },
    ];

    for (const p of panels) {
        svg.append('text')
            .attr('x', p.x + panelW / 2)
            .attr('y', PAD.top - 8)
            .attr('text-anchor', 'middle')
            .attr('fill', p.color)
            .attr('font-size', 10)
            .text(p.label);

        renderHeatmap(g, p.data, ds, hd, p.x, PAD.top, panelW, plotH, absMax);
    }

    // Arrows between panels
    for (let i = 0; i < panels.length - 1; i++) {
        const x = panels[i].x + panelW + 2;
        const y = PAD.top + plotH / 2;
        svg.append('text')
            .attr('x', x + 3).attr('y', y + 4)
            .attr('fill', '#8b90a8').attr('font-size', 12)
            .text('+');
    }

    // Bottom: honest dimension annotation for the decimated view
    svg.append('text')
        .attr('x', PAD.left + plotW / 2).attr('y', H_MAIN - 6)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 8)
        .text(`${ds} x ${hd} subsample of the full ${fullD} x ${fullH} s_l`);
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
                .attr('fill', cScale(val))
                .append('title')
                .text(`s_l[${r},${c}] = ${val.toFixed(4)}`);
        }
    }
}

function renderDecay() {
    if (!svgDecay) return;
    const svg = d3.select(svgDecay);
    svg.selectAll('*').remove();

    const decays = decayPerHead;
    const barW = (W_DECAY - 20) / nHeads;

    svg.append('text')
        .attr('x', 10).attr('y', 12)
        .attr('fill', '#8b90a8').attr('font-size', 9)
        .text(decays
            ? 'exp(A*dt) per head -- real A and dt (darker = more forgetting)'
            : 'exp(A*dt) per head -- awaiting A (model_meta) and dt (proj_out) data');

    const g = svg.append('g').attr('transform', 'translate(10,18)');
    const cScale = d3.scaleSequential((t: number) => d3.interpolateBlues(1 - t)).domain([0, 1]);

    for (let h = 0; h < nHeads; h++) {
        const v = decays ? Math.min(1, Math.max(0, decays[h])) : 0.5;
        g.append('rect')
            .attr('x', h * barW).attr('y', 0)
            .attr('width', barW - 0.5).attr('height', 28)
            .attr('fill', decays ? cScale(v) : '#2e3347')
            .attr('stroke', h === selectedHead ? '#fb923c' : 'none')
            .attr('stroke-width', 2)
            .style('cursor', 'pointer')
            .on('click', () => { selectedHead = h; })
            .append('title')
            .text(decays
                ? `head ${h}: exp(A*dt) = ${decays[h].toFixed(4)}`
                : `head ${h}`);
    }
    g.append('text')
        .attr('x', selectedHead * barW + barW / 2)
        .attr('y', 40)
        .attr('text-anchor', 'middle').attr('fill', '#fb923c').attr('font-size', 8)
        .text(`head ${selectedHead}`);
}

onMount(renderDecay);
// Re-render everything whenever the underlying data or selection moves
$: decayPerHead, renderDecay();
$: void renderPipeline(stepIdx, layerIdx, tensors.get('ssm_state') ?? null, selectedHead);
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
                on:click={() => { selectedHead = h; }}
            >{h}</button>
        {/each}
    </div>

    <!-- Main heatmap: S_before / update / S_after -->
    <svg bind:this={svgMain} width={W_MAIN} height={H_MAIN} />

    <!-- Decay bars per head -->
    <svg bind:this={svgDecay} width={W_DECAY} height={H_DECAY} />

    <div class="ssm-note">
        s_l is {fullD * fullH * nHeads * 4} bytes (f32) and constant. Transformer KV grows +{dModel * 2 * 2} bytes/token/layer.
        Heatmaps show a {subW} x {subD} strided subsample; update = S_t - S_{'{t-1}'}.
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
