<script lang="ts">
/**
 * ScanPanel.svelte
 * Decode vs Prefill mode indicator.
 *
 * DECODE (n_tok=1, sequential scan):
 *   Animate the recurrence: S = exp(A*dt)*S + dt*(B outer x)
 *   The dt bar chart shows the REAL dt slice of the current step's proj_out
 *   (one value per head, before softplus + dt_bias) and the real decay
 *   exp(A*dt) per head using A from model_meta.json.
 *
 * PREFILL (n_tok>1, SSD path):
 *   Show causal decay mask L as lower-triangular heatmap.
 *   Chunk structure at chunk_size=256.
 *   Platform note: SSD only on NVIDIA Turing+; HIP/AMD uses scan.
 *
 * See mamba2-ssm-dataflow.md: SSD formula:
 *   Y = (L o (C @ B^T)) @ (X * dt)  +  decay * C @ S_init
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';
import { appState, type ModelMeta } from '$lib/data/tensorLoader';

const W = 300;
const H = 280;
const CHUNK = 16;  // display chunk size (actual = 256, but we show 16 for demo)

$: stepIdx = $appState.stepIndex;
$: tokens  = $appState.tokens;

let mode: 'decode' | 'prefill' = 'decode';
let svgEl: SVGSVGElement;

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

function buildDecayMask(n: number): Float32Array {
    // Lower-triangular decay mask L
    const arr = new Float32Array(n * n);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j <= i; j++) {
            // Simulated decay: L[i,j] = exp(-0.3*(i-j))
            arr[i * n + j] = Math.exp(-0.3 * (i - j));
        }
    }
    return arr;
}

function render() {
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    if (mode === 'decode') {
        renderDecode(svg);
    } else {
        renderPrefill(svg);
    }
}

function renderDecode(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const cx = W / 2;

    svg.append('text')
        .attr('x', cx).attr('y', 20)
        .attr('text-anchor', 'middle').attr('fill', '#16a34a').attr('font-size', 11).attr('font-weight', '700')
        .text('DECODE MODE  (n_tok = 1)');

    svg.append('text')
        .attr('x', cx).attr('y', 36)
        .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 9)
        .text('Sequential scan -- one thread per state element');

    // Formula box
    const eq = 'S_t = exp(A*dt) * S_{t-1}  +  dt * B @ x^T';
    svg.append('rect')
        .attr('x', 12).attr('y', 46).attr('width', W - 24).attr('height', 28)
        .attr('fill', '#eef1f5').attr('stroke', '#c9cfda').attr('rx', 4);
    svg.append('text')
        .attr('x', cx).attr('y', 65)
        .attr('text-anchor', 'middle').attr('fill', '#22272e').attr('font-family', 'monospace').attr('font-size', 9)
        .text(eq);
    // A: learned per-head decay, constant at inference (A_log is a {n_heads}
    // parameter of every layer; A = -exp(A_log) < 0, input-independent). With
    // dt it fixes the retention exp(A*dt) applied to S_{t-1}.
    svg.append('text')
        .attr('x', cx).attr('y', 84)
        .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 7)
        .text('A: learned per-head decay, constant at inference (A = -exp(A_log) < 0)');

    // Real dt values from the proj_out dt slice (one per head)
    const meta = $appState.meta;
    const dt = dtSlice($appState.tensors.get('proj_out') ?? null, meta);
    const nH = dt ? dt.length : 0;

    if (!dt || nH === 0) {
        svg.append('text')
            .attr('x', cx).attr('y', 140)
            .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 10)
            .text('awaiting proj_out data ...');
        return;
    }

    const barW = (W - 24) / nH;
    const barMaxH = 80;
    const dtMax = Math.max(...Array.from(dt), 1e-6);
    const cScale = d3.scaleSequential(d3.interpolateOranges).domain([0, dtMax]);

    svg.append('text')
        .attr('x', 12).attr('y', 100)
        .attr('fill', '#64748b').attr('font-size', 9)
        .text(`dt raw slice of proj_out, ${nH} heads (softplus + dt_bias applied in the scan):`);

    for (let h = 0; h < nH; h++) {
        const bh = (Math.max(dt[h], 0) / dtMax) * barMaxH;
        const x = 12 + h * barW;
        const y = 110 + barMaxH - bh;
        svg.append('rect')
            .attr('x', x + 1).attr('y', y).attr('width', barW - 2).attr('height', bh)
            .attr('fill', cScale(dt[h])).attr('rx', 2)
            .append('title')
            .text(`head ${h}: dt = ${dt[h].toFixed(4)}`);
        if (nH <= 12) {
            svg.append('text')
                .attr('x', x + barW / 2).attr('y', 110 + barMaxH + 11)
                .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 7)
                .text(`h${h}`);
        }
    }

    // Real decay exp(A*dt) using A from model_meta
    const A = meta?.A?.[$appState.layerIndex];
    svg.append('text')
        .attr('x', 12).attr('y', 215)
        .attr('fill', '#64748b').attr('font-size', 9)
        .text(A ? 'exp(A*dt) per head (real A from A_log):' : 'exp(A*dt) per head (awaiting A in model_meta):');

    for (let h = 0; h < nH; h++) {
        const decay = A ? Math.exp(A[h] * dt[h]) : 0.5;
        const x = 12 + h * barW;
        svg.append('rect')
            .attr('x', x + 1).attr('y', 220).attr('width', barW - 2).attr('height', 22)
            .attr('fill', d3.interpolateBlues(1 - Math.min(1, Math.max(0, decay)))).attr('rx', 2)
            .append('title')
            .text(`head ${h}: exp(A*dt) = ${decay.toFixed(4)}`);
    }

    svg.append('text')
        .attr('x', cx).attr('y', 258)
        .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 9)
        .text('O(T*N) serial per token -- O(1) cache');
}

function renderPrefill(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const cx = W / 2;

    svg.append('text')
        .attr('x', cx).attr('y', 20)
        .attr('text-anchor', 'middle').attr('fill', '#7c3aed').attr('font-size', 11).attr('font-weight', '700')
        .text('PREFILL MODE  (SSD, n_tok > 1)');

    svg.append('text')
        .attr('x', cx).attr('y', 36)
        .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 9)
        .text('Chunked into matmuls -- chunk_size = 256');

    // Causal decay mask L (lower-triangular heatmap)
    const n = CHUNK;
    const mask = buildDecayMask(n);
    const x0 = cx - 110;
    const y0 = 52;
    const size = 190;
    const cell = size / n;
    const cScale = d3.scaleSequential(d3.interpolateViridis).domain([0, 1]);

    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const v = mask[i * n + j];
            svg.append('rect')
                .attr('x', x0 + j * cell).attr('y', y0 + i * cell)
                .attr('width', cell).attr('height', cell)
                .attr('fill', v > 0 ? cScale(v) : '#eef0f4');
        }
    }

    svg.append('text')
        .attr('x', cx).attr('y', y0 + size + 16)
        .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 9)
        .text('causal decay mask L (16x16 cut of chunk_size 256)');
}

onMount(render);
afterUpdate(render);
</script>

<div class="scan-wrap">
    <div class="mode-toggle">
        <button class="mode-btn" class:active={mode === 'decode'} on:click={() => { mode = 'decode'; render(); }}>
            Decode (n_tok=1)
        </button>
        <button class="mode-btn" class:active={mode === 'prefill'} on:click={() => { mode = 'prefill'; render(); }}>
            Prefill (SSD)
        </button>
    </div>

    <svg bind:this={svgEl} width={W} height={H} />

    {#if mode === 'prefill'}
        <div class="platform-note">
            SSD (n_tok > 128): NVIDIA Turing+ only via cuBLAS.<br/>
            AMD HIP and MUSA always use the sequential scan path.
        </div>
    {/if}
</div>

<style>
.scan-wrap { display: inline-block; }

.mode-toggle {
    display: flex;
    gap: 6px;
    margin-bottom: 8px;
}
.mode-btn {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-dim);
    border-radius: 4px;
    padding: 4px 12px;
    font-size: 0.78rem;
    transition: all var(--transition);
}
.mode-btn.active {
    background: var(--accent-purple);
    color: #fff;
    border-color: var(--accent-purple);
    font-weight: 700;
}

.platform-note {
    font-size: 0.7rem;
    color: var(--text-dim);
    border-left: 2px solid var(--accent-orange);
    padding-left: 8px;
    margin-top: 4px;
    line-height: 1.5;
}
</style>
