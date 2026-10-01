<script lang="ts">
/**
 * ScanPanel.svelte
 * Decode vs Prefill mode indicator.
 *
 * DECODE (n_tok=1, sequential scan):
 *   Animate the recurrence: S = exp(A*dt)*S + dt*(B outer x)
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
import { appState } from '$lib/data/tensorLoader';

const W = 300;
const H = 280;
const CHUNK = 16;  // display chunk size (actual = 256, but we show 16 for demo)

$: stepIdx = $appState.stepIndex;
$: tokens  = $appState.tokens;
$: isMultiToken = false;  // set by toggle

let mode: 'decode' | 'prefill' = 'decode';
let svgEl: SVGSVGElement;

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
        .attr('text-anchor', 'middle').attr('fill', '#4ade80').attr('font-size', 11).attr('font-weight', '700')
        .text('DECODE MODE  (n_tok = 1)');

    svg.append('text')
        .attr('x', cx).attr('y', 36)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
        .text('Sequential scan -- one thread per state element');

    // Formula box
    const eq = 'S_t = exp(A*dt) * S_{t-1}  +  dt * B @ x^T';
    svg.append('rect')
        .attr('x', 12).attr('y', 46).attr('width', W - 24).attr('height', 28)
        .attr('fill', '#1a1d27').attr('stroke', '#2e3347').attr('rx', 4);
    svg.append('text')
        .attr('x', cx).attr('y', 65)
        .attr('text-anchor', 'middle').attr('fill', '#e8eaf0').attr('font-family', 'monospace').attr('font-size', 9)
        .text(eq);

    // Show a bar chart of synthetic dt values (one per head)
    const nH = 8;
    const dtVals = Array.from({ length: nH }, (_, h) =>
        0.05 + 0.9 * ((Math.sin(h * 0.7 + stepIdx * 0.3) + 1) / 2)
    );
    const barW = (W - 24) / nH;
    const barMaxH = 80;
    const cScale = d3.scaleSequential(d3.interpolateOranges).domain([0, 1]);

    svg.append('text')
        .attr('x', 12).attr('y', 90)
        .attr('fill', '#8b90a8').attr('font-size', 9)
        .text('dt (softplus) per head:');

    for (let h = 0; h < nH; h++) {
        const bh = dtVals[h] * barMaxH;
        const x = 12 + h * barW;
        const y = 100 + barMaxH - bh;
        svg.append('rect')
            .attr('x', x + 1).attr('y', y).attr('width', barW - 2).attr('height', bh)
            .attr('fill', cScale(dtVals[h])).attr('rx', 2);
        svg.append('text')
            .attr('x', x + barW / 2).attr('y', 100 + barMaxH + 11)
            .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 7)
            .text(`h${h}`);
    }

    // Decay visualization
    const decays = Array.from({ length: nH }, (_, h) =>
        Math.exp(-Math.abs(dtVals[h]) * 0.5)
    );
    svg.append('text')
        .attr('x', 12).attr('y', 215)
        .attr('fill', '#8b90a8').attr('font-size', 9)
        .text('exp(A*dt) decay per head:');

    for (let h = 0; h < nH; h++) {
        const x = 12 + h * barW;
        svg.append('rect')
            .attr('x', x + 1).attr('y', 220).attr('width', barW - 2).attr('height', 22)
            .attr('fill', d3.interpolateBlues(1 - decays[h])).attr('rx', 2);
    }

    svg.append('text')
        .attr('x', cx).attr('y', 258)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
        .text('O(T*N) serial per token -- O(1) cache');
}

function renderPrefill(svg: d3.Selection<SVGSVGElement, unknown, null, undefined>) {
    const cx = W / 2;
    const n = CHUNK;

    svg.append('text')
        .attr('x', cx).attr('y', 20)
        .attr('text-anchor', 'middle').attr('fill', '#a78bfa').attr('font-size', 11).attr('font-weight', '700')
        .text('PREFILL MODE  (n_tok > 128)');

    svg.append('text')
        .attr('x', cx).attr('y', 34)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 8)
        .text('SSD matmul path -- NVIDIA Turing+ only  (HIP/AMD falls back to scan)');

    // SSD formula
    svg.append('rect')
        .attr('x', 12).attr('y', 42).attr('width', W - 24).attr('height', 28)
        .attr('fill', '#1a1d27').attr('stroke', '#2e3347').attr('rx', 4);
    svg.append('text')
        .attr('x', cx).attr('y', 60)
        .attr('text-anchor', 'middle').attr('fill', '#e8eaf0').attr('font-family', 'monospace').attr('font-size', 8)
        .text('Y = (L * (C @ B^T)) @ (X * dt)  +  decay * C @ S_init');

    // L matrix (causal decay mask, lower triangular)
    const L = buildDecayMask(n);
    const cellSz = Math.floor(Math.min(180 / n, 12));
    const hmW = n * cellSz;
    const hmH = n * cellSz;
    const hmX = (W - hmW) / 2;
    const hmY = 80;

    svg.append('text')
        .attr('x', cx).attr('y', hmY - 6)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
        .text(`L -- causal decay mask (${n} x ${n}, chunk_size=${CHUNK})`);

    const cScale = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, 1]);
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            const val = L[i * n + j];
            svg.append('rect')
                .attr('x', hmX + j * cellSz).attr('y', hmY + i * cellSz)
                .attr('width', cellSz - 0.3).attr('height', cellSz - 0.3)
                .attr('fill', j <= i ? cScale(val) : '#1a1d27');
        }
    }

    svg.append('text')
        .attr('x', cx).attr('y', hmY + hmH + 18)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 8)
        .text('chunk_size=256 keeps L at O(chunk^2), not O(T^2)');
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
            SSD (n_tok &gt; 128): NVIDIA Turing+ only via cuBLAS.<br/>
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
    color: #000;
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
