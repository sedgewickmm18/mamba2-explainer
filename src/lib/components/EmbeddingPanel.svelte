<script lang="ts">
/**
 * EmbeddingPanel.svelte
 * Shows the input token sequence as colored rectangles.
 * Bar height = the REAL ||embedding|| of each token, taken from the
 * 'embeddings' member of the per-prompt output npz ({T, d_model}, the
 * layer-0 inputs captured during the token-by-token forward).
 * Hover shows the exact magnitude.
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';
import { appState } from '$lib/data/tensorLoader';

let svgEl: SVGSVGElement;

const W = 300;
const H = 160;
const PAD = { top: 32, right: 12, bottom: 28, left: 12 };

$: tokens  = $appState.tokens;
$: stepIdx = $appState.stepIndex;
$: step    = $appState.stepIndex;
$: prompt  = $appState.promptIndex;
$: outT    = $appState.outputTensors;

let tooltip: { visible: boolean; x: number; y: number; token: string; mag: number } = {
    visible: false, x: 0, y: 0, token: '', mag: 0,
};

function magnitude(arr: Float32Array, row: number, d: number): number {
    let s = 0;
    const base = row * d;
    for (let i = 0; i < d; i++) {
        const v = arr[base + i] ?? 0;
        s += v * v;
    }
    return Math.sqrt(s);
}

// Real per-token magnitudes from the exported embeddings {T, d_model}
$: magnitudes = computeMagnitudes(outT, tokens.length);

function computeMagnitudes(out: Map<string, Float32Array> | null, n: number): number[] {
    const emb = out?.get('embeddings');
    if (!emb || emb.length < n) {
        return new Array(n).fill(NaN); // no data yet -- rendered as empty stubs
    }
    const d = emb.length / n;
    const result = new Array(n);
    for (let i = 0; i < n; i++) result[i] = magnitude(emb, i, d);
    return result;
}

function render() {
    if (!svgEl || tokens.length === 0) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const n = tokens.length;
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    const barW = Math.max(4, Math.min(32, plotW / n - 2));
    const xScale = d3.scaleLinear().domain([0, n]).range([PAD.left, PAD.left + plotW]);

    const valid = magnitudes.filter((m) => !isNaN(m));
    const yMax = valid.length ? Math.max(...valid) : 1;
    const yScale = d3.scaleLinear().domain([0, yMax]).range([H - PAD.bottom, PAD.top]);
    const colorScale = d3.scaleSequential(d3.interpolatePlasma).domain([0, yMax]);

    const g = svg.append('g');

    // Axes
    g.append('line')
        .attr('x1', PAD.left).attr('x2', W - PAD.right)
        .attr('y1', H - PAD.bottom).attr('y2', H - PAD.bottom)
        .attr('stroke', '#c9cfda').attr('stroke-width', 1);

    // Bars
    tokens.forEach((tok, i) => {
        const mag = magnitudes[i];
        const has = !isNaN(mag);
        const x = xScale(i) + (xScale(1) - xScale(0) - barW) / 2;
        const yTop = has ? yScale(mag) : H - PAD.bottom - 4;
        const barH = has ? Math.max(2, H - PAD.bottom - yTop) : 4;

        const rect = g.append('rect')
            .attr('x', x)
            .attr('y', yTop)
            .attr('width', barW)
            .attr('height', barH)
            .attr('fill', i === stepIdx ? '#14b8a6' : (has ? colorScale(mag) : '#cbd5e1'))
            .attr('rx', 2)
            .attr('opacity', i <= stepIdx ? 1 : 0.3)
            .style('cursor', 'pointer');

        rect.on('mouseenter', (event: MouseEvent) => {
            tooltip = {
                visible: true,
                x: event.offsetX + 12,
                y: event.offsetY - 8,
                token: tok,
                mag,
            };
        });
        rect.on('mouseleave', () => { tooltip = { ...tooltip, visible: false }; });

        // Token label below bar
        if (barW >= 10) {
            g.append('text')
                .attr('x', x + barW / 2)
                .attr('y', H - PAD.bottom + 14)
                .attr('text-anchor', 'middle')
                .attr('fill', i === stepIdx ? '#0d9488' : '#64748b')
                .attr('font-size', Math.min(10, barW - 2))
                .attr('font-family', 'monospace')
                .text(tok.length > 5 ? tok.slice(0, 5) : tok);
        }
    });

    // Y-axis label
    svg.append('text')
        .attr('transform', `rotate(-90)`)
        .attr('x', -(H / 2))
        .attr('y', 11)
        .attr('text-anchor', 'middle')
        .attr('fill', '#64748b')
        .attr('font-size', 9)
        .text('|embedding| (real)');
}

onMount(render);
afterUpdate(render);
</script>

<div class="embedding-wrap" style="position:relative">
    <svg bind:this={svgEl} width={W} height={H} />

    {#if tooltip.visible}
        <div
            class="tooltip"
            style="left:{tooltip.x}px;top:{tooltip.y}px"
        >
            <div class="tt-token">{tooltip.token}</div>
            <div class="tt-mag">{isNaN(tooltip.mag) ? 'no data' : `||emb|| = ${tooltip.mag.toFixed(3)}`}</div>
        </div>
    {/if}
</div>

<style>
.embedding-wrap { display: inline-block; }

.tooltip {
    position: absolute;
    pointer-events: none;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 0.78rem;
    color: var(--text);
    z-index: 50;
    white-space: nowrap;
    box-shadow: 0 4px 16px rgba(30, 41, 59, 0.15);
}
.tt-token { font-family: var(--font-mono); color: var(--accent-teal); margin-bottom: 2px; }
.tt-mag   { color: var(--text-dim); }
</style>
