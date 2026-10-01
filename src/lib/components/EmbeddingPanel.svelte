<script lang="ts">
/**
 * EmbeddingPanel.svelte
 * Shows the input token sequence as colored rectangles.
 * Each bar height represents the embedding vector magnitude.
 * Hover shows miniature heatmap of the 768-dim vector.
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
$: hiddenIn = $appState.tensors.get('hidden_in') ?? null;

let tooltip: { visible: boolean; x: number; y: number; token: string; mag: number } = {
    visible: false, x: 0, y: 0, token: '', mag: 0,
};

function magnitude(arr: Float32Array): number {
    let s = 0;
    for (let i = 0; i < arr.length; i++) s += arr[i] * arr[i];
    return Math.sqrt(s);
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

    // Magnitude is a placeholder until we load per-token embeddings.
    // Use stepIdx to highlight active token, and a synthetic magnitude.
    const magnitudes = tokens.map((_, i) => (i <= stepIdx ? 0.6 + 0.4 * Math.sin(i * 0.9) : 0.1));
    const yScale = d3.scaleLinear().domain([0, 1]).range([H - PAD.bottom, PAD.top]);

    const colorScale = d3.scaleSequential(d3.interpolatePlasma).domain([0, 1]);

    const g = svg.append('g');

    // Axes
    g.append('line')
        .attr('x1', PAD.left).attr('x2', W - PAD.right)
        .attr('y1', H - PAD.bottom).attr('y2', H - PAD.bottom)
        .attr('stroke', '#2e3347').attr('stroke-width', 1);

    // Bars
    tokens.forEach((tok, i) => {
        const mag = magnitudes[i];
        const x = xScale(i) + (xScale(1) - xScale(0) - barW) / 2;
        const yTop = yScale(mag);
        const barH = H - PAD.bottom - yTop;

        const rect = g.append('rect')
            .attr('x', x)
            .attr('y', yTop)
            .attr('width', barW)
            .attr('height', Math.max(2, barH))
            .attr('fill', i === stepIdx ? '#2dd4bf' : colorScale(mag))
            .attr('rx', 2)
            .attr('opacity', i <= stepIdx ? 1 : 0.3)
            .style('cursor', 'pointer');

        rect.on('mouseenter', function (event) {
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
                .attr('fill', i === stepIdx ? '#2dd4bf' : '#8b90a8')
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
        .attr('fill', '#8b90a8')
        .attr('font-size', 9)
        .text('|embedding|');
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
            <div class="tt-mag">||emb|| = {tooltip.mag.toFixed(3)}</div>
            {#if hiddenIn}
                <div class="tt-hint">d_model = {hiddenIn.length}</div>
            {/if}
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
    box-shadow: 0 4px 16px #0006;
}
.tt-token { font-family: var(--font-mono); color: var(--accent-teal); margin-bottom: 2px; }
.tt-mag   { color: var(--text-dim); }
.tt-hint  { color: var(--text-dim); font-size: 0.72rem; margin-top: 2px; }
</style>
