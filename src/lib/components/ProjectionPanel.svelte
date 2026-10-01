<script lang="ts">
/**
 * ProjectionPanel.svelte
 * Shows the W_in input projection output (proj_out tensor).
 * Renders the zxBCdt vector split into three color-coded segments:
 *   z   (purple)  -- gate
 *   xBC (teal)    -- conv input
 *   dt  (orange)  -- time step
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';
import { appState, type ModelMeta } from '$lib/data/tensorLoader';

let svgEl: SVGSVGElement;

const W = 300;
const H = 200;
const PAD = { top: 28, right: 16, bottom: 16, left: 48 };

// Color coding used throughout all panels
const COLORS = {
    z:   '#a78bfa',   // purple
    xBC: '#2dd4bf',   // teal
    dt:  '#fb923c',   // orange
};

$: projOut = $appState.tensors.get('proj_out') ?? null;
$: meta    = $appState.meta;

// proj_out = [z | xBC | dt], split dims from the model meta
$: segments = computeSegments(projOut, meta);

function computeSegments(proj: Float32Array | null, m: ModelMeta | null) {
    if (!proj) return null;
    const dInner =
        m?.intermediate_size ??
        (m?.head_dim && m?.num_heads ? m.head_dim * m.num_heads : null);
    const dState = m?.state_size ?? null;
    const nGroups = m?.n_groups ?? null;
    const nHeads = m?.num_heads ?? null;

    if (dInner && dState && nGroups && nHeads) {
        const xBCdim = dInner + 2 * nGroups * dState;
        if (dInner + xBCdim + nHeads === proj.length) {
            return [
                { name: 'z', start: 0, end: dInner, color: COLORS.z },
                { name: 'xBC', start: dInner, end: dInner + xBCdim, color: COLORS.xBC },
                { name: 'dt', start: dInner + xBCdim, end: dInner + xBCdim + nHeads, color: COLORS.dt },
            ];
        }
    }
    // unknown dims: fall back to equal thirds
    const t = Math.floor(proj.length / 3);
    return [
        { name: 'z', start: 0, end: t, color: COLORS.z },
        { name: 'xBC', start: t, end: t * 2, color: COLORS.xBC },
        { name: 'dt', start: t * 2, end: proj.length, color: COLORS.dt },
    ];
}

function render() {
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    if (!projOut || !segments) {
        svg.append('text')
            .attr('x', W / 2).attr('y', H / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#8b90a8').attr('font-size', 12)
            .text('No projection data');
        return;
    }

    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;

    // One bar per pixel column, each the mean of its elements
    const totalDim = projOut.length;
    const xScale = d3.scaleLinear().domain([0, totalDim]).range([PAD.left, PAD.left + plotW]);
    const vMin = d3.min(projOut) ?? -2;
    const vMax = d3.max(projOut) ?? 2;
    const absMax = Math.max(Math.abs(vMin), Math.abs(vMax));
    const cScale = d3.scaleSequential(d3.interpolateRdBu).domain([absMax, -absMax]);

    const g = svg.append('g');
    const nBars = Math.max(segments.length, Math.floor(plotW));
    const binSize = totalDim / nBars;

    for (let b = 0; b < nBars; b++) {
        const start = Math.floor(b * binSize);
        const end = Math.max(start + 1, Math.floor((b + 1) * binSize));
        let sum = 0;
        for (let i = start; i < end; i++) sum += projOut[i];
        const mean = sum / (end - start);
        const mid = Math.floor((start + end) / 2);
        const seg = segments.find((s) => mid >= s.start && mid < s.end);
        const col = seg ? cScale(mean) : '#374151';
        g.append('rect')
            .attr('x', xScale(start))
            .attr('y', PAD.top)
            .attr('width', Math.max(1, xScale(end) - xScale(start) - 0.3))
            .attr('height', plotH)
            .attr('fill', col)
            .append('title')
            .text(`dims [${start}, ${end}) mean = ${mean.toFixed(4)}`);
    }

    // Segment border lines + labels
    for (const seg of segments) {
        const x = xScale(seg.start);
        const x2 = xScale(seg.end);
        const midX = (x + x2) / 2;

        // Bracket
        g.append('rect')
            .attr('x', x).attr('y', PAD.top - 18)
            .attr('width', x2 - x).attr('height', 14)
            .attr('fill', seg.color).attr('rx', 3).attr('opacity', 0.85);

        g.append('text')
            .attr('x', midX).attr('y', PAD.top - 7)
            .attr('text-anchor', 'middle')
            .attr('fill', '#000').attr('font-size', 10).attr('font-weight', '700')
            .text(seg.name);

        // Dim annotation
        g.append('text')
            .attr('x', midX).attr('y', PAD.top + plotH + 14)
            .attr('text-anchor', 'middle')
            .attr('fill', seg.color).attr('font-size', 9)
            .text(`${seg.end - seg.start}d`);
    }

    // Left axis label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -(PAD.top + plotH / 2)).attr('y', 13)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
        .text('value');
}

onMount(render);
afterUpdate(render);
</script>

<div class="proj-wrap">
    <svg bind:this={svgEl} width={W} height={H} />
    <div class="legend">
        <span style="color:#a78bfa">■ z (gate)</span>
        <span style="color:#2dd4bf">■ xBC (conv input)</span>
        <span style="color:#fb923c">■ dt (time step)</span>
    </div>
</div>

<style>
.proj-wrap { display: inline-block; }
.legend {
    display: flex;
    gap: 0.75rem;
    font-size: 0.72rem;
    margin-top: 4px;
    color: var(--text-dim);
}
</style>
