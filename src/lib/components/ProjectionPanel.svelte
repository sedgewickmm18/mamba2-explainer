<script lang="ts">
/**
 * ProjectionPanel.svelte
 * Shows the W_in input projection output (proj_out tensor).
 * Renders the zxBCdt vector split into three color-coded segments:
 *   z   (purple)  -- gate
 *   xBC (teal)    -- conv input
 *   dt  (orange)  -- time step
 *
 * Each segment ends in a small exit chevron with an invisible 1x1 anchor div
 * placed below it. +page.svelte measures these anchors to route the stage
 * fan-out arrows (z -> SwiGLU gate, xBC -> conv, dt -> scan) and redraws them
 * whenever the 'fanout' event fires (anchors repositioned).
 */
import { onMount, afterUpdate, createEventDispatcher } from 'svelte';
import * as d3 from 'd3';
import { appState, type ModelMeta } from '$lib/data/tensorLoader';

const dispatch = createEventDispatcher();

let svgEl: SVGSVGElement;
let anchorEls: HTMLDivElement[] = [];

const W = 300;
const H = 224;
const PAD = { top: 28, right: 16, bottom: 34, left: 48 };

// Color coding used throughout all panels:
//   fill -- brackets / bars / arrow strokes, text -- labels on light bg
const COLORS = {
    z:   { fill: '#a78bfa', text: '#7c3aed' },   // purple
    xBC: { fill: '#14b8a6', text: '#0d9488' },   // teal
    dt:  { fill: '#fb923c', text: '#ea580c' },   // orange
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
                { name: 'z',   start: 0, end: dInner, color: COLORS.z.fill, textColor: COLORS.z.text },
                { name: 'xBC', start: dInner, end: dInner + xBCdim, color: COLORS.xBC.fill, textColor: COLORS.xBC.text },
                { name: 'dt',  start: dInner + xBCdim, end: dInner + xBCdim + nHeads, color: COLORS.dt.fill, textColor: COLORS.dt.text },
            ];
        }
    }
    // unknown dims: fall back to equal thirds
    const t = Math.floor(proj.length / 3);
    return [
        { name: 'z',   start: 0, end: t, color: COLORS.z.fill, textColor: COLORS.z.text },
        { name: 'xBC', start: t, end: t * 2, color: COLORS.xBC.fill, textColor: COLORS.xBC.text },
        { name: 'dt',  start: t * 2, end: proj.length, color: COLORS.dt.fill, textColor: COLORS.dt.text },
    ];
}

function hideAnchors() {
    for (const el of anchorEls) if (el) el.style.display = 'none';
    dispatch('fanout');
}

function positionAnchors(
    xScale: (v: number) => number,
    segs: { name: string; start: number; end: number }[]
) {
    segs.forEach((seg, i) => {
        const el = anchorEls[i];
        if (!el) return;
        el.style.display = 'block';
        el.style.left = `${(xScale(seg.start) + xScale(seg.end)) / 2}px`;
        el.style.top = `${H - 4}px`;
    });
    dispatch('fanout');
}

function render() {
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    if (!projOut || !segments) {
        svg.append('text')
            .attr('x', W / 2).attr('y', H / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#64748b').attr('font-size', 12)
            .text('No projection data');
        hideAnchors();
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
        const col = seg ? cScale(mean) : '#d1d5db';
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
    const chevronY = PAD.top + plotH + 22;
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
            .attr('fill', seg.textColor).attr('font-size', 9)
            .text(`${seg.end - seg.start}d`);

        // Exit chevron: marks where this component leaves the panel.
        // The stage fan-out overlay picks the arrow up right below it.
        g.append('line')
            .attr('x1', midX).attr('x2', midX)
            .attr('y1', chevronY).attr('y2', chevronY + 7)
            .attr('stroke', seg.color).attr('stroke-width', 2);
        g.append('path')
            .attr('d', `M ${midX - 3.5} ${chevronY + 4} L ${midX} ${chevronY + 8} L ${midX + 3.5} ${chevronY + 4}`)
            .attr('fill', 'none')
            .attr('stroke', seg.color).attr('stroke-width', 2)
            .attr('stroke-linecap', 'round').attr('stroke-linejoin', 'round');
    }

    // Left axis label
    svg.append('text')
        .attr('transform', 'rotate(-90)')
        .attr('x', -(PAD.top + plotH / 2)).attr('y', 13)
        .attr('text-anchor', 'middle').attr('fill', '#64748b').attr('font-size', 9)
        .text('value');

    positionAnchors(xScale, segments);
}

onMount(render);
afterUpdate(render);
</script>

<div class="proj-wrap">
    <div class="legend">
        <span style="color:{COLORS.z.text}">■ z (gate)</span>
        <span style="color:{COLORS.xBC.text}">■ xBC (conv input)</span>
        <span style="color:{COLORS.dt.text}">■ dt (time step)</span>
    </div>
    <div class="proj-chart">
        <svg bind:this={svgEl} width={W} height={H} />
        {#each segments ?? [] as seg, i}
            <div class="seg-anchor" bind:this={anchorEls[i]} data-seg={seg.name}></div>
        {/each}
    </div>
</div>

<style>
.proj-wrap { display: inline-block; }
.proj-chart { position: relative; }
.legend {
    display: flex;
    gap: 0.75rem;
    font-size: 0.72rem;
    margin-bottom: 4px;
    color: var(--text-dim);
}
/* Invisible measurement point for the stage fan-out arrows */
.seg-anchor {
    position: absolute;
    width: 1px;
    height: 1px;
    pointer-events: none;
}
</style>
