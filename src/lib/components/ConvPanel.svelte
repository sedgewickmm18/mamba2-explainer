<script lang="ts">
/**
 * ConvPanel.svelte
 * Shows the r_l conv history sliding window from the recurrent state cache:
 * channels x d_conv, columns [x_{t-d_conv+1} .. x_t], zero-padded at the start.
 * When step advances: leftmost column slides out, new column slides in.
 * Badge: "State size is constant - does not grow with context length".
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';
import { appState } from '$lib/data/tensorLoader';

let svgEl: SVGSVGElement;

const W = 300;
const H = 220;
const PAD = { top: 44, right: 12, bottom: 28, left: 12 };
const MAX_ROWS = 24; // displayed channels, evenly sampled

// conv_state is {channels, d_conv} flattened row-major
$: convState = $appState.tensors.get('conv_state') ?? null;
$: dConv     = $appState.meta?.conv_kernel_size ?? 4;

function render() {
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    if (!convState || convState.length === 0) {
        svg.append('text')
            .attr('x', W / 2).attr('y', H / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#8b90a8').attr('font-size', 12)
            .text('No conv state data');
        return;
    }

    const channels = Math.floor(convState.length / dConv);
    const hist = dConv - 1;
    const nRows = Math.min(MAX_ROWS, channels);
    const plotW = W - PAD.left - PAD.right;
    const plotH = H - PAD.top - PAD.bottom;
    const cellW = plotW / (hist + 1.5);  // +1.5 for new-token column
    const cellH = plotH / nRows;

    // Sample row indices so the displayed channels span the full width
    const rows = Array.from({ length: nRows }, (_, r) =>
        Math.round((r * (channels - 1)) / Math.max(1, nRows - 1))
    );

    // conv[c * dConv + k] is channel c, window column k; last column is the new token
    const val = (ch: number, k: number) => convState![ch * dConv + k] ?? 0;

    const allVals: number[] = [];
    for (const ch of rows) for (let k = 0; k < dConv; k++) allVals.push(val(ch, k));
    const absMax = Math.max(0.01, ...allVals.map(Math.abs));
    const cScale = d3.scaleSequential(d3.interpolateRdBu).domain([absMax, -absMax]);

    const g = svg.append('g');

    // Header labels
    svg.append('text')
        .attr('x', W / 2).attr('y', 16)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 10)
        .text(`Conv history r_l  (${hist} slots x ${nRows} of ${channels} channels, d_conv=${dConv})`);

    // History columns
    for (let c = 0; c < hist; c++) {
        const x = PAD.left + c * cellW;
        for (let r = 0; r < nRows; r++) {
            const y = PAD.top + r * cellH;
            const v = val(rows[r], c);
            g.append('rect')
                .attr('x', x + 1).attr('y', y + 1)
                .attr('width', cellW - 2).attr('height', cellH - 2)
                .attr('fill', cScale(v)).attr('rx', 1)
                .append('title').text(`history[${c}][ch ${rows[r]}] = ${v.toFixed(4)}`);
        }
        // Column label
        svg.append('text')
            .attr('x', x + cellW / 2).attr('y', PAD.top - 6)
            .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
            .text(`t-${hist - c}`);
    }

    // Separator + "new token" column
    const sepX = PAD.left + hist * cellW + 8;
    g.append('line')
        .attr('x1', sepX).attr('x2', sepX)
        .attr('y1', PAD.top - 2).attr('y2', PAD.top + plotH)
        .attr('stroke', '#2dd4bf').attr('stroke-width', 1.5).attr('stroke-dasharray', '3,3');

    const newX = sepX + 8;
    for (let r = 0; r < nRows; r++) {
        const y = PAD.top + r * cellH;
        const v = val(rows[r], dConv - 1);
        g.append('rect')
            .attr('x', newX).attr('y', y + 1)
            .attr('width', cellW - 2).attr('height', cellH - 2)
            .attr('fill', cScale(v)).attr('rx', 1).attr('stroke', '#2dd4bf').attr('stroke-width', 0.5)
            .append('title').text(`new[ch ${rows[r]}] = ${v.toFixed(4)}`);
    }
    svg.append('text')
        .attr('x', newX + cellW / 2).attr('y', PAD.top - 6)
        .attr('text-anchor', 'middle').attr('fill', '#2dd4bf').attr('font-size', 9)
        .text('t (new)');

    // Bottom: state size badge
    svg.append('text')
        .attr('x', W / 2).attr('y', H - 6)
        .attr('text-anchor', 'middle').attr('fill', '#4ade80').attr('font-size', 9)
        .text(`State = ${hist} x channels -- constant, does not grow with context`);
}

onMount(render);
afterUpdate(render);
</script>

<div class="conv-wrap">
    <svg bind:this={svgEl} width={W} height={H} />
    <div class="conv-note">
        After this step, the leftmost column (t-{dConv - 1}) is discarded
        and the new token slides in. Buffer size stays fixed.
    </div>
</div>

<style>
.conv-wrap { display: inline-block; }
.conv-note {
    font-size: 0.72rem;
    color: var(--text-dim);
    margin-top: 4px;
    max-width: 300px;
    line-height: 1.4;
}
</style>
