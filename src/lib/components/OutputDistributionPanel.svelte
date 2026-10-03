<script lang="ts">
/**
 * OutputDistributionPanel.svelte
 * Panel 7: always-visible next-token distribution bar chart.
 *
 * Shows the top-20 of the exported top-50 tokens for the currently selected
 * step, as horizontal bars. The actually-following prompt token is highlighted
 * with a ring + rank/prob annotation; entropy of the full softmax distribution
 * is shown as a readout.
 *
 * The distribution is a property of the whole stack (final RMS norm ->
 * lm_head -> logits -> softmax), not of the selected layer, so the panel dims
 * with an explanatory note whenever layer < maxLayer.
 *
 * Data:
 *   - manifest.outputs[p00].steps[t].top_tokens  display strings for the top-20
 *   - manifest.outputs[p00].steps[t].actual_next / actual_rank
 *   - p00_output.npz: probs {T,50}, ids {T,50}, entropy {T} (via tensorLoader)
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';
import { appState, maxLayer } from '$lib/data/tensorLoader';

let svgEl: SVGSVGElement;

const W = 330;
const H = 400;
const N_BARS = 20;
const ROW_H = 15;
const PAD = { top: 40, right: 14, bottom: 92, left: 10 };
const LABEL_W = 78; // right-aligned token label gutter

let actualNote = '';

function render() {
    actualNote = '';
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const s = $appState;
    const pid = `p${String(s?.promptIndex ?? 0).padStart(2, '0')}`;
    const step = s?.stepIndex ?? 0;
    const outInfo = s?.manifest?.outputs?.[pid];
    const outT = s?.outputTensors;

    if (!outInfo || !outT || !outT.has('probs')) {
        svg.append('text')
            .attr('x', W / 2).attr('y', H / 2)
            .attr('text-anchor', 'middle')
            .attr('fill', '#64748b').attr('font-size', 11)
            .text('No output distribution data');
        return;
    }

    const probs = outT.get('probs')!;
    const entropy = outT.get('entropy');
    const stepInfo = outInfo.steps[step];
    const row = probs.subarray(step * 50, step * 50 + 50);
    const top = Array.from(row.subarray(0, N_BARS));
    const maxP = Math.max(...top, 1e-9);

    // Header: step, token, entropy of the full softmax
    svg.append('text')
        .attr('x', PAD.left).attr('y', 16)
        .attr('fill', '#22272e').attr('font-size', 11).attr('font-weight', '700')
        .text(`next token after '${stepInfo?.token ?? '?'}'`);
    if (entropy) {
        svg.append('text')
            .attr('x', W - PAD.right).attr('y', 16)
            .attr('text-anchor', 'end')
            .attr('fill', '#ea580c').attr('font-size', 9).attr('font-family', 'monospace')
            .text(`entropy ${entropy[step].toFixed(2)}`);
    }

    const plotX = PAD.left + LABEL_W;
    const plotW = W - PAD.right - plotX;

    // Bars
    top.forEach((p, i) => {
        const y = PAD.top + i * ROW_H;
        const bw = Math.max(1.5, (p / maxP) * plotW);
        const isActual = stepInfo ? i + 1 === stepInfo.actual_rank : false;

        svg.append('rect') // track
            .attr('x', plotX).attr('y', y)
            .attr('width', plotW).attr('height', ROW_H - 4)
            .attr('fill', '#e8ebf0').attr('rx', 2);

        svg.append('rect') // bar
            .attr('x', plotX).attr('y', y)
            .attr('width', bw).attr('height', ROW_H - 4)
            .attr('fill', isActual ? '#14b8a6' : '#94a3b8')
            .attr('rx', 2)
            .append('title')
            .text(`p = ${p.toExponential(3)}`);

        if (isActual) { // ring highlight for the actual next token
            svg.append('rect')
                .attr('x', plotX - 1.5).attr('y', y - 1.5)
                .attr('width', bw + 3).attr('height', ROW_H - 1)
                .attr('fill', 'none').attr('stroke', '#14b8a6').attr('stroke-width', 1.5)
                .attr('rx', 3);
        }

        const label = stepInfo?.top_tokens[i] ?? `#${i + 1}`;
        svg.append('text')
            .attr('x', plotX - 6).attr('y', y + ROW_H - 7)
            .attr('text-anchor', 'end')
            .attr('fill', isActual ? '#0d9488' : '#64748b')
            .attr('font-size', 9).attr('font-family', 'monospace')
            .text(label.length > 10 ? label.slice(0, 10) : label);

        svg.append('text')
            .attr('x', plotX + bw + 5).attr('y', y + ROW_H - 7)
            .attr('fill', isActual ? '#0d9488' : '#64748b')
            .attr('font-size', 8).attr('font-family', 'monospace')
            .text(p.toFixed(4));
    });

    // Actual-next-token annotation
    if (stepInfo) {
        const r = stepInfo.actual_rank;
        let msg: string;
        if (r >= 1 && r <= N_BARS) {
            msg = `actual next token '${stepInfo.actual_next}' — rank ${r} of 50 (ringed)`;
        } else if (r > N_BARS) {
            msg = `actual next token '${stepInfo.actual_next}' is rank ${r} of 50 (below the top-20 cut)`;
        } else if (stepInfo.actual_next) {
            msg = `actual next token '${stepInfo.actual_next}' is outside the top-50`;
        } else {
            msg = 'end of prompt — no following token';
        }
        const y = PAD.top + N_BARS * ROW_H + 18;
        svg.append('text')
            .attr('x', PAD.left).attr('y', y)
            .attr('fill', '#0d9488').attr('font-size', 9).attr('font-family', 'monospace')
            .text(msg.length > 58 ? msg.slice(0, 57) + '…' : msg);
        actualNote = msg;
    }

    // Caption: the data-flow tail that produced this distribution
    const capY = H - 46;
    for (const [dy, txt] of [
        [0, `hidden_out (layer ${s?.meta?.n_layers ? s.meta.n_layers - 1 : 23})`],
        [13, '  → final RMSNorm → lm_head'],
        [26, '  → logits → softmax → top-50'],
    ] as [number, string][]) {
        svg.append('text')
            .attr('x', PAD.left).attr('y', capY + dy)
            .attr('fill', dy === 0 ? '#64748b' : '#94a3b8')
            .attr('font-size', 8.5).attr('font-family', 'monospace')
            .text(txt);
    }
}

onMount(render);
afterUpdate(render);
</script>

<div class="dist-wrap">
    <svg bind:this={svgEl} width={W} height={H} role="img" aria-label="Next-token distribution chart" />

    {#if $appState.manifest && $appState.layerIndex < $maxLayer}
        <div class="dim-overlay">
            <span>
                Computed from the output of the full {($appState.meta?.n_layers ?? 24)}-layer stack,
                not from layer {$appState.layerIndex} alone — shown dimmed until you select the last layer.
            </span>
        </div>
    {/if}
    <div class="dist-note" title={actualNote}>{actualNote}</div>
</div>

<style>
.dist-wrap {
    position: relative;
    display: inline-block;
}
.dim-overlay {
    position: absolute;
    inset: 40px 0 92px 0;
    background: #f8fafccc;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 0 1rem;
    text-align: center;
    font-size: 0.72rem;
    color: var(--text-dim);
    line-height: 1.5;
    pointer-events: none;
}
.dist-note {
    font-size: 0.68rem;
    font-family: var(--font-mono);
    color: var(--text-dim);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    max-width: 320px;
}
</style>
