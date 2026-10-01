<script lang="ts">
/**
 * GateOutputPanel.svelte
 * SwiGLU gate + skip connection + grouped RMS norm + W_out + residual add.
 *
 * Layout:
 *   [z vector] -- sigmoid --> [gate] --\
 *                                       x --> [y_gated]
 *   [y_ssm]   -------------------------/
 *   [y_ssm + D*x] (skip) --> [y_gated] --> GroupNorm --> W_out --> + residual
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';
import { appState } from '$lib/data/tensorLoader';

let svgEl: SVGSVGElement;

const W = 300;
const H = 300;
const VEC_ROWS = 24;   // show first 24 elements

$: stepIdx  = $appState.stepIndex;
$: hiddenIn = $appState.tensors.get('hidden_in') ?? null;
$: hiddenOut = $appState.tensors.get('hidden_out') ?? null;

// Derive synthetic z, y_ssm, gate from available data
$: vecData = computeVecData();

interface VecData {
    z:      Float32Array;
    ySsm:   Float32Array;
    gate:   Float32Array;
    yGated: Float32Array;
    yOut:   Float32Array;
}

function sigmoid(x: number): number { return 1 / (1 + Math.exp(-x)); }

function computeVecData(): VecData {
    const dim = VEC_ROWS;
    const z = new Float32Array(dim);
    const ySsm = new Float32Array(dim);
    const gate = new Float32Array(dim);
    const yGated = new Float32Array(dim);
    const yOut = new Float32Array(dim);

    // Use hidden_in as proxy for z and hidden_out as proxy for y_ssm
    const base = hiddenIn ?? new Float32Array(dim);
    const out  = hiddenOut ?? new Float32Array(dim);

    for (let i = 0; i < dim; i++) {
        z[i]      = base[i % base.length] ?? 0;
        ySsm[i]   = out[i % out.length] ?? 0;
        gate[i]   = sigmoid(z[i]);
        yGated[i] = gate[i] * ySsm[i];
        // skip: D*x + y_gated (D=1 for synthetic)
        yOut[i]   = yGated[i] + 0.1 * z[i];
    }
    return { z, ySsm, gate, yGated, yOut };
}

function renderVec(
    svg: d3.Selection<SVGSVGElement, unknown, null, undefined>,
    data: Float32Array,
    x: number,
    y: number,
    w: number,
    h: number,
    color: string,
    label: string
) {
    const cellH = h / data.length;
    const absMax = Math.max(0.01, Math.max(...data.map(Math.abs)));
    const cScale = d3.scaleSequential(d3.interpolateRdBu).domain([absMax, -absMax]);

    for (let i = 0; i < data.length; i++) {
        svg.append('rect')
            .attr('x', x).attr('y', y + i * cellH)
            .attr('width', w).attr('height', cellH - 0.5)
            .attr('fill', cScale(data[i]));
    }
    svg.append('text')
        .attr('x', x + w / 2).attr('y', y - 5)
        .attr('text-anchor', 'middle').attr('fill', color).attr('font-size', 9)
        .text(label);
}

function render() {
    if (!svgEl) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const vd = vecData;
    const vecH = 160;
    const vecW = 22;
    const top = 30;
    const spacing = 40;

    // --- Row 1: z, gate, y_ssm, y_gated ---
    const cols = [
        { data: vd.z,      color: '#a78bfa', label: 'z (gate in)' },
        { data: vd.gate,   color: '#f9a8d4', label: 'sigmoid(z)' },
        { data: vd.ySsm,   color: '#2dd4bf', label: 'y_ssm' },
        { data: vd.yGated, color: '#4ade80', label: 'y_gated' },
    ];

    let x = 16;
    for (const col of cols) {
        renderVec(svg, col.data, x, top, vecW, vecH, col.color, col.label);
        x += vecW + spacing;
    }

    // Arrows
    // sigmoid: z -> gate
    svg.append('text').attr('x', 16 + vecW + 6).attr('y', top + vecH / 2 + 5)
        .attr('fill', '#8b90a8').attr('font-size', 10).text('->');
    // element-wise multiply
    svg.append('text').attr('x', 16 + 2 * (vecW + spacing) - spacing + vecW + 2).attr('y', top + vecH / 2 + 5)
        .attr('fill', '#8b90a8').attr('font-size', 10).text('x');
    // result arrow
    svg.append('text').attr('x', 16 + 3 * (vecW + spacing) - spacing + vecW + 2).attr('y', top + vecH / 2 + 5)
        .attr('fill', '#8b90a8').attr('font-size', 10).text('=');

    // --- Row 2: annotations ---
    const midY = top + vecH + 22;
    svg.append('text')
        .attr('x', W / 2).attr('y', midY)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
        .text('y = sigmoid(z) * y_ssm  (SwiGLU gate)');

    // Skip connection note
    svg.append('text')
        .attr('x', W / 2).attr('y', midY + 14)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
        .text('y += D * x  (skip, D per head)');

    // Output vector
    const outTop = midY + 28;
    renderVec(svg, vd.yOut, (W - vecW) / 2, outTop, vecW, 60, '#60a5fa', 'y_out');

    svg.append('text')
        .attr('x', W / 2).attr('y', outTop + 75)
        .attr('text-anchor', 'middle').attr('fill', '#8b90a8').attr('font-size', 9)
        .text('-> GroupRMSNorm -> W_out -> + residual');

    // Residual arrow
    const residY = outTop + 90;
    svg.append('line')
        .attr('x1', 16).attr('y1', top + vecH / 2)
        .attr('x2', 16).attr('y2', residY)
        .attr('stroke', '#60a5fa').attr('stroke-width', 1).attr('stroke-dasharray', '3,2');
    svg.append('text')
        .attr('x', 28).attr('y', residY + 4)
        .attr('fill', '#60a5fa').attr('font-size', 8)
        .text('x_residual (bypass)');
}

onMount(render);
afterUpdate(render);
</script>

<div class="gate-wrap">
    <svg bind:this={svgEl} width={W} height={H} />
</div>

<style>
.gate-wrap { display: inline-block; }
</style>
