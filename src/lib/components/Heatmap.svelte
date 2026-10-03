<script lang="ts">
/**
 * Heatmap.svelte - reusable D3 heatmap for float32 arrays.
 * Props:
 *   data    - Float32Array, row-major
 *   rows    - number of rows
 *   cols    - number of columns
 *   width   - SVG width in px
 *   height  - SVG height in px
 *   label   - optional title
 *   colorMin / colorMax  - clamp range (defaults: auto)
 */
import { onMount, afterUpdate } from 'svelte';
import * as d3 from 'd3';

export let data: Float32Array | null = null;
export let rows: number = 1;
export let cols: number = 1;
export let width: number = 200;
export let height: number = 120;
export let label: string = '';
export let colorMin: number | null = null;
export let colorMax: number | null = null;
export let showValues: boolean = false;

let svgEl: SVGSVGElement;

const PAD_TOP = label ? 20 : 4;
const PAD = 4;

$: cellW = (width - PAD * 2) / cols;
$: cellH = (height - PAD_TOP - PAD) / rows;

function colorScale(arr: Float32Array) {
    const mn = colorMin ?? d3.min(arr) ?? -1;
    const mx = colorMax ?? d3.max(arr) ?? 1;
    const absMax = Math.max(Math.abs(mn), Math.abs(mx));
    return d3.scaleSequential(d3.interpolateRdBu).domain([absMax, -absMax]);
}

function render() {
    if (!svgEl || !data || data.length === 0) return;
    const svg = d3.select(svgEl);
    svg.selectAll('*').remove();

    const scale = colorScale(data);

    if (label) {
        svg.append('text')
            .attr('x', width / 2)
            .attr('y', 14)
            .attr('text-anchor', 'middle')
            .attr('fill', '#64748b')
            .attr('font-size', 11)
            .text(label);
    }

    const g = svg.append('g').attr('transform', `translate(${PAD},${PAD_TOP})`);

    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            const val = data[r * cols + c] ?? 0;
            const x = c * cellW;
            const y = r * cellH;

            const rect = g.append('rect')
                .attr('x', x)
                .attr('y', y)
                .attr('width', cellW - 0.5)
                .attr('height', cellH - 0.5)
                .attr('fill', scale(val));

            if (showValues && cellW > 28 && cellH > 14) {
                g.append('text')
                    .attr('x', x + cellW / 2)
                    .attr('y', y + cellH / 2 + 4)
                    .attr('text-anchor', 'middle')
                    .attr('fill', '#111827')
                    .attr('font-size', 9)
                    .attr('pointer-events', 'none')
                    .text(val.toFixed(2));
            }

            rect.append('title').text(`[${r},${c}] = ${val.toFixed(4)}`);
        }
    }
}

onMount(render);
afterUpdate(render);
</script>

<svg bind:this={svgEl} {width} {height} class="heatmap" />

<style>
.heatmap {
    display: block;
    border-radius: 4px;
    overflow: hidden;
}
</style>
