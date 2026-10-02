<script lang="ts">
/**
 * ModelDiagramModal.svelte
 * Full-screen overlay rendering the Mamba2 architecture as a mermaid
 * flowchart (see src/lib/data/modelDiagram.ts for the FalkorDB origin and
 * the shared open/close store).
 *
 * - mermaid is dynamically imported on first open to keep the initial
 *   bundle small
 * - Esc key and backdrop click close it
 * - the SVG is scrollable in both axes
 */
import { tick } from 'svelte';
import { MAMBA2_MERMAID, modelDiagramOpen } from '$lib/data/modelDiagram';

let container: HTMLDivElement | null = null;
let mermaidPromise: Promise<any> | null = null;
let renderCount = 0;

function ensureMermaid() {
    if (!mermaidPromise) {
        mermaidPromise = import('mermaid').then((mod: any) => {
            // interop: vite's optimizer and rollup expose mermaid differently
            // (named-only namespace vs default) -- accept both shapes
            const mermaid = mod.default ?? mod;
            mermaid.initialize({
                startOnLoad: false,
                theme: 'dark',
                // htmlLabels uses <foreignObject> measurement which stalls in
                // embedded webviews -- pure SVG text labels render everywhere
                flowchart: { htmlLabels: false, curve: 'basis' },
            });
            return mermaid;
        });
    }
    return mermaidPromise;
}

async function renderDiagram() {
    const el = container;
    if (!el) return;
    el.innerHTML = '<div class="mermaid-loading">importing mermaid …</div>';
    try {
        const mermaid = await ensureMermaid();
        el.innerHTML = '<div class="mermaid-loading">rendering diagram …</div>';
        // unique id per render; mermaid caches by id and refuses re-renders
        renderCount += 1;
        const { svg } = await Promise.race([
            mermaid.render(`mamba2-arch-${renderCount}`, MAMBA2_MERMAID),
            // some embedded webviews stall mermaid's layout measurement;
            // fail visibly instead of spinning forever
            new Promise<never>((_, reject) =>
                setTimeout(() => reject(new Error('render timed out (15s)')), 15000)
            ),
        ]);
        if (container) container.innerHTML = svg;
    } catch (err) {
        console.error('mermaid render failed:', err);
        if (container) {
            container.innerHTML = `<div class="mermaid-error">Diagram rendering failed: ${String(err)}</div>`;
        }
    }
}

$: if ($modelDiagramOpen && container) {
    tick().then(renderDiagram);
}

function close() {
    modelDiagramOpen.set(false);
}

function onKeydown(e: KeyboardEvent) {
    if (e.key === 'Escape') close();
}
</script>

<svelte:window on:keydown={onKeydown} />

{#if $modelDiagramOpen}
    <div
        class="modal-backdrop"
        on:click|self={close}
        role="dialog"
        aria-modal="true"
        aria-label="Mamba2 architecture diagram"
    >
        <div class="modal-content">
            <div class="modal-header">
                <h2 class="modal-title">Mamba2 Architecture</h2>
                <span class="modal-sub">derived from the arch:llama.cpp:mamba2-ssm graph</span>
                <button class="modal-close" on:click={close} aria-label="Close diagram">✕</button>
            </div>
            <div class="modal-body">
                <div class="mermaid-container" bind:this={container}></div>
            </div>
            <div class="modal-footer">
                <span class="modal-legend">
                    <span class="lg lg-op">op</span>
                    <span class="lg lg-tensor">tensor</span>
                    <span class="lg lg-state">state</span>
                    <span class="lg lg-weight">weight</span>
                    <span class="lg lg-writeback">write-back</span>
                </span>
                <button class="modal-btn" on:click={close}>Close (Esc)</button>
            </div>
        </div>
    </div>
{/if}

<style>
.modal-backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.8);
    z-index: 1000;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 1rem;
}

.modal-content {
    background: var(--bg-card, #141824);
    border: 1px solid var(--border, #2e3347);
    border-radius: 12px;
    width: 100%;
    max-width: 960px;
    max-height: 85vh;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    box-shadow: 0 20px 60px #00000050;
}

.modal-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
    padding: 0.9rem 1.5rem;
    border-bottom: 1px solid var(--border, #2e3347);
}

.modal-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-teal, #2dd4bf);
    margin: 0;
    white-space: nowrap;
}

.modal-sub {
    font-size: 0.7rem;
    color: var(--text-dim, #8b90a8);
    font-family: var(--font-mono, monospace);
    flex: 1;
}

.modal-close {
    background: none;
    border: none;
    color: var(--text-dim, #8b90a8);
    font-size: 1.3rem;
    cursor: pointer;
    padding: 4px 8px;
    border-radius: 4px;
    transition: background 0.2s;
}
.modal-close:hover {
    background: var(--bg-panel, #1a1d27);
    color: var(--text, #e8eaf0);
}

.modal-body {
    padding: 1rem;
    flex: 1;
    overflow: auto;
}

.mermaid-container {
    min-width: 100%;
    display: flex;
    justify-content: center;
}
.mermaid-container :global(svg) {
    max-width: none;
}

:global(.mermaid-loading),
:global(.mermaid-error) {
    color: var(--text-dim, #8b90a8);
    font-size: 0.85rem;
    padding: 2rem;
}
:global(.mermaid-error) {
    color: #fca5a5;
}

.modal-footer {
    padding: 0.6rem 1.5rem;
    border-top: 1px solid var(--border, #2e3347);
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 1rem;
}

.modal-legend {
    display: flex;
    gap: 0.6rem;
    font-size: 0.68rem;
    color: var(--text-dim, #8b90a8);
    flex-wrap: wrap;
}
.lg {
    padding: 1px 7px;
    border-radius: 3px;
    border: 1px solid transparent;
}
.lg-op { background: #0e7490; color: #e0f2fe; }
.lg-tensor { background: #1e293b; color: #e2e8f0; border-color: #64748b; }
.lg-state { background: #064e3b; color: #d1fae5; border-color: #34d399; }
.lg-weight { background: #1a1d27; color: #c4b5fd; border-color: #a78bfa; border-style: dotted; }
.lg-writeback { background: #4c1d95; color: #ede9fe; border-color: #a78bfa; border-style: dashed; }

.modal-btn {
    background: var(--accent-teal, #2dd4bf);
    color: #000;
    border: none;
    border-radius: 6px;
    padding: 6px 12px;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    white-space: nowrap;
    transition: background 0.2s;
}
.modal-btn:hover {
    background: #46e0cd;
}
</style>
