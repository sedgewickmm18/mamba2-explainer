<script lang="ts">
/**
 * GuidedTour.svelte
 * 15-step guided tour overlay with prev/next navigation.
 * Opens automatically on first visit (localStorage flag).
 * Each step highlights a panel, scrolls to it, shows a tooltip card.
 */
import { onMount } from 'svelte';
import { writable } from 'svelte/store';

interface TourStep {
    title: string;
    body: string;
    panel: string;    // id of the stage panel element
    step: number;     // 1-indexed for display
}

const TOUR_STEPS: TourStep[] = [
    {
        step: 1,
        panel: 'panel-embedding',
        title: 'What is Mamba2?',
        body: 'Mamba2 is a State Space Model (SSM). Unlike Transformers, it processes tokens one at a time using a fixed-size state matrix S -- no growing KV cache.',
    },
    {
        step: 2,
        panel: 'panel-ssm',
        title: 'Fixed state size',
        body: 'The SSM state S has size d_state x head_dim x n_head. This is CONSTANT regardless of context length -- 50k tokens uses the same memory as 5 tokens.',
    },
    {
        step: 3,
        panel: 'panel-embedding',
        title: 'Input: token embedding',
        body: 'Each input token is converted to a dense vector of size d_model (e.g. 768 for mamba2-130m). Bar height = vector magnitude. Active token is highlighted.',
    },
    {
        step: 4,
        panel: 'panel-projection',
        title: 'Input projection W_in',
        body: 'A single large GEMM projects the embedding to three outputs: z (purple gate), xBC (teal conv input), dt (orange time step). One matrix, three roles.',
    },
    {
        step: 5,
        panel: 'panel-conv',
        title: 'Conv history r_l',
        body: 'Before the SSM, a 1-D depthwise convolution mixes the last d_conv=4 tokens. The r_l buffer holds the last 3 tokens (d_conv-1). It slides: oldest discarded, newest added.',
    },
    {
        step: 6,
        panel: 'panel-conv',
        title: '1-D convolution',
        body: 'The convolution kernel W_conv (4 values per channel) is applied to the sliding window. Output is activated with SiLU. This gives local context within the fixed state.',
    },
    {
        step: 7,
        panel: 'panel-projection',
        title: 'x, B, C split',
        body: 'Conv output is split: x (values, head_dim x n_head), B (keys, d_state x n_group), C (queries, d_state x n_group). B and C are per-token, not stored -- unlike Transformer KV.',
    },
    {
        step: 8,
        panel: 'panel-scan',
        title: 'dt: time step',
        body: 'dt (after softplus + bias) controls how much new information to write into the state vs retain old state. Larger dt = faster forgetting and faster learning.',
    },
    {
        step: 9,
        panel: 'panel-ssm',
        title: 'A decay parameter',
        body: 'A is a LEARNED scalar per head (negative, so exp(A*dt) is in (0,1)). It sets the base decay rate. Darker bar = more forgetting this step.',
    },
    {
        step: 10,
        panel: 'panel-ssm',
        title: 'State update',
        body: 'S_t = exp(A*dt) * S_{t-1} + dt * outer(B, x). Left panel = old state, middle = update (dt * B*x^T), right = new state. Watch the heatmap change with each token.',
    },
    {
        step: 11,
        panel: 'panel-ssm',
        title: 'Readout',
        body: 'y_t = C_t @ S_t. The query C selects which parts of the state to read out as output. This is the SSM equivalent of attention scoring.',
    },
    {
        step: 12,
        panel: 'panel-gate',
        title: 'Skip + SwiGLU gate',
        body: 'y += D*x adds a skip connection (D per head). Then y = sigmoid(z) * y applies the SwiGLU gate. z (purple) modulates how much SSM output passes through.',
    },
    {
        step: 13,
        panel: 'panel-gate',
        title: 'Grouped norm + output',
        body: 'Grouped RMS norm stabilises within-group activations. W_out projects back to d_model. Residual add completes the layer.',
    },
    {
        step: 14,
        panel: 'panel-scan',
        title: 'Decode vs prefill',
        body: 'For single-token decoding (n_tok=1): sequential scan, O(1) state. For long prefill (n_tok>128, NVIDIA only): SSD matmul path -- chunked at 256 to keep matrix sizes manageable.',
    },
    {
        step: 15,
        panel: 'panel-ssm',
        title: 'Why this matters',
        body: 'At 50k tokens a Transformer KV cache is ~gigabytes. The SSM state is a few KB regardless of length. Decode speed stays flat. This is the core engineering advantage of Mamba2.',
    },
];

const tourVisible = writable(false);
const tourStep = writable(0);

onMount(() => {
    const seen = typeof localStorage !== 'undefined' && localStorage.getItem('mamba2-tour-seen');
    if (!seen) {
        tourVisible.set(true);
    }
});

function close() {
    tourVisible.set(false);
    if (typeof localStorage !== 'undefined') {
        localStorage.setItem('mamba2-tour-seen', '1');
    }
}

function next() {
    tourStep.update((s) => {
        const n = Math.min(s + 1, TOUR_STEPS.length - 1);
        scrollToPanel(TOUR_STEPS[n].panel);
        return n;
    });
}

function prev() {
    tourStep.update((s) => {
        const n = Math.max(s - 1, 0);
        scrollToPanel(TOUR_STEPS[n].panel);
        return n;
    });
}

function scrollToPanel(panelId: string) {
    if (typeof document === 'undefined') return;
    const el = document.getElementById(panelId);
    if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        // Pulse highlight
        el.classList.add('tour-highlight');
        setTimeout(() => el.classList.remove('tour-highlight'), 1500);
    }
}

function open() {
    tourStep.set(0);
    tourVisible.set(true);
    scrollToPanel(TOUR_STEPS[0].panel);
}

export { open };
</script>

{#if $tourVisible}
    <div class="tour-backdrop" on:click|self={close} role="dialog" aria-modal="true" aria-label="Guided tour">
        <div class="tour-card">
            <div class="tour-header">
                <span class="tour-counter">{$tourStep + 1} / {TOUR_STEPS.length}</span>
                <button class="tour-close" on:click={close} aria-label="Close tour">&#10005;</button>
            </div>

            <div class="tour-step-dots">
                {#each TOUR_STEPS as _, i}
                    <button
                        class="tour-dot"
                        class:active={i === $tourStep}
                        on:click={() => { tourStep.set(i); scrollToPanel(TOUR_STEPS[i].panel); }}
                        aria-label="Step {i + 1}"
                    />
                {/each}
            </div>

            <h2 class="tour-title">{TOUR_STEPS[$tourStep].title}</h2>
            <p class="tour-body">{TOUR_STEPS[$tourStep].body}</p>

            <div class="tour-nav">
                <button class="tour-btn secondary" on:click={prev} disabled={$tourStep === 0}>
                    &larr; Prev
                </button>
                <span class="tour-panel-hint">See: #{TOUR_STEPS[$tourStep].panel.replace('panel-', '')}</span>
                {#if $tourStep < TOUR_STEPS.length - 1}
                    <button class="tour-btn primary" on:click={next}>Next &rarr;</button>
                {:else}
                    <button class="tour-btn primary" on:click={close}>Done</button>
                {/if}
            </div>
        </div>
    </div>
{/if}

<!-- Tour launcher button (always visible) -->
<button class="tour-launcher" on:click={open} title="Open guided tour">
    ? Tour
</button>

<style>
.tour-backdrop {
    position: fixed;
    inset: 0;
    background: #0008;
    z-index: 200;
    display: flex;
    align-items: flex-end;
    justify-content: center;
    padding-bottom: 2rem;
}

.tour-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    max-width: 480px;
    width: calc(100% - 2rem);
    box-shadow: 0 8px 40px #0009;
}

.tour-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.75rem;
}
.tour-counter {
    font-size: 0.78rem;
    color: var(--text-dim);
    font-family: var(--font-mono);
}
.tour-close {
    background: none;
    border: none;
    color: var(--text-dim);
    font-size: 1rem;
    padding: 2px 6px;
    border-radius: 4px;
    transition: background var(--transition);
}
.tour-close:hover { background: var(--bg-panel); }

.tour-step-dots {
    display: flex;
    gap: 5px;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
}
.tour-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--border);
    border: none;
    padding: 0;
    cursor: pointer;
    transition: background var(--transition);
}
.tour-dot.active { background: var(--accent-teal); }

.tour-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 0.5rem;
}
.tour-body {
    font-size: 0.875rem;
    color: var(--text-dim);
    line-height: 1.6;
    margin-bottom: 1rem;
}

.tour-nav {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
}
.tour-btn {
    border-radius: 6px;
    padding: 6px 18px;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid var(--border);
    transition: all var(--transition);
}
.tour-btn.primary {
    background: var(--accent-teal);
    color: #000;
    border-color: var(--accent-teal);
}
.tour-btn.secondary {
    background: var(--bg-panel);
    color: var(--text);
}
.tour-btn:disabled { opacity: 0.35; cursor: not-allowed; }
.tour-panel-hint {
    font-size: 0.72rem;
    color: var(--text-dim);
    font-family: var(--font-mono);
}

/* Tour launcher */
.tour-launcher {
    position: fixed;
    bottom: 1.5rem;
    right: 1.5rem;
    background: var(--accent-teal);
    color: #000;
    border: none;
    border-radius: 20px;
    padding: 8px 18px;
    font-size: 0.85rem;
    font-weight: 700;
    box-shadow: 0 4px 16px #0005;
    z-index: 150;
    transition: transform var(--transition);
}
.tour-launcher:hover { transform: scale(1.06); }

/* Panel highlight animation (added/removed programmatically) */
:global(.tour-highlight) {
    outline: 2px solid #2dd4bf !important;
    outline-offset: 4px;
    animation: tour-pulse 1.4s ease-out forwards;
}
@keyframes tour-pulse {
    0%   { outline-color: #2dd4bf; }
    60%  { outline-color: #2dd4bf88; }
    100% { outline-color: transparent; }
}
</style>
