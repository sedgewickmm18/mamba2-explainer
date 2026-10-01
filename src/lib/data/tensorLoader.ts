/**
 * tensorLoader.ts
 * Fetch and parse manifest + npz tensor files.
 * Exposes a Svelte writable store: AppState.
 */

import { writable, derived, get } from 'svelte/store';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ManifestEntry {
    prompt_id: string;
    prompt_text: string;
    layer: number;
    step: number;
    token: string;
    file: string;
    shapes: Record<string, number[]>;
}

export interface Manifest {
    prompts: string[];
    entries: ManifestEntry[];
    _note?: string;
}

export interface ModelMeta {
    model_name: string;
    n_layers: number;
    d_model: number;
    vocab_size: number;
    state_size?: number;
    intermediate_size?: number;
    num_heads?: number;
    n_groups?: number;
    conv_kernel_size?: number;
    expand?: number;
    head_dim?: number;
}

export type TensorMap = Map<string, Float32Array>;

export interface AppState {
    manifest: Manifest | null;
    meta: ModelMeta | null;
    promptIndex: number;
    layerIndex: number;
    stepIndex: number;
    tokens: string[];
    tensors: TensorMap;
    loading: boolean;
    error: string | null;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

const INITIAL: AppState = {
    manifest: null,
    meta: null,
    promptIndex: 0,
    layerIndex: 0,
    stepIndex: 0,
    tokens: [],
    tensors: new Map(),
    loading: false,
    error: null,
};

export const appState = writable<AppState>(INITIAL);

// Derived helpers
export const currentPrompt = derived(appState, ($s) =>
    $s.manifest ? $s.manifest.prompts[$s.promptIndex] ?? '' : ''
);

export const maxStep = derived(appState, ($s) => {
    if (!$s.manifest) return 0;
    const pid = promptId($s.promptIndex);
    const entries = $s.manifest.entries.filter(
        (e) => e.prompt_id === pid && e.layer === $s.layerIndex
    );
    return Math.max(0, entries.length - 1);
});

export const maxLayer = derived(appState, ($s) => {
    if (!$s.meta) return 23;
    return $s.meta.n_layers - 1;
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function promptId(idx: number): string {
    return `p${String(idx).padStart(2, '0')}`;
}

function dataPath(file: string): string {
    return `./data/${file}`;
}

// ---------------------------------------------------------------------------
// npz parser (pure JS - no wasm needed for float32)
// ---------------------------------------------------------------------------

const NPY_MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // \x93NUMPY

function parseNpy(buf: ArrayBuffer): Float32Array {
    const bytes = new Uint8Array(buf);
    // Check magic
    for (let i = 0; i < NPY_MAGIC.length; i++) {
        if (bytes[i] !== NPY_MAGIC[i]) throw new Error('Not a .npy file');
    }
    // Header length at offset 8 (little-endian uint16)
    const headerLen = bytes[8] | (bytes[9] << 8);
    const headerStr = new TextDecoder().decode(bytes.slice(10, 10 + headerLen));
    // Extract shape
    const shapeMatch = headerStr.match(/'shape':\s*\(([^)]*)\)/);
    const shape = shapeMatch
        ? shapeMatch[1].split(',').map((s) => parseInt(s.trim())).filter((n) => !isNaN(n))
        : [];
    const n = shape.reduce((a, b) => a * b, 1);
    // Data starts after magic(6) + version(2) + headerLen field(2) + headerLen bytes
    const dataOffset = 10 + headerLen;
    // Check dtype - we only handle float32 here
    if (headerStr.includes("'f4'") || headerStr.includes("<f4") || headerStr.includes("|f4")) {
        return new Float32Array(buf, dataOffset, n);
    }
    // Fallback: copy as float32 anyway (may be wrong for other dtypes)
    const raw = new Float32Array(buf.slice(dataOffset), 0, n);
    return raw;
}

async function parseNpz(url: string): Promise<TensorMap> {
    // npz = zip archive of .npy files
    // We use a minimal zip parser (no dependency needed for simple cases)
    const response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} fetching ${url}`);
    const buf = await response.arrayBuffer();
    return await unzipNpz(buf);
}

async function unzipNpz(buf: ArrayBuffer): Promise<TensorMap> {
    // np.savez_compressed writes ZIP64 headers: the local-header sizes are
    // 0xFFFFFFFF placeholders and the real sizes live in the extra field
    const result: TensorMap = new Map();
    const bytes = new Uint8Array(buf);
    const view = new DataView(buf);

    const u16 = (o: number) => view.getUint16(o, true);
    const u32 = (o: number) => view.getUint32(o, true);

    let offset = 0;
    while (offset + 4 < bytes.length) {
        // Local file header signature
        if (u32(offset) !== 0x04034b50) {
            break;
        }
        const compression = u16(offset + 8);
        let compressedSize = u32(offset + 18);
        let uncompressedSize = u32(offset + 22);
        const fnLen = u16(offset + 26);
        const extraLen = u16(offset + 28);

        // ZIP64: pull real sizes from the extended-info extra field (id 0x0001)
        if (compressedSize === 0xffffffff || uncompressedSize === 0xffffffff) {
            let e = offset + 30 + fnLen;
            const eEnd = e + extraLen;
            while (e + 4 <= eEnd) {
                const id = u16(e);
                const size = u16(e + 2);
                if (id === 0x0001) {
                    let p = e + 4;
                    if (uncompressedSize === 0xffffffff && p + 8 <= eEnd) {
                        uncompressedSize = Number(view.getBigUint64(p, true));
                        p += 8;
                    }
                    if (compressedSize === 0xffffffff && p + 8 <= eEnd) {
                        compressedSize = Number(view.getBigUint64(p, true));
                    }
                    break;
                }
                e += 4 + size;
            }
        }

        const nameBytes = bytes.slice(offset + 30, offset + 30 + fnLen);
        const name = new TextDecoder().decode(nameBytes).replace(/\.npy$/, '');

        const dataStart = offset + 30 + fnLen + extraLen;

        if (compression === 0) {
            // Stored (no compression)
            const npyBuf = buf.slice(dataStart, dataStart + uncompressedSize);
            result.set(name, parseNpy(npyBuf));
        } else if (compression === 8) {
            // Deflate
            const inflated = await inflateRaw(bytes.subarray(dataStart, dataStart + compressedSize));
            result.set(name, parseNpy(inflated));
        }
        // np.savez_compressed only produces stored/deflate members; skip anything else

        offset = dataStart + compressedSize;
    }
    if (result.size === 0) {
        throw new Error('npz contained no parsable tensors');
    }
    return result;
}

async function inflateRaw(compressed: Uint8Array): Promise<ArrayBuffer> {
    // slice() copies the view into its own ArrayBuffer; pipeThrough keeps the
    // readable side draining while decompression runs - writing and closing
    // without a concurrent reader can deadlock
    const stream = new Blob([compressed.slice()])
        .stream()
        .pipeThrough(new DecompressionStream('deflate-raw'));
    return await new Response(stream).arrayBuffer();
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

export async function initLoader(basePath = '') {
    appState.update((s) => ({ ...s, loading: true, error: null }));
    try {
        const [manifestRes, metaRes] = await Promise.all([
            fetch(`${basePath}/data/manifest.json`),
            fetch(`${basePath}/data/model_meta.json`).catch(() => null),
        ]);

        if (!manifestRes.ok) {
            throw new Error(`Could not load manifest: HTTP ${manifestRes.status}`);
        }

        const manifest: Manifest = await manifestRes.json();
        let meta: ModelMeta | null = null;
        if (metaRes && metaRes.ok) {
            meta = await metaRes.json();
        }

        // Extract token sequence for prompt 0, layer 0
        const tokens = extractTokens(manifest, 0, 0);

        appState.update((s) => ({
            ...s,
            manifest,
            meta,
            tokens,
            loading: false,
        }));

        // Pre-load tensors for first step
        await loadStep(0, 0, 0);
    } catch (err) {
        appState.update((s) => ({
            ...s,
            loading: false,
            error: String(err),
        }));
    }
}

function extractTokens(manifest: Manifest, promptIdx: number, layerIdx: number): string[] {
    const pid = promptId(promptIdx);
    return manifest.entries
        .filter((e) => e.prompt_id === pid && e.layer === layerIdx)
        .sort((a, b) => a.step - b.step)
        .map((e) => e.token);
}

async function loadStep(promptIdx: number, layerIdx: number, stepIdx: number) {
    const state = get(appState);
    if (!state.manifest) return;

    const pid = promptId(promptIdx);
    const entry = state.manifest.entries.find(
        (e) => e.prompt_id === pid && e.layer === layerIdx && e.step === stepIdx
    );

    if (!entry) {
        appState.update((s) => ({ ...s, tensors: new Map() }));
        return;
    }

    appState.update((s) => ({ ...s, loading: true }));
    try {
        const tensors = await parseNpz(dataPath(entry.file));
        appState.update((s) => ({
            ...s,
            tensors,
            loading: false,
            stepIndex: stepIdx,
        }));
    } catch (err) {
        appState.update((s) => ({
            ...s,
            loading: false,
            error: String(err),
        }));
    }
}

export async function selectPrompt(idx: number) {
    const state = get(appState);
    if (!state.manifest) return;
    const tokens = extractTokens(state.manifest, idx, state.layerIndex);
    appState.update((s) => ({ ...s, promptIndex: idx, stepIndex: 0, tokens }));
    await loadStep(idx, state.layerIndex, 0);
}

export async function selectLayer(idx: number) {
    const state = get(appState);
    appState.update((s) => ({ ...s, layerIndex: idx }));
    await loadStep(state.promptIndex, idx, state.stepIndex);
}

export async function stepForward() {
    const state = get(appState);
    const mStep = get(maxStep);
    const next = Math.min(state.stepIndex + 1, mStep);
    if (next === state.stepIndex) return;
    await loadStep(state.promptIndex, state.layerIndex, next);
}

export async function stepBack() {
    const state = get(appState);
    const prev = Math.max(state.stepIndex - 1, 0);
    if (prev === state.stepIndex) return;
    await loadStep(state.promptIndex, state.layerIndex, prev);
}

export async function goToStep(idx: number) {
    const state = get(appState);
    await loadStep(state.promptIndex, state.layerIndex, idx);
}
