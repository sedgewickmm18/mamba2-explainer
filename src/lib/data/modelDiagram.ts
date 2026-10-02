/**
 * modelDiagram.ts
 * Mermaid flowchart definition for the Mamba2 architecture.
 *
 * Authored from the FalkorDB graph `arch:llama.cpp:mamba2-ssm`
 * (Op / Tensor / State / Weight / WriteBack nodes). The topology mirrors the
 * graph one-to-one:
 *   - input -> RMSNorm -> W_in projection -> split z/xBC/dt
 *   - dt path adds dt_bias before the scan
 *   - xBC path concatenates the r_l conv history -> depthwise conv1d -> SiLU
 *     -> split x/B/C
 *   - SSM scan reads s_l and the A decay, writes s_l back, emits y
 *   - skip y + D*x, SwiGLU gate with z, grouped RMS norm, W_out
 *   - residual add with the layer input, layer output
 *   - states r_l / s_l / A / D are side nodes with read edges,
 *     write-backs are dashed, weights are dotted side nodes
 *
 * Static embed by design: the app never talks to FalkorDB at runtime.
 */
import { writable } from 'svelte/store';

/** Modal open/close state, shared between the topbar link and the modal. */
export const modelDiagramOpen = writable(false);

export function openModelDiagram() {
    modelDiagramOpen.set(true);
}

export const MAMBA2_MERMAID = `
flowchart TD
    %% ---- main data path -------------------------------------------------
    XIN["x  input embedding"]:::tensor
    NORM["RMS Norm"]:::op
    XN["x normed"]:::tensor
    PROJ["Input projection"]:::op
    ZXBCDT["zxBCdt projected"]:::tensor
    SPLIT1["Split z / xBC / dt"]:::op

    Z["z gate"]:::tensor
    XBC["xBC conv input"]:::tensor
    DTRAW["dt raw"]:::tensor
    DTB["dt plus dt_bias"]:::op
    DT["dt with bias"]:::tensor

    CONCAT["Concat history + xBC"]:::op
    CONVX["conv_x  history+xBC"]:::tensor
    CONV["SSM Conv 1-D depthwise"]:::op
    SILU["SiLU activation"]:::op
    XBCACT["xBC after conv+SiLU"]:::tensor
    SPLIT2["Split x / B / C"]:::op

    XV["x values"]:::tensor
    BK["B keys"]:::tensor
    CQ["C queries"]:::tensor

    SCAN["SSM Scan recurrence"]:::op
    Y["y  SSM output"]:::tensor
    SKIP["Skip connection  y + D*x"]:::op
    YSKIP["y after skip"]:::tensor
    GATE["SwiGLU gate"]:::op
    YGATE["y after SwiGLU"]:::tensor
    GNORM["Grouped RMS Norm"]:::op
    YGN["y after group norm"]:::tensor
    OUTP["Output projection"]:::op
    RESADD["Residual add"]:::op
    LOUT["out  layer output"]:::tensor

    %% ---- state side nodes -------------------------------------------------
    RL["r_l  conv history"]:::state
    SL["s_l  SSM state S"]:::state
    ADEC["A log-decay"]:::state
    DSC["D skip scale"]:::state

    %% ---- write-backs ------------------------------------------------------
    WSL["Write S_new to s_l"]:::writeback
    WRL["Write conv history to r_l"]:::writeback

    %% ---- weights (dotted side nodes) --------------------------------------
    WIN["W_in"]:::weight
    WCONV["W_conv"]:::weight
    BCONV["b_conv"]:::weight
    DTBIAS["dt_bias"]:::weight
    WNORM["W_norm"]:::weight
    WOUT["W_out"]:::weight

    %% ---- edges -------------------------------------------------------------
    XIN -->|FLOWS_INTO| NORM
    XIN -->|residual| RESADD
    NORM -->|PRODUCES| XN
    XN -->|FLOWS_INTO| PROJ
    PROJ -->|PRODUCES| ZXBCDT
    ZXBCDT -->|FLOWS_INTO| SPLIT1
    SPLIT1 -->|PRODUCES| Z
    SPLIT1 -->|PRODUCES| XBC
    SPLIT1 -->|PRODUCES| DTRAW
    DTRAW -->|FLOWS_INTO| DTB
    DTB -->|PRODUCES| DT
    DT -->|FLOWS_INTO| SCAN

    XBC -->|FLOWS_INTO| CONCAT
    RL -.->|READ_BY| CONCAT
    CONCAT -->|PRODUCES| CONVX
    CONVX -->|FLOWS_INTO| CONV
    CONVX -->|FLOWS_INTO| WRL
    CONV -->|PRODUCES| SILU
    SILU -->|PRODUCES| XBCACT
    XBCACT -->|FLOWS_INTO| SPLIT2
    SPLIT2 -->|PRODUCES| XV
    SPLIT2 -->|PRODUCES| BK
    SPLIT2 -->|PRODUCES| CQ

    XV -->|FLOWS_INTO| SCAN
    XV -->|FLOWS_INTO| SKIP
    BK -->|FLOWS_INTO| SCAN
    CQ -->|FLOWS_INTO| SCAN
    SL -.->|READ_BY| SCAN
    ADEC -.->|READ_BY| SCAN

    SCAN -->|FLOWS_INTO| WSL
    SCAN -->|PRODUCES| Y
    WSL -.->|WRITES_TO| SL
    WRL -.->|WRITES_TO| RL

    Y -->|FLOWS_INTO| SKIP
    DSC -.->|READ_BY| SKIP
    SKIP -->|PRODUCES| YSKIP
    YSKIP -->|FLOWS_INTO| GATE
    Z -->|FLOWS_INTO| GATE
    GATE -->|PRODUCES| YGATE
    YGATE -->|FLOWS_INTO| GNORM
    GNORM -->|PRODUCES| YGN
    YGN -->|FLOWS_INTO| OUTP
    OUTP -->|FLOWS_INTO| RESADD
    RESADD -->|PRODUCES| LOUT

    %% ---- weight usage -------------------------------------------------------
    WIN -.->|USED_BY| PROJ
    WCONV -.->|USED_BY| CONV
    BCONV -.->|USED_BY| CONV
    DTBIAS -.->|USED_BY| DTB
    WNORM -.->|USED_BY| GNORM
    WOUT -.->|USED_BY| OUTP

    %% ---- styling -------------------------------------------------------------
    classDef op fill:#0e7490,stroke:#22d3ee,color:#e0f2fe,stroke-width:1.5px;
    classDef tensor fill:#1e293b,stroke:#64748b,color:#e2e8f0,stroke-width:1px;
    classDef state fill:#064e3b,stroke:#34d399,color:#d1fae5,stroke-width:1px;
    classDef weight fill:#1a1d27,stroke:#a78bfa,color:#c4b5fd,stroke-width:1px,stroke-dasharray:3 3;
    classDef writeback fill:#4c1d95,stroke:#a78bfa,color:#ede9fe,stroke-width:1px,stroke-dasharray:6 3;
`;
