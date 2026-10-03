Yes — your intuition is basically right, with one important correction:

 > **SSD in Mamba-2 is not “a prefill operation.” It is an alternative way of computing the SSM that is especially useful during training/prefill, because it converts much of the sequential recurrence into matrix multiplications. During autoregressive decode, Mamba-2 can fall back to the recurrent SSM form.**

 The key idea is that **the same computation can be viewed either as an SSM recurrence or as a structured, attention-like matrix multiplication**.  Tri Dao+1

 Let me unpack that from the ground up.

 ## 1\. Start with ordinary Mamba's SSM

 At the heart of Mamba you have something conceptually like

 $$
h_t = A_t h_{t-1} + B_t x_t
$$

 and

 $$
y_t = C_t^T h_t.
$$

 So token $t$ updates a recurrent state, and that state produces the output.

 This is naturally **sequential**:

```
x0 -> h0 -> y0
       |
x1 ---> h1 -> y1
       |
x2 ---> h2 -> y2
       |
x3 ---> h3 -> y3
```

 That's excellent for autoregressive inference because when token $t+1$ arrives, you only need the previous state.

 But it's awkward for **prefill/training**, where you might have 4K, 16K, or 64K tokens available simultaneously. A GPU would much rather do large matrix multiplications than execute a long chain of dependent operations.

 This was one of the important motivations for Mamba-2/SSD. The original Mamba selective scan did not make particularly good use of GPU tensor cores; SSD was designed to exploit matrix multiplication hardware. Tri Dao explicitly describes this as a central motivation for SSD.  Tri Dao

---

 # 2\. The surprising thing: the recurrence is a matrix multiplication

 Let's temporarily simplify the state to a scalar.

 Suppose

 $$
h_t = a_t h_{t-1} + b_t x_t.
$$

 Expand the recurrence:

 $$
h_0 = b_0x_0
$$

 $$
h_1 = a_1b_0x_0 + b_1x_1
$$

 $$
h_2 = a_2a_1b_0x_0 + a_2b_1x_1+b_2x_2
$$

 etc.

 So the entire sequence can be represented as

 $$
\begin{bmatrix}
h_0\\
h_1\\
h_2\\
h_3
\end{bmatrix}
=
\underbrace{
\begin{bmatrix}
1 & 0 & 0 & 0\\
a_1 & 1 & 0 & 0\\
a_2a_1 & a_2 & 1 & 0\\
a_3a_2a_1 & a_3a_2 & a_3 & 1
\end{bmatrix}}_{L}
\begin{bmatrix}
b_0x_0\\
b_1x_1\\
b_2x_2\\
b_3x_3
\end{bmatrix}.
$$

 That matrix $L$ is called **1-semiseparable**.

 And this is the first really important insight:

 **The SSM recurrence is equivalent to multiplying by a particular structured lower-triangular matrix.**  Tri Dao

 So there are now two ways to compute exactly the same thing:

```
                 same computation
                       |
             +---------+---------+
             |                   |
        recurrent SSM       matrix form
             |                   |
        sequential          matmuls / GEMMs
```

 That's the "duality" in **State Space Duality**.

---

 # 3\. Where the "attention-like" part comes from

 Mamba-2 imposes additional structure on the SSM.

 One particularly important change from Mamba-1 is that instead of each individual channel having its own independent scalar-ish dynamics, Mamba-2 organizes things into **heads**.

 For one head, conceptually:

 $$
h_t = a_t h_{t-1} + B_t x_t
$$

 where:

 - $h_t$ has $N$ state dimensions,
- $x_t$ has $P$ head dimensions,
- $B_t$ maps $P\rightarrow N$,
- $C_t$ maps $N\rightarrow P$,
- and crucially $A_t$ is a **scalar times identity**.

 That last point is important.

 Instead of

 $$
A_t =
\begin{bmatrix}
a_{t,1}& &\\
& a_{t,2}&\\
&&a_{t,3}
\end{bmatrix},
$$

 Mamba-2 essentially has

 $$
A_t = a_t I.
$$

 This restriction is what allows the state-space recurrence to factor into something resembling attention.  Tri Dao

---

 # 4\. Expand the recurrence

 Consider the contribution of token $j$ to output $i$.

 Because the state decays through the intermediate $A$'s, you get something of the form

 $$
y_i =
\sum_{j\le i}
C_i^T
\left(
\prod_{k=j+1}^{i}a_k
\right)
B_jx_j.
$$

 Because $a_k$ is a **scalar**, you can move the scalar product outside:

 $$
y_i =
\sum_{j\le i}
\underbrace{
\left(\prod_{k=j+1}^{i}a_k\right)
}_{\text{SSM decay}}
\underbrace{C_i^TB_j}_{\text{attention-like similarity}}
x_j.
$$

 This is extremely revealing.

 Compare it with attention:

 $$
y_i = \sum_j
\underbrace{q_i^Tk_j}_{\text{similarity}}
v_j.
$$

 SSD has essentially:

 $$
\boxed{
\text{decay}_{ij}
\times
\text{similarity}_{ij}
\times
x_j
}
$$

 with a **structured causal decay matrix** instead of the usual softmax attention matrix.

 So Mamba-2's SSD can be interpreted as a particular form of **structured masked attention**.  Tri Dao

---

 # 5\. Now we get to your matrix-multiplication intuition

 Suppose the sequence has length $T$.

 The completely expanded representation is something like

 $$
Y = M(X)X
$$

 where $M$ is a structured $T\times T$ token-mixing matrix.

 For normal attention:

 $$
M = \operatorname{softmax}(QK^T).
$$

 For SSD, $M$ has a special semiseparable structure.

 Conceptually:

```
          token 0   token 1   token 2   token 3
token 0     X
token 1     X         X
token 2     X         X         X
token 3     X         X         X         X
```

 It's causal and lower triangular.

 But unlike arbitrary attention, the values in this matrix have a very special structure:

 $$
M_{ij}
=
\left(\prod_{k=j+1}^{i}a_k\right)
C_i^TB_j.
$$

 That structure is what lets you avoid constructing the entire $T\times T$ matrix.

---

 # 6\. This is where **chunking** enters

 This is probably the part you're thinking of when you say "a set of matrix multiplications."

 Instead of processing the entire sequence as one giant matrix, SSD divides it into chunks.

 For example:

```
sequence:

|----------------|----------------|----------------|----------------|
     chunk 0          chunk 1          chunk 2          chunk 3
```

 Suppose the chunk size is $Q=64$.

 The SSD algorithm then breaks the computation into four conceptual operations. Tri Dao describes exactly this chunkwise interpretation.  Tri Dao

 ### Step 1 — intra-chunk computation

 For each chunk, calculate how tokens within that chunk affect each other.

 For example:

```
chunk 0:

x0 ---> y0
x1 ---> y1
x2 ---> y2
...
x63 --> y63
```

 But instead of running the recurrence 64 times, SSD forms the appropriate structured matrix and performs the computation using tensor contractions / matrix multiplications.

 This is the **attention-like quadratic part**.

 So within a chunk you effectively get something conceptually like

 $$
Y_{\text{chunk}}
=
C
\;L\;
B
\;X.
$$

 The exact tensor dimensions are more complicated because of heads and state dimensions, but that's the essential structure.

 And **this is where GPU tensor cores become useful**.

---

 # 7\. Step 2 — calculate the chunk's outgoing state

 After processing a chunk, you need to know:

 > "If I started this chunk with zero state, what state would this chunk produce?"

 Call that

 $$
s_c.
$$

 Conceptually:

```
chunk 0 ---> state s0
chunk 1 ---> state s1
chunk 2 ---> state s2
chunk 3 ---> state s3
```

 These states can again be calculated in parallel using matrix multiplications.

---

 # 8\. Step 3 — pass the states between chunks

 Now comes the part that **cannot simply be turned into independent GEMMs**.

 Suppose:

 $$
h_{\text{end},0}=s_0
$$

 and

 $$
h_{\text{end},1}
=
A_1 h_{\text{end},0}+s_1.
$$

 Then:

 $$
h_{\text{end},2}
=
A_2h_{\text{end},1}+s_2.
$$

 So we have a recurrence **between chunks**.

 But notice the crucial difference:

```
original sequence:

token 0 -> token 1 -> token 2 -> ... -> token 4095
                 4096 sequential dependencies

chunked:

chunk 0 -> chunk 1 -> chunk 2 -> ... -> chunk 63
               only 64-ish dependencies
```

 If sequence length is 4096 and chunk size is 64, you've reduced the sequential problem from 4096 steps to only 64 chunk states.

 And even this chunk-level recurrence can use a scan or other structured-matrix technique. The production Triton implementation uses an efficient state-passing implementation rather than literally materializing the whole matrix.  Tri Dao+1

---

 # 9\. Step 4 — inject the incoming state into each chunk

 Now each chunk knows its **real initial state**.

 For example:

```
              incoming state
                   ↓
chunk 0: xxxxxxx [s0] ---> outputs

                   ↓
chunk 1: xxxxxxx [s1] ---> outputs

                   ↓
chunk 2: xxxxxxx [s2] ---> outputs
```

 The contribution of the incoming state to all tokens in the chunk can again be computed with matrix multiplications.

 So the final result is:

 $$
Y =
\underbrace{Y_{\text{intra}}}_{\text{within chunk}}
+
\underbrace{Y_{\text{state}}}_{\text{previous chunks}}.
$$

 That's essentially the SSD algorithm.

---

 # 10\. So the important picture is this

 I would mentally model Mamba-2 like this:

```
                    MAMBA-2 / SSD
                         |
             +-----------+-----------+
             |                       |
       SSM interpretation      attention-like
             |                  interpretation
             |                       |
       recurrence               structured
             |                  matrix multiply
             |                       |
             +-----------+-----------+
                         |
                    SSD algorithm
                         |
             +-----------+-----------+
             |           |           |
          intra-      chunk       output
          chunk       states      from state
          matmul      passing      matmul
             |           |           |
             +-----------+-----------+
                         |
                    final output
```

 And the really clever part is:

 > **SSD isn't replacing the SSM with attention. It is finding a faster way to evaluate the exact SSM computation by exploiting the fact that its sequence-mixing matrix has special structure.**

 That's why the word **duality** is important.  Tri Dao

---

 # 11\. Why this matters specifically for prefill

 This is where your intuition is particularly correct.

 Consider a prompt of 8,192 tokens.

 During prefill, you already have:

```
x0 x1 x2 x3 ... x8191
```

 all at once.

 A sequential implementation would theoretically have to do:

```
x0 -> state0
      ↓
x1 -> state1
      ↓
x2 -> state2
      ↓
...
```

 That's terrible utilization of the GPU.

 SSD instead says:

```
8192 tokens
     |
     +---- chunk 0 ---- GEMM
     +---- chunk 1 ---- GEMM
     +---- chunk 2 ---- GEMM
     +---- ...
     +---- chunk 127 --- GEMM
              |
          small state
            passing
              |
        combine results
```

 The expensive bulk of the work is therefore exposed as **large parallel tensor operations**.

 This is precisely why the Mamba-2 implementation has a `chunk_size` and a fused SSD path. The current reference implementation defaults to a chunk size of 256.  GitHub

---

 # 12\. But SSD isn't _only_ for prefill

 This distinction is important.

 There are really two operating modes:

 ### Prefill / training

 You have many tokens simultaneously.

 SSD can exploit:

 - chunk-level parallelism
- matrix multiplication
- tensor cores
- parallel intra-chunk computation

 So you get something much more GPU-friendly.

 ### Decode

 You have:

```
token T
   ↓
state
   ↓
token T+1
   ↓
state
   ↓
token T+2
```

 There is only **one new token at a time**.

 In that situation, doing a 64×64 matrix multiplication just to process one token would be silly.

 So you maintain the recurrent state and perform the SSM update directly.

 This is why the Mamba-2 authors describe the SSM mode as particularly appropriate for autoregressive inference, while the dual/matrix formulation is useful for training.  Tri Dao

---

 # 13\. There's an important subtlety about "quadratic"

 You might look at the intra-chunk computation and say:

 > "Wait, isn't that attention? Doesn't that make Mamba-2 quadratic?"

 **Locally, yes. Globally, no.**

 If the chunk size is $Q$, then each chunk has $Q^2$ work.

 There are $T/Q$ chunks.

 So:

 $$
\frac{T}{Q}Q^2 = TQ.
$$

 Therefore the total work is approximately

 $$
O(TQ)
$$

 rather than

 $$
O(T^2).
$$

 With a fixed chunk size $Q$, that's effectively linear in sequence length.

 This is one of the central tricks.

```
Full attention:

T × T
████████████████████
████████████████████
████████████████████
████████████████████

SSD:

Q × Q   Q × Q   Q × Q   Q × Q
████    ████    ████    ████

+ cheap/short-range state passing
```

 The production implementation uses this chunk decomposition rather than materializing a $T\times T$ attention matrix.  Tri Dao+1

---

 # 14\. Why matrix multiplication is so valuable on GPUs

 This is arguably the biggest practical reason SSD exists.

 Modern GPUs have specialized tensor cores that are **extremely good at GEMM**:

 $$
C=AB.
$$

 But they're much less efficient at arbitrary scalar operations and long chains of dependencies.

 Tri Dao gives the example that on an A100, BF16 matrix multiplication throughput can be dramatically higher than general FP32 arithmetic throughput.  Tri Dao

 So you can think of SSD as taking:

 > "a computation that mathematically looks like a recurrence"

 and restructuring it into:

 > "a computation that looks like GEMMs plus a much smaller recurrence."

 That's a very hardware-conscious algorithmic transformation.

---

 # 15\. One more distinction: SSD vs the SSD algorithm

 This terminology causes a lot of confusion.

 There are actually three related things:

 ### **SSD framework**

 The theoretical framework.

 It establishes relationships between:

 - SSMs
- structured matrices
- structured attention
- different ways of evaluating sequence transformations.

 ### **SSD model**

 The particular structured SSM used as the core of Mamba-2.

 ### **SSD algorithm**

 The chunked algorithm used to efficiently compute that model.

 Tri Dao explicitly distinguishes these three meanings.  Tri Dao

 So when people say:

 > "Mamba-2 uses SSD"

 they can mean the model.

 When someone says:

 > "SSD uses matrix multiplications"

 they're often talking about the **SSD algorithm**.

---

 ## The shortest possible mental model

 I'd summarize the whole thing as:

 $$
\boxed{
\text{SSM recurrence}
\quad\equiv\quad
\text{structured causal matrix multiplication}
}
$$

 Then:

 $$
\boxed{
\text{SSD}
=
\text{split sequence into chunks}
+
\text{GEMMs inside chunks}
+
\text{state passing between chunks}
}
$$

 And therefore:

```
                 Mamba-2

             SSM recurrence
                   |
          mathematically equivalent
                   ↓
       structured sequence matrix
                   |
             block decomposition
                   |
       ┌───────────┴───────────┐
       ↓                       ↓
  intra-chunk               inter-chunk
   GEMMs                    recurrence
       ↓                       ↓
       └───────────┬───────────┘
                   ↓
                output
```

 So **yes: your "prefill + a set of matrix multiplications" intuition is very close**. The refinement is that SSD is the _mathematical formulation_, while the chunked SSD algorithm is the implementation strategy that turns most of the prefill computation into highly parallel matrix multiplications. During decode, you generally don't need that machinery—you exploit the recurrent form instead.  Tri Dao+1

 If you want to go one level deeper, the really interesting next step is to take the actual Mamba-2 `ssd_minimal.py` implementation and **derive every one of its four `einsum`s from the recurrence**, including exactly why `segsum(A)` creates the equivalent of the causal attention matrix. That makes the implementation much less mysterious.  GitHub
