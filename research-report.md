# Parameter Golf Research Report

## 1. Executive Summary

The current leaderboard is dominated by 11-layer Transformer models. Success comes from meticulous combination of architectural tweaks, advanced training recipes, and aggressive quantization — not single breakthroughs.

**Key Findings:**
- **Dominant Architecture:** 11-layer Transformers with 3x MLP ratio
- **Critical Techniques:** Muon optimizer (2x faster convergence than AdamW), Test-Time Training (TTT), GPTQ-lite + QAT, EMA, LeakyReLU²
- **Parameter Budgeting:** Small vocab (4k-8k) essential — large vocab wastes budget on embeddings
- **Unexplored Potential:** Mamba/RWKV (linear attention), byte-level models, BitNet 1.58-bit

**Recommended Primary Approach:** Hybrid strategy — proven Transformer baseline + Mamba experimental track

---

## 2. Top Submissions Deep Dive

| Rank | Score | Key Techniques | Analysis |
|------|-------|---------------|----------|
| 1 | 1.1194 | LeakyReLU² + Legal TTT + Parallel Muon | Lead comes from TTT (adapting model to validation data during inference) and Parallel Muon for superior convergence. Most sophisticated entry. |
| 2 | 1.1228 | 11L EMA + GPTQ-lite + warmdown | Strong conventional entry. EMA stabilizes, GPTQ-lite optimizes clipping per weight row, warmdown is specific LR decay. Robust baseline. |
| 3 | 1.1248 | 11L Partial RoPE + LN Scale + EMA + XSA4 | Partial RoPE on subset of layers saves params. XSA4 is custom efficient attention on last 4 layers. |
| 4 | 1.1271 | 11L XSA4 + EMA + Int6 MLP3x | Mixed-precision: 6-bit for MLP layers with 3x hidden ratio. |
| 5 | 1.1307 | 11L Efficient Partial XSA | Iteration on XSA — technique is powerful but has optimization room. |

---

## 3. Key Techniques Explained

### Muon Optimizer
Matrix-aware optimizer that orthogonalizes gradient updates for 2D weight matrices, preventing spectral collapse. Up to 2x faster convergence than AdamW. Used in hybrid setup (Muon for 2D weights, AdamW for 1D biases/embeddings). **Parallel Muon** is the distributed multi-GPU version.

### Test-Time Training (TTT)
Model performs gradient steps on the validation set at inference time, adapting weights to compress the context. Legal within challenge rules. The current leader's biggest edge.

### EMA vs SWA
- **EMA:** Maintains shadow weights updated with exponential decay every step. Online, continuous stabilization.
- **SWA:** Averages checkpoints late in training. More of a post-processing step.
- **Verdict:** EMA preferred for its continuous effect during short training windows.

### GPTQ-lite
Post-training quantization that searches for optimal clipping value per weight row (vs fixed clipping in standard GPTQ). Critical for maximizing precision in int6.

### XSA (Cross-Segment Attention)
Custom attention over different sequence chunks — improves long-range dependencies without quadratic cost. Applied selectively to last few layers only.

### SmearGate
L2-normalizes Query and Key vectors before dot product, uses learned per-head temperature scalar. Bounds attention scores, prevents logit explosion in small deep models.

### BigramHash
Tokenizer trick — likely hashes token bigrams to create secondary vocabulary/embedding, capturing common word pairs without expanding primary vocab.

---

## 4. Unexplored Directions Analysis

### BitNet 1.58-bit (Ternary Weights)
- **Pros:** ~10x memory savings vs FP16. Multiplication-free compute (additions only).
- **Cons:** Requires training from scratch. Performance for small models lower unless width compensates. Needs custom kernels.
- **Verdict:** High-risk, high-reward. Could fit ~84M params in 16MB.

### Mamba / RWKV / Linear Attention ⭐
- **Pros:** O(N) time, O(1) memory. Mamba 3B matches Transformer 7B in parameter efficiency.
- **Cons:** Weaker at specific associative recall tasks.
- **Verdict:** **Extremely promising.** Top contender for innovative approach. Nobody on leaderboard has tried it.

### Mixture of Experts (MoE)
- **Pros:** Decouples total params from active params. Could have 30M+ quantized params with only 2-4M active per token.
- **Cons:** Router overhead, memory bandwidth bottleneck.
- **Verdict:** Promising but complex to implement under constraints.

### Byte-level Models
- **Pros:** No OOV tokens. Tiny embedding (256 entries). All budget goes to model.
- **Cons:** Longer sequences. Solved by MambaByte or Byte Latent Transformer.
- **Verdict:** Very promising. Byte-level Mamba = two powerful concepts combined.

### Depth Recurrence / Universal Transformers
- **Pros:** Massive parameter savings from weight reuse.
- **Cons:** Notoriously hard to train. May be too complex for 10-minute limit.
- **Verdict:** Likely not viable due to training complexity.

### MatMul-free LMs
- **Pros:** Extreme efficiency, similar to BitNet.
- **Cons:** Too new, no established training practices.
- **Verdict:** Too experimental for this challenge.

---

## 5. Tokenizer Analysis

- **Optimal vocab size:** 4,000-8,000 BPE tokens for 16MB budget
- **Reasoning:** Large vocab (32K) wastes most budget on embeddings. Small vocab reserves 80-90% for transformer blocks.
- **Byte-level:** Competitive with 256-token vocab if using MambaByte/BLT architecture
- **Current entries:** Likely using small BPE with weight tying

---

## 6. Training Efficiency (10 min on 8xH100s)

- **Achievable throughput:** 8-12 billion tokens
- **Steps:** ~8,000-16,000 depending on batch size
- **Data curriculum:** CRITICAL at this scale. Sorting by length minimizes padding (up to 50% speedup). Easy-first curriculum improves stability.

---

## 7. Recommended Strategy (Ranked)

### Track 1: Advanced Transformer Baseline (Low Risk, Top-5 Target)
- 11-layer, ~768-dim Transformer
- Hybrid Muon + AdamW optimizer
- EMA, LeakyReLU², curriculum learning, warmdown schedule
- Two-stage quant: train → GPTQ-lite → short QAT phase
- **Expected:** Top-5 position

### Track 2: Mamba Architecture (Medium Risk, Potential #1)
- Stack of Mamba blocks sized for 16MB
- Small BPE (4k) or byte-level tokenizer (MambaByte)
- **Expected:** Could outperform Transformer due to superior parameter efficiency
- **Nobody has tried this yet** — first-mover advantage

### Track 3: BitNet 1.58-bit (High Risk, Moonshot)
- Wider Transformer with ternary weights from scratch
- Custom QAT from step 1
- **Expected:** Transformative if it works, but high failure risk

---

## 8. Implementation Priorities

### Week 1: Environment + Baseline
- Fork repo, set up local eval on 5090
- Download FineWeb validation set
- Reproduce mid-tier entry (#4 or #5)
- Train small BPE tokenizer on FineWeb

### Week 2: Core Optimization
- Integrate Muon optimizer (hybrid with AdamW)
- Implement EMA + data curriculum/sorting
- Profile convergence vs AdamW baseline
- **Parallel:** Start Mamba prototype

### Week 3: Quantization + TTT
- Implement GPTQ-lite with optimal clip search
- Add QAT loop at end of training
- Implement TTT for inference-time adaptation
- **Parallel:** Benchmark Mamba vs Transformer at same param count

### Week 4: Combine + Submit
- Stack best techniques
- Apply for OpenAI compute grant (H100 time)
- Zstd compression tuning
- Hyperparameter sweep on H100s
- Submit PR

---

## 9. Risk Assessment

| Approach | Risk | Mitigation |
|----------|------|------------|
| Transformer Baseline | Low — main risk is tuning complexity | Add techniques one at a time, benchmark each |
| Mamba | Medium — less mature, fewer community resources | Start from official implementation, focus on stable training first |
| BitNet 1.58-bit | High — requires significant R&D, convergence not guaranteed | Pursue as parallel track only, leverage existing research |


---

## 10. Updated Findings (Mar 25)

### Mamba/SSM: Tested, Not Competitive
PR #599 (Hymba) submitted a hybrid Attention + Mamba SSM architecture:
- Score: 1.1828 BPB (non-record) — beats naive baseline (1.2244) but far from leader (1.1194)
- 7-layer hybrid model, attention + Mamba in parallel per block
- ~85ms/step on 8xH100 (~7,000 steps in 10 min)
- Key finding: SSM makes layers more powerful, so fewer layers needed (7L vs 9-11L)
- Key finding: Training stability critical for quantization — needed LR=0.02 + aggressive warmdown
- Missing: No EMA, no GPTQ-lite, no TTT, no XSA — none of the winning techniques applied
- Verdict: Mamba alone is not competitive. Gap too large (0.06+ BPB). Deprioritized.

### Local Testing Results (RTX 5090, 2-min runs)
| Config | Steps | BPB (post-quant) | VRAM | Notes |
|--------|-------|-------------------|------|-------|
| Baseline 9L/512d/2xMLP | 193 | 2.1915 | 10.9GB | Reference |
| 11L/512d/3xMLP | 144 | 2.5078 | 14.7GB | Slower steps = fewer iterations = worse |
| LeakyReLU(0.5)^2 | 192 | 2.2226 | 10.9GB | No improvement at 192 steps |

### Key Local Testing Insights
1. Step efficiency > model size at short horizons: Bigger models are slower per step, get fewer iterations in fixed time, and perform worse despite more parameters
2. Activation changes don't help early training: LeakyReLU^2 helps at convergence (7000+ steps) but not at 200 steps
3. 5090 is ~7.5x slower per step than 8xH100: 623ms vs ~83ms. Good for relative comparisons, not absolute scores
4. VRAM headroom: Only using 11GB/32GB — room to grow when it helps

### Revised Strategy (Ranked by Confidence)
1. Optimized Transformer (HIGH confidence) — Stack proven techniques: EMA, GPTQ-lite, warmdown, sliding window eval, TTT. Reliably top-10.
2. BitNet convergence speed (MEDIUM confidence) — Binary model proved 1.1239 BPB possible. If training can be accelerated to fit 10-min budget, this wins.
3. Mamba/Hymba (LOW confidence) — Deprioritized. Gap too large, double integration work for uncertain payoff.


## 11. Latest Findings — Real SOTA (Mar 25 evening)

### The Leaderboard Has Moved Far Beyond What README Shows

The official README leaderboard shows 1.1194 as #1, but pending PRs show the real frontier is around **1.067 BPB**:

| Score | Technique | PR | Status |
|-------|-----------|-----|--------|
| 1.0672 | SwiGLU + XSA4 + U-Net + AdamW TTT | #462 | 3-seed validated |
| 1.0887 | TrigramHash + ValueResidual + GradQuant + Cosine TTT | #486 | 3-seed validated |
| 1.0891 | Value Residual + Gated Attention + AdamW TTT | #490 | 1-seed, pending |
| 1.0920 | GEPA arch + legal SGD TTT (4xA100, non-record) | #668 | non-record |
| 1.0944 | GEPA 25k steps + GPTQ-lite + TTT (4xA100) | - | non-record |

### New Techniques Not In Our Original Research

1. **GEPA (Gemini-driven Evolutionary Architecture Search)**: AI-designed architecture via Gemini. PR #462 used this to discover its winning config. Meta-level innovation.

2. **Value Residual / ResFormer (PR #413)**: Layer-0 V projection shortcut to deeper layers. Only 18 extra scalars. **-0.015 BPB**. Extremely cheap, high impact.

3. **Gated Attention (PR #413)**: Per-head sigmoid gate on attention output. **-0.003 BPB**. Minimal overhead.

4. **SwiGLU FFN**: Replaces relu-squared MLP. Used by PR #462 (1.0672). Star-ReLU variant. Likely better than LeakyReLU-squared.

5. **U-Net Skip Connections**: Learned skip gates between encoder/decoder layers. Used by both BitNet entries and PR #462. Key architectural innovation.

6. **Catalytic Residuals (PR #450)**: **-0.024 BPB**. Combined with BigramHash(10240) which gives -0.070 BPB vs BigramHash(2048).

7. **AdamW TTT >> SGD TTT**: Switching TTT optimizer from SGD to AdamW gives **5x more improvement** (0.053 vs 0.011 BPB). The single biggest TTT finding. Recipe: lr=0.0005, 10 epochs, all blocks unfrozen.

8. **Cosine TTT Scheduling**: Per-layer LR groups based on quantization error sensitivity. MLP output projections get 3x base LR (highest quant damage). 0.5x for input projections. Acts as a 3x multiplier on TTT gains.

9. **TrigramHash**: Extension of BigramHash to trigrams. Used in PR #486.

10. **12 Layers**: PR by joshuaswarren found 12L gives -0.023 BPB vs 11L. More layers are worth it IF step time doesn't blow up.

### The Current Meta Stack (as of Mar 25)
Architecture:
- SwiGLU FFN (Star-ReLU) with 3x+ width
- U-Net skip connections with learned gating
- Value Residual (layer-0 V shortcut)
- Gated Attention (per-head sigmoid)
- XSA on last 4+ layers (or all 11)
- BigramHash(8192+) or TrigramHash
- SmearGate
- Partial RoPE (16 dims)
- LN Scale (1/sqrt(layer_idx+1))

Training:
- Muon + AdamW hybrid optimizer
- EMA (decay ~0.9985)
- Aggressive warmdown
- GPTQ-lite quantization + QAT

Evaluation:
- Sliding window eval (stride 16)
- Legal score-first AdamW TTT (lr=0.0005, 10 epochs)
- Cosine TTT with per-layer LR groups

### Revised Priority for Our Agent
1. Value Residual (-0.015 BPB, trivial to implement)
2. SwiGLU FFN (replace relu-squared)
3. U-Net skip connections
4. Gated Attention (-0.003 BPB)
5. BigramHash increase (10240 vs current)
6. Sliding window eval
7. TTT with AdamW (complex but huge payoff)
