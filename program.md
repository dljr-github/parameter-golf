# Parameter Golf Autonomous Research

## Context
You are an autonomous researcher working on the OpenAI Parameter Golf challenge.
Goal: Train the best language model that fits in 16MB, scored by bits per byte (BPB) — lower is better.

## Required Reading
Before starting ANY experiments, read these files thoroughly:
1. `research-report.md` — detailed analysis of the top leaderboard entries, key techniques, and recommended approaches
2. `records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/README.md` — the #1 entry details
3. `records/track_10min_16mb/2026-03-24_74M_Ternary_UNet_FP8_10L_8192BPE_YaRN_NeoMuon/README.md` — the ternary BitNet entry
4. `train_gpt.py` — the full training code you will modify
5. `README.md` — contains the full leaderboard with all submissions, scores, and links to each entry
6. Browse `records/track_10min_16mb/` — each submission has its own README.md, train_gpt.py, and training logs

Use insights from these files to guide your experiments. Do not experiment blindly.


## Setup
- **Repo**: `~/repo/parameter-golf/`
- **Conda env**: `parameter-golf` (activate with `source ~/libraries/anaconda3/etc/profile.d/conda.sh && conda activate parameter-golf`)
- **GPU**: NVIDIA RTX 5090 (32GB VRAM)
- **Dataset**: Already downloaded at `./data/datasets/fineweb10B_sp1024/`
- **Tokenizer**: `./data/tokenizers/fineweb_1024_bpe.model`
- **nvidia-smi**: Located at `/usr/lib/wsl/lib/nvidia-smi` (WSL2 environment)

## The File You Modify
`train_gpt.py` — the single training script. Everything is fair game: architecture, optimizer, hyperparameters, quantization, training loop.

Do NOT modify files in `data/` or `records/`.

## Running an Experiment

```bash
source ~/libraries/anaconda3/etc/profile.d/conda.sh
conda activate parameter-golf
cd ~/repo/parameter-golf
export PATH="/usr/lib/wsl/lib:$PATH"

RUN_ID=exp_<name> \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=50 \
python3 train_gpt.py > run.log 2>&1
```

Time budget: 10 minutes per experiment. Use `MAX_WALLCLOCK_SECONDS=600`.

## Reading Results
```bash
grep "final_int8_zlib_roundtrip_exact" run.log
grep "peak memory" run.log
```

The key metric is `val_bpb` from the `final_int8_zlib_roundtrip_exact` line. This is the post-quantization BPB which is what the competition scores.

## Current Baseline (10-min run on 5090)
- **BPB: 2.1915** (post-quant) at 193 steps
- Config: 9 layers, 512 dim, 8 heads, 4 KV heads, 2x MLP, vocab 1024
- Step avg: ~623ms, VRAM: 10.9GB, Model size: 7.5MB

## What the Top Competition Entries Use (for reference)
Read `research-report.md` in this repo for full analysis. Key techniques from the leaderboard:
1. **11 layers, 3x MLP** — but this is slower per step. On 5090, stick with 9L/2x for 10-min runs
2. **Muon optimizer** — already in the baseline code
3. **EMA (Exponential Moving Average)** — maintain shadow weights, use for eval. NOT in baseline.
4. **GPTQ-lite** — better post-training quantization. NOT in baseline.
5. **LeakyReLU(0.5)²** — replace `torch.relu` with `F.leaky_relu(..., 0.5)` in MLP. Helps at convergence.
6. **Warmdown schedule** — specific LR decay. Already partially in baseline.
7. **Sliding window eval** — evaluate with overlapping windows. NOT in baseline.
8. **Test-Time Training (TTT)** — adapt model to val data during eval. Advanced technique.
9. **XSA (Cross-Segment Attention)** — custom attention on last few layers.
10. **SmearGate** — L2 normalize Q/K, learned temperature. Stabilizes deep models.

## Important Constraints
- **16MB artifact limit**: Code + compressed model must be under 16,000,000 bytes
- **No FlashAttention-3**: The 5090 uses PyTorch SDPA, not FA3. Don't try to import flash_attn.
- **Single GPU**: No distributed training, no torchrun needed for local experiments.
- **VRAM budget**: 32GB available, baseline uses ~11GB. Room to grow.

## Environment Variables (key knobs)
| Var | Default | Description |
|-----|---------|-------------|
| NUM_LAYERS | 9 | Transformer depth |
| MODEL_DIM | 512 | Hidden dimension |
| NUM_HEADS | 8 | Attention heads |
| NUM_KV_HEADS | 4 | KV heads (GQA) |
| MLP_MULT | 2 | MLP width multiplier |
| VOCAB_SIZE | 1024 | Vocabulary size |
| TRAIN_BATCH_TOKENS | 524288 | Batch size in tokens |
| TRAIN_SEQ_LEN | 1024 | Sequence length |
| WARMDOWN_ITERS | 1200 | LR warmdown iterations |
| WARMUP_STEPS | 20 | LR warmup steps |
| MUON_MOMENTUM | 0.95 | Muon optimizer momentum |
| MATRIX_LR | 0.04 | Learning rate for weight matrices |
| EMBED_LR | 0.6 | Embedding learning rate |

## Logging Results
Maintain `results.tsv` (tab-separated):
```
commit	val_bpb	memory_gb	status	description
```

## Experiment Loop
LOOP FOREVER:
1. Look at current state and results so far
2. Modify `train_gpt.py` with an experimental idea
3. `git commit -am "exp: <description>"`
4. Run experiment (redirect output to run.log)
5. Read results from run.log
6. Log to results.tsv
7. If BPB improved → keep the commit
8. If BPB worse → `git reset --hard HEAD~1`
9. Repeat

**NEVER STOP.** The human may be sleeping. Run experiments indefinitely until manually interrupted. If stuck, try combining previous wins, try radical changes, re-read research-report.md for ideas.


## Independent Research
You have internet access. If you need to understand a technique better:
- Search GitHub for implementations (e.g. Muon optimizer, EMA, GPTQ)
- Read the actual PRs on https://github.com/openai/parameter-golf/pulls — they contain discussion and ablation results
- Look at other submissions' train_gpt.py files in records/ for implementation patterns
- If a technique from the research report is unclear, look up the paper or reference implementation

Don't just guess at implementations — find working code and adapt it.



## CRITICAL: Architecture First, Training Recipe Second

DO NOT spend more time on training recipe tuning (LR, batch size, warmdown, etc). Those are diminishing returns at this point.

FOCUS ON ARCHITECTURAL CHANGES. The top entries achieve 1.067 BPB through novel architectures, not hyperparameter tuning. Once we find the best architecture, THEN we optimize training.

### Architecture Priority (implement these NOW):

1. **Value Residual (PR #413)** — -0.015 BPB for 18 scalars
   - Save layer-0's V projection output
   - At each subsequent layer, blend: V_final = (1-alpha) * V_current + alpha * V_layer0
   - alpha is a learned scalar per layer (init ~0.1)
   
2. **U-Net Skip Connections** — used by top entries and BitNet
   - Add learned skip gates between early and late layers (e.g., layer 0→10, 1→9, 2→8)
   - Gate: sigmoid(learned_scalar) * skip_input + (1 - sigmoid(learned_scalar)) * current
   - Init gates at 0 (sigmoid(0) = 0.5) or small negative for safe residual start

3. **Gated Attention (PR #413)** — -0.003 BPB
   - Add per-head sigmoid gate on attention output
   - gate = sigmoid(learned_param_per_head)
   - attn_out = gate * attn_out

4. **XSA (Cross-Segment Attention)** — on last 4 layers
   - Look at records/track_10min_16mb/2026-03-23_LeakyReLU_LegalTTT_ParallelMuon/train_gpt.py for implementation

5. **SmearGate** — L2 normalize Q and K, learned per-head temperature
   - Q = Q / Q.norm(dim=-1, keepdim=True)
   - K = K / K.norm(dim=-1, keepdim=True)  
   - attn_weights = (Q @ K.T) * learned_temperature

6. **BigramHash increase** — try 8192-10240 (check if baseline has BigramHash and increase it)

7. **Increase to 11-12 layers** — -0.023 BPB for 12L vs 11L. Worth it if step time stays under control.

8. **Larger vocab (8192 BPE)** — download with: python3 data/cached_challenge_fineweb.py --variant sp8192 --train-shards 1

### Implementation approach:
- Read the actual train_gpt.py files in records/ to see how top entries implement these
- Add ONE architectural change at a time, measure, keep or revert
- Once all beneficial arch changes are stacked, THEN go back to training recipe optimization

## OpenAI Wishlist (for future moonshot track, NOT current priority)
These are techniques OpenAI explicitly wants to see explored. Save for after competitive submission:
- JEPA (Joint Embedding Predictive Architecture)
- Text diffusion
- H-net tokenization
- Universal Transformer (4-hour non-record run — our 5090 can do this)
- Megakernels
- E2E TTT / super long context eval
- Learning adapters on random linear maps

Reference implementations for 1-bit and ternary quantization are already in records/ (Ciprian submissions).


## Submission Requirements
When we're ready to submit, the PR must add a new folder to /records/ containing:
1. **README.md** — explains the submission in detail
2. **submission.json** — name, GitHub ID, val_bpb, metadata (see existing examples)
3. **Train logs** — must demonstrate statistical significance (typically 3-seed average)
4. **train_gpt.py** — must compile and run successfully within the records folder

### Record submissions:
- Must beat existing SOTA by at least 0.005 nats
- Must show p < 0.01 significance (provide enough run logs)
- Must run in under 10 minutes on 8xH100s
- Tokenizer changes will be scrutinized — prove val_bpb is correctly calculated

### Non-record submissions:
- Must satisfy 16MB artifact limit
- Unique/creative approaches welcome even if they don't beat SOTA
- Unlimited compute track available (no 10-min cutoff, note in README)
- Interesting negative results are also accepted
- Justify ideas and results in detail

### Our submission plan:
- Develop locally on 5090 (10-min runs for fast iteration)
- When architecture is finalized, do full 10-min run on 5090 to sanity check
- Use H100 compute credits for official 3-seed benchmark runs
- Submit PR with all required files

## STOP HYPERPARAMETER TUNING
We are NOT on the final hardware (5090 vs 8xH100). LR, batch size, warmdown values will NOT transfer. 
STOP tuning these. Only architectural changes transfer across hardware:
- U-Net skip connections
- XSA (cross-segment attention)  
- SmearGate
- BigramHash size increase
- Larger vocab (8192 BPE)
- Catalytic Residuals

Implement these. Read the top entries train_gpt.py files in records/ for reference code.
Start with U-Net skip connections — used by BOTH the #1 entry and the BitNet entries.
