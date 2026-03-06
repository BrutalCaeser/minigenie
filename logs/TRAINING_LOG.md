# Training Log

*Per-session records of Colab GPU training runs. Updated after every training session.*

---

## Session 1 — VQ-VAE Training (Phase 6)

**Date:** 2026-03-03  
**GPU:** Colab Pro (T4/A100)  
**Steps:** 50,000  

### Results
| Metric | Result | Target | Status |
|---|---|---|---|
| Reconstruction PSNR | 31.12 dB | ≥ 28 dB | ✅ PASS |
| Codebook utilization | 100% (512/512) | ≥ 80% | ✅ PASS |

### Notes
- VQ-VAE converged comfortably above target
- 100% codebook utilization indicates no dead codes — EMA updates + dead code reset working well
- Checkpoint saved to Drive: `checkpoints/vqvae/step_0050000.pt`

---

## Session 2 — Dynamics Training Session 1 (Phase 7)

**Date:** 2026-03-04  
**GPU:** Colab Pro  
**Steps:** 0 → 40,000 (of 150K planned)  
**Game:** CoinRun only (scoped down from 5 games to save Drive space)  

### Results
| Metric | Result | Target | Status |
|---|---|---|---|
| Single-step PSNR (training log, 320 samples) | 26.06 dB | ≥ 22 dB | ✅ PASS |

### Notes
- Model learning — single-step prediction already well above target
- Did not run full evaluation suite during training
- Checkpoint saved to Drive: `checkpoints/dynamics/step_0040000.pt`

---

## Session 2.5 — Evaluation at 40K (Phase 8)

**Date:** 2026-03-05  
**GPU:** Colab Pro  

### Full Evaluation Results (eval_results.json)
| Metric | Result |
|---|---|
| Single-step PSNR mean | 20.36 dB* |
| Single-step SSIM | 0.716 |
| Rollout step 1 | 26.09 dB |
| Rollout step 10 | 14.77 dB |
| Rollout step 25 | 12.98 dB |
| Rollout step 50 | 11.82 dB |
| Action L2 distance | 0.063 |

*\*Bug found: `evaluate_single_step` used sequential indices 0–999 (biased toward first ~5 episodes). Fixed to random sampling. The training log (26.06 dB) and rollout step-1 (26.09 dB) are the correct reference — they agree.*

### Diagnosis
- Single-step: strong (26 dB corrected), well above 22 dB target
- Rollout: steep degradation, model not robust to autoregressive feeding at 40K steps
- Action differentiation: very weak (L2 = 0.063), conditioning hasn't converged
- **Decision: train 40K more steps to 80K**

---

## Session 3 — Dynamics Training Session 2 (Phase 7 continued)

**Date:** 2026-03-06  
**GPU:** Colab Pro  
**Steps:** 40,000 → 80,000  
**Game:** CoinRun  

### Training Log Output (320 samples at step 80K)
```
Single-step PSNR over 320 predictions:
   Mean:   26.99 dB
   Median: 27.30 dB
   Min:    11.44 dB
   Max:    40.68 dB
   Target: 22.0 dB [PASS]
```

### Full Evaluation Results (eval_results.json)
| Metric | 40K | 80K | Δ |
|---|---|---|---|
| Single-step PSNR mean | ~26.06 | 26.75 | +0.69 |
| Single-step SSIM | 0.716 | 0.840 | +0.124 |
| Rollout step 1 | 26.09 | 27.00 | +0.91 |
| Rollout step 10 | 14.77 | 14.17 | −0.60 |
| Rollout step 25 | 12.98 | 12.04 | −0.94 |
| Rollout step 50 | 11.82 | 10.98 | −0.84 |
| Action L2 distance | 0.063 | 0.064 | ~flat |

### Honest Analysis

**What improved:**
- Single-step PSNR: +0.9 dB — model produces sharper, more accurate single-frame predictions
- SSIM jumped from 0.716 → 0.840 — structurally better predictions (textures, edges)
- Rollout step 1: +0.9 dB — consistent with single-step improvement

**What didn't improve (or got worse):**
- Rollout steps 10–50 all degraded by 0.6–0.9 dB — the model became *more* sensitive to distribution shift from autoregressive feeding, not less
- Action differentiation flat at 0.064 — actions are still mostly ignored by the model

**Root cause:** Classic sharpening-vs-robustness tradeoff. As the model overfits to clean context frames, it produces sharper single-step outputs but becomes more brittle when fed its own (slightly imperfect) predictions as input. The noise augmentation (σ ≤ 0.3, p=0.5) isn't strong enough at this training regime to bridge the gap.

**What this means for the project:**
- Single-step prediction: **excellent** — 27 dB is strong for a 42M model at 53% of planned training
- Autoregressive rollouts: produces recognizable 3–5 frames before degrading — honest limitation of the approach at this scale
- Action conditioning: weak — the 10% CFG dropout + 15-action space is a hard learning problem; more training or stronger conditioning architecture might help
- **The model works. It just works better for single-step than for rollouts.** This is a legitimate result worth documenting honestly.

### Decision
- Proceed to Phase 9 (Demo) with the 80K checkpoint
- The demo will focus on single-step interactive prediction where the model excels
- Evaluation report will document both strengths and limitations honestly
