# Efficiency Playbook

## Purpose
Define non-negotiable training efficiency and reproducibility policies for DistillVG. All knobs must be configurable through Hydra.

## Required Efficiency Features

### Precision
- Default: AMP (`autocast` + `GradScaler`)
- Hardware-specific BF16 can be enabled only through config and validated benchmarks
- Teacher inference remains deterministic and detached from gradient flow

### Effective Batch Size
- Use gradient accumulation to simulate larger batches under memory constraints
- Track and log:
  - per-device micro-batch
  - accumulation steps
  - effective global batch

### Memory Controls
- Support activation checkpointing on fusion transformer blocks
- Expose gradient clipping via config (`clip_grad_norm_`)
- Monitor and log peak GPU memory every epoch

### Data Throughput
Expose and tune via config:
- `num_workers`
- `pin_memory`
- `persistent_workers`
- `prefetch_factor`

Teacher cache access should avoid pathological random I/O patterns.

### Runtime Backends
- Compile path (`torch.compile`) must be toggleable with fallback
- Distributed path (DDP) must be toggleable with clear single-GPU baseline path
- No hidden runtime switches in source files

## Required Logging Contract (each epoch)
- Accuracy@0.5 for each required split
- Loss decomposition (`CE`, `KL`, `L1`, `GIoU`, `MSE`, `attn`, `SelfEQ`, `verifier`)
- Effective batch size
- Precision mode and gradient scale
- Peak GPU memory
- Epoch time

## Reproducibility Contract
Persist in checkpoints and tracking metadata:
- Random seed(s)
- Full config snapshot
- Commit hash / run identifier
- Precision/runtime toggles

## Config Surface Checklist
Hydra config surfaces should include:
- `train/precision`
- `train/optimization`
- `train/schedule`
- `train/memory`
- `train/distributed`
- `train/runtime` (compile/checkpointing toggles)
- `data/loader`
- `logging/wandb`

## Implementation Notes
- Efficiency defaults are required unless explicitly overridden for ablations
- Any ablation that disables efficiency features must be tagged and logged as a policy exception
- Benchmark trade-offs (throughput vs stability) should be documented in `docs/repo_state.md`
