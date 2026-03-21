# Compression & Export

## Pruning Strategy (`src/compression/pruning.py`)
Applied only to the **fusion transformer** (4 layers × d=384):
- **FFN channels**: 20% removed by L1-norm ranking per layer
- **Attention heads**: 1 head removed per layer by gradient-norm ranking
- Followed by 5 fine-tuning epochs to recover accuracy

**Do not prune**: backbone (YOLOv8), text encoder (MobileCLIP), verifier head.

## QAT (`src/compression/qat.py`)
- Mixed INT8/INT4 quantisation-aware training
- 5 epochs
- Calibration: 15,000 samples stratified by expression length (RefCOCO short / RefCOCOg long / mixed)
- Framework: PyTorch FX quantisation or brevitas
- Target: INT8 model ≈ 30 MB on disk, ~3–4 GFLOPs

## Training → Compression Handoff Requirements
- Compression must consume checkpoints that include precision mode, seed, and config snapshot metadata
- Teacher artifacts must never be part of export bundles
- QAT config must be reproducible and Hydra-driven (bit-widths, calibration set size, epochs)
- Maintain split-wise evaluation protocol before and after compression

## Export Targets (`src/export/`)
| Target | Script | Platform |
|---|---|---|
| ONNX | `export_onnx.py` | All |
| TensorRT | `export_trt.py` | Jetson (Nvidia) |
| CoreML | `export_coreml.py` | Apple Silicon (A17 NPU) |
| ONNX Runtime Mobile | `export_onnx.py --mobile` | ARM Android (Snapdragon 8 Gen 3) |

## Latency Benchmarking
- Measured on **physical hardware only** (not emulated)
- Devices: Snapdragon 8 Gen 3 Android, Apple A17 NPU device
- Metric: end-to-end latency (image decode → bounding box output), median over 200 runs, 50 warm-up
- Report alongside Accuracy@0.5 in the paper
