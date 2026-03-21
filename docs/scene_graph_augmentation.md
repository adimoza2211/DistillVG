# Scene Graph Augmentation

## Objective
Replace brittle, costly, and potentially noisy VLM paraphrasing pipelines with deterministic context expansions using pre-existing Scene Graph alignments (`_graphs_.pth`).

## Why We Switched from VLM Paraphrasing
- **Cost / Throughput Mitigation**: Inference generation of millions of records strictly blocked the pipeline, preventing rapid evaluation passes.
- **Poison Data Avoidance**: VLM models have a known rate of hallucination which diverges referential semantics. Ensuring "0 poisoned records" with arbitrary generation is difficult.
- **Ground Truth Relational Structures included**: The existing `_graphs_.pth` directly provides context-awareness centered squarely around the designated Root entity, bypassing the necessity to deduce relational structures visually.

## Indexing Mechanism
Due to differences in global indices (`dataset_index` inside metadata pointing either to the global dataset scope or `corpus.pth`), a unified cross-dataset mapping script (`verify_scene_graphs.py`) was engineered leveraging exact content hashes over purely positional joining. 

1. Both regular dataset features (`train.pth`) and graph collections are loaded.
2. Two sets of keys act as the primary lookup table:
   - Match by `(image_filename, normalized_phrase)`
   - Fallback by `(image_filename, mask_id)`
3. Alignment acts globally allowing robust correlation, yielding perfect alignment across `RefCOCO`, `RefCOCO+`, `RefCOCOg` and `ReferItGame` out-of-the-box.

## Features Extracted from Scene Graphs
Each `_graphs_` payload natively holds:
- **`nodes`**: A catalog of local actors including `root` (id:0) mapping to the ground truth box and bounding coordinates, and additional node properties providing contextual `attributes` (ex. 'distant', 'volcanic', 'white').
- **`edges`**: Object interplay detailing explicit relations representing `relation_name` (e.g. `root` `over` `clouds`) connected structurally with scalar confident weights.
- **`metadata`**: Includes canonical source images and phrase mappings.

We rely on this topological representation as a structured proxy over raw LLM-produced rewrites, driving consistency losses computationally without LLM inference overheads.
