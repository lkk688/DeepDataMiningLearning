"""Phase 2a — annotation-free pseudo-labeling for domain adaptation.

Components:
  - cluster_proposer: Validator A (LiDAR DBSCAN)
  - cam2d_proposer:   Validator B (2D Faster R-CNN + LiDAR depth lift)
  - vlm_voter:        Tier-2 tie-breaker via NVIDIA Gemma VLM API
  - fusion:           A ↔ B matching + conflict labels
  - pseudo_labeler:   end-to-end driver → nuScenes-format info pkls
"""
