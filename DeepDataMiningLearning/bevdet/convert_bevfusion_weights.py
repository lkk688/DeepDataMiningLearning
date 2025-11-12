# save as: tools/convert_to_weights_only.py
import argparse, torch
from collections.abc import Mapping

def load_raw(path):
    """
    Robust loader for PyTorch 2.6+: tries safe allowlist first (weights_only=True),
    then falls back to weights_only=False (ONLY if you trust the file).
    """
    try:
        from torch.serialization import safe_globals, add_safe_globals
        # allowlist common types seen in mmengine checkpoints
        try:
            from mmengine.logging.history_buffer import HistoryBuffer
            add_safe_globals([HistoryBuffer])
        except Exception:
            pass
        try:
            import numpy as np
            add_safe_globals([np.core.multiarray._reconstruct])  # seen in some ckpts
        except Exception:
            pass
        with safe_globals([]):  # globals were added above
            return torch.load(path, map_location="cpu")  # weights_only=True by default
    except Exception as e:
        print(f"[info] safe load failed ({e}); using weights_only=False (trusted ckpt).")
        return torch.load(path, map_location="cpu", weights_only=False)

def pick_state_dict(raw):
    """Pick the best weights dict from a raw checkpoint structure."""
    if isinstance(raw, Mapping):
        # Prefer EMA if present; then canonical; else guess common keys.
        for k in ("state_dict_ema", "state_dict", "model", "net"):
            if k in raw and isinstance(raw[k], Mapping):
                return raw[k]
        # Raw may already be a flat state_dict {name: Tensor}
        if all(isinstance(v, torch.Tensor) for v in raw.values()):
            return raw
    raise RuntimeError("Could not find a state_dict in the checkpoint.")

def to_tensor_only(state):
    """Ensure values are torch.Tensors; convert numpy arrays if any slipped in."""
    clean = {}
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            clean[k] = v
        else:
            try:
                import numpy as np
                if isinstance(v, np.ndarray):
                    clean[k] = torch.from_numpy(v)
                else:
                    # skip anything not tensor-like
                    continue
            except Exception:
                continue
    return clean

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.fixed_spconvv2.pth", help="original checkpoint path (.pth)")
    ap.add_argument("--output", default="modelzoo_mmdetection3d/bevfusion_lidar-cam_voxel0075_second_secfpn_8xb4-cyclic-20e_nus-3d-5239b1af.weightsonly.pth", help="output weights-only path (.pth)")
    args = ap.parse_args()

    raw = load_raw(args.input)
    state = pick_state_dict(raw)
    clean = to_tensor_only(state)

    # Save as plain dict[str, Tensor] → safe with weights_only=True
    torch.save(clean, args.output)
    print(f"Saved weights-only: {args.output}")
    print(f"  keys: {len(clean)} | total params: "
          f"{sum(v.numel() for v in clean.values())/1e6:.2f}M")

if __name__ == "__main__":
    main()