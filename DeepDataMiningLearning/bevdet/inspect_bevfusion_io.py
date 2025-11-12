# tools/inspect_bevfusion_io.py
import argparse, math, torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmdet3d.registry import MODELS

def as_list(x):
    if x is None: return None
    return list(x) if isinstance(x, (list, tuple)) else [x]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('-c','--config', required=True, help='Path to config.py')
    ap.add_argument('-w','--weights', default=None, help='(Optional) checkpoint path')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    init_default_scope('mmdet3d')

    # Build model from config (no dataset needed)
    model = MODELS.build(cfg.model)
    model.eval()
    if args.weights:
        print(f'Loading checkpoint: {args.weights}')
        _ = load_checkpoint(model, args.weights, map_location='cpu')

    print('\n=== LIDAR (SECOND → SECONDFPN) ===')
    bb = getattr(model, 'pts_backbone', None)
    neck = getattr(model, 'pts_neck', None)

    print('pts_backbone:', bb.__class__.__name__ if bb else None)
    # SECOND usually exposes out_channels list
    bb_out = getattr(bb, 'out_channels', None)
    print('SECOND out_channels:', as_list(bb_out))

    # SECONDFPN takes in_channels = SECOND out_channels
    neck_in = getattr(neck, 'in_channels', None)
    neck_out = getattr(neck, 'out_channels', None)
    up_strides = getattr(neck, 'upsample_strides', None)
    print('SECFPN in_channels:', as_list(neck_in))
    print('SECFPN out_channels:', as_list(neck_out))
    print('SECFPN upsample_strides:', as_list(up_strides))

    # Also read from checkpoint tensor shapes if provided
    if args.weights:
        sd = torch.load(args.weights, map_location='cpu')
        state = sd.get('state_dict', sd)
        deblock_keys = [k for k in state.keys() if 'pts_neck.deblocks' in k and k.endswith('.weight')]
        deblock_keys.sort()
        if deblock_keys:
            print('\n[Checkpoint sanity: first few deblock weights]')
            for k in deblock_keys[:6]:
                print('  ', k, tuple(state[k].shape))

    print('\n=== CAMERA (FPN → ViewTransform) ===')
    img_bb = getattr(model, 'img_backbone', None)
    img_neck = getattr(model, 'img_neck', None)
    vt = getattr(model, 'view_transform', None)

    print('img_backbone:', img_bb.__class__.__name__ if img_bb else None)
    out_indices = getattr(img_bb, 'out_indices', None)
    print('img_backbone.out_indices:', as_list(out_indices))
    print('img_neck:', img_neck.__class__.__name__ if img_neck else None)
    neck_out_ch = getattr(img_neck, 'out_channels', None)
    num_outs = getattr(img_neck, 'num_outs', None)
    print('img_neck.out_channels:', neck_out_ch)
    print('img_neck.num_outs:', num_outs)

    vt_name = vt.__class__.__name__ if vt else None
    print('view_transform:', vt_name)
    ft_size = getattr(vt, 'feature_size', None)
    final_dim = None
    # try to grab ImageAug3D final_dim from train/test pipeline for clarity
    for pl in ['train_pipeline','test_pipeline']:
        for step in cfg.get(pl, []):
            if step.get('type') == 'ImageAug3D':
                final_dim = step.get('final_dim')
                break
        if final_dim: break

    print('VT feature_size:', as_list(ft_size))
    print('Image final_dim (from pipeline):', as_list(final_dim))

    # Heuristic: deduce which FPN level VT consumes by stride ratio
    fpn_level = None
    if ft_size and final_dim:
        try:
            rx = final_dim[0] / ft_size[0]
            ry = final_dim[1] / ft_size[1]
            if abs(rx - ry) < 1e-6 and rx > 0:
                stride = int(round(rx))
                # P3≈stride8, P4≈16, P5≈32 by convention
                fpn_level = {8:'P3',16:'P4',32:'P5'}.get(stride, f'stride{stride}')
                print(f'Deduced VT consumes FPN level with stride {stride} → {fpn_level}')
            else:
                print('Non-uniform ratio; VT level not a single canonical pyramid stage.')
        except Exception as e:
            print('Could not deduce VT level:', e)

    print('\nDone.')

if __name__ == '__main__':
    main()