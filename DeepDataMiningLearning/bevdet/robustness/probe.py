"""Fusion-robustness probe -- single entry point.

Question this answers
---------------------
The nuScenes val leaderboard ranks LiDAR-camera fusion detectors by NDS on clean
data.  Does that ranking survive when the sensors degrade?  If it does, the
"which fusion representation" question is settled by the leaderboard and there is
nothing for us to add.  If it inverts, the leaderboard is measuring the wrong
thing and the inversion is the result.

The probe evaluates a set of checkpoints across a set of sensor conditions on the
**full official nuScenes val split with the official metric**, so every number is
directly comparable to published ones.  Corruptions are applied at the sensor
level and are seeded per-sample, so a condition is byte-identical across models.

Usage
-----
    cd /data/rnd-liu/MyRepo/mmdetection3d
    python projects/bevdet/robustness/probe.py \
        --config projects/bevdet/robustness/configs/gonogo_v1.yaml

    # continue an interrupted sweep (skips cells that already have results)
    python projects/bevdet/robustness/probe.py \
        --config .../gonogo_v1.yaml --resume outputs/2026-09-04_.../

Each cell runs in its own subprocess so a crash in one condition cannot take the
sweep down and GPU memory cannot leak across 40+ evaluations.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import os.path as osp
import random
import subprocess
import sys
import time
from datetime import datetime

import yaml


# --------------------------------------------------------------------------- #
# reproducibility
# --------------------------------------------------------------------------- #

def set_global_seed(seed: int) -> None:
    """Fix every RNG we can reach in one call."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def git_meta(repo: str) -> dict:
    def run(*args):
        try:
            return subprocess.check_output(
                ['git', '-C', repo, *args], text=True,
                stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None
    return {
        'commit': run('rev-parse', 'HEAD'),
        'branch': run('rev-parse', '--abbrev-ref', 'HEAD'),
        'dirty': bool(run('status', '--porcelain')),
    }


# --------------------------------------------------------------------------- #
# worker: one (model, condition) cell
# --------------------------------------------------------------------------- #

LOAD_PREFIXES = ('Load', 'BEVLoad')

# import path of corruptions.py as seen from the mmdet3d root, where
# projects/bevdet is a symlink to this repo's DeepDataMiningLearning/bevdet
CORRUPTIONS_MODULE = 'projects.bevdet.robustness.corruptions'


def _insert_corruption(pipeline: list, condition: str, ops: list,
                       seed: int) -> list:
    """Put SensorCorruption directly after the last loading transform."""
    last_load = -1
    for i, t in enumerate(pipeline):
        if str(t.get('type', '')).startswith(LOAD_PREFIXES):
            last_load = i
    if last_load < 0:
        raise RuntimeError('no Load* transform found in the test pipeline; '
                           'cannot place the corruption at sensor level')
    out = copy.deepcopy(pipeline)
    out.insert(last_load + 1, dict(type='SensorCorruption', condition=condition,
                                   ops=ops, seed=seed))
    return out


def run_cell(cfg_yaml: dict, model: dict, condition: dict, out_dir: str) -> dict:
    """Evaluate one checkpoint under one condition. Runs inside the worker."""
    import importlib

    from mmengine.config import Config
    from mmengine.runner import Runner

    seed = int(cfg_yaml.get('seed', 0))
    set_global_seed(seed)

    # Register SensorCorruption BEFORE anything builds a pipeline.  Appending to
    # cfg.custom_imports would be too late: Config.fromfile() processes that key
    # at load time, so a later append is silently ignored.  Dataloader workers
    # are forked, so registering here covers them too.
    root = osp.abspath(cfg_yaml['mmdet3d_root'])
    if root not in sys.path:
        sys.path.insert(0, root)
    importlib.import_module(CORRUPTIONS_MODULE)

    # torch>=2.6 defaults torch.load to weights_only=True, which refuses a full
    # mmengine *training* checkpoint: its meta/message_hub pickles HistoryBuffer,
    # numpy reconstructors and whatever else the trainer stashed.  Allowlisting
    # those one by one is whack-a-mole, so the worker opts out for its own
    # process.  This is a deliberate trust decision and it is narrow: the worker
    # only ever loads checkpoint paths named in the probe YAML, which are local
    # files the user already trains and evaluates with.  Do not widen it to
    # checkpoints from an untrusted source -- unpickling those runs their code.
    import torch
    if not getattr(torch.load, '_probe_patched', False):
        _orig_load = torch.load

        def _load_trusted_local(*a, **kw):
            kw.setdefault('weights_only', False)
            return _orig_load(*a, **kw)

        _load_trusted_local._probe_patched = True
        torch.load = _load_trusted_local
    for extra in model.get('extra_imports', []):
        importlib.import_module(extra)

    cfg = Config.fromfile(model['config'])
    imports = list(cfg.get('custom_imports', {}).get('imports', []))
    if CORRUPTIONS_MODULE not in imports:
        imports.append(CORRUPTIONS_MODULE)      # keep the dumped cfg self-describing
    cfg.custom_imports = dict(imports=imports, allow_failed_imports=False)

    # Per-model environment, e.g. BEV_MODALITY=lidar to zero one BEV branch
    # *inside* the model. That gives a within-model floor -- same weights, camera
    # branch masked -- which is stricter than comparing against a different
    # LiDAR-only checkpoint. Only BEVFusionCA / FlowGuidedTemporalBEVFusion read
    # it (bevfusion_ca.py `_modality_mask`); it is a no-op elsewhere, so a model
    # that ignores it will silently produce its normal score -- check the class
    # before trusting a floor row.
    for k, v in (model.get('env') or {}).items():
        os.environ[str(k)] = str(v)

    cfg.work_dir = out_dir
    cfg.load_from = model['checkpoint']
    cfg.randomness = dict(seed=seed, deterministic=False)
    cfg.log_level = 'INFO'

    dl = cfg_yaml.get('dataloader', {})
    if 'batch_size' in dl:
        cfg.test_dataloader.batch_size = dl['batch_size']
        cfg.val_dataloader.batch_size = dl['batch_size']
    if 'num_workers' in dl:
        cfg.test_dataloader.num_workers = dl['num_workers']
        cfg.val_dataloader.num_workers = dl['num_workers']

    pipeline = cfg.test_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.pipeline = _insert_corruption(
        pipeline, condition['key'], condition.get('ops', []), seed)

    # some configs alias test_dataloader = val_dataloader; keep them distinct
    cfg.val_dataloader = copy.deepcopy(cfg.test_dataloader)

    for k, v in (model.get('cfg_options') or {}).items():
        cfg.merge_from_dict({k: v})

    t0 = time.time()
    runner = Runner.from_cfg(cfg)
    metrics = runner.test()
    elapsed = time.time() - t0

    def pick(suffix):
        for k, v in metrics.items():
            if k.endswith(suffix):
                return float(v)
        return None

    return {
        'model': model['key'],
        'condition': condition['key'],
        'NDS': pick('/NDS'),
        'mAP': pick('/mAP'),
        'mATE': pick('/mATE'),
        'mASE': pick('/mASE'),
        'mAOE': pick('/mAOE'),
        'mAVE': pick('/mAVE'),
        'mAAE': pick('/mAAE'),
        'elapsed_s': round(elapsed, 1),
        'metrics': {k: (float(v) if isinstance(v, (int, float)) else v)
                    for k, v in metrics.items()},
    }


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def cell_dir(run_dir: str, model_key: str, cond_key: str) -> str:
    return osp.join(run_dir, 'cells', f'{model_key}__{cond_key}')


def write_summary(run_dir: str) -> None:
    """(Re)build summary.csv and a pivot table from whatever cells have finished."""
    rows = []
    cells = osp.join(run_dir, 'cells')
    for name in sorted(os.listdir(cells)) if osp.isdir(cells) else []:
        f = osp.join(cells, name, 'result.json')
        if osp.exists(f):
            with open(f) as fh:
                rows.append(json.load(fh))
    if not rows:
        return
    cols = ['model', 'condition', 'NDS', 'mAP', 'mATE', 'mASE', 'mAOE',
            'mAVE', 'mAAE', 'elapsed_s']
    with open(osp.join(run_dir, 'summary.csv'), 'w') as fh:
        fh.write(','.join(cols) + '\n')
        for r in rows:
            fh.write(','.join('' if r.get(c) is None else str(r.get(c))
                              for c in cols) + '\n')

    models = sorted({r['model'] for r in rows})
    conds = sorted({r['condition'] for r in rows})
    table = {(r['model'], r['condition']): r for r in rows}
    lines = ['| condition | ' + ' | '.join(models) + ' |',
             '|---' * (len(models) + 1) + '|']
    for c in conds:
        cells_txt = []
        for m in models:
            r = table.get((m, c))
            cells_txt.append('—' if r is None or r.get('NDS') is None
                             else f"{r['NDS']:.4f}")
        lines.append(f'| {c} | ' + ' | '.join(cells_txt) + ' |')
    with open(osp.join(run_dir, 'summary_NDS.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True, help='probe YAML')
    ap.add_argument('--resume', default=None,
                    help='existing run dir to continue (skips finished cells)')
    ap.add_argument('--only', default=None,
                    help='substring filter on "<model>__<condition>"')
    ap.add_argument('--dry-run', action='store_true')
    # worker mode
    ap.add_argument('--cell', default=None, help='internal: "<model>::<cond>"')
    ap.add_argument('--run-dir', default=None, help='internal')
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg_yaml = yaml.safe_load(fh)

    models = {m['key']: m for m in cfg_yaml['models']}
    conds = {c['key']: c for c in cfg_yaml['conditions']}

    # ---------------- worker ----------------
    if args.cell:
        mk, ck = args.cell.split('::')
        out = cell_dir(args.run_dir, mk, ck)
        os.makedirs(out, exist_ok=True)
        res = run_cell(cfg_yaml, models[mk], conds[ck], out)
        with open(osp.join(out, 'result.json'), 'w') as fh:
            json.dump(res, fh, indent=2)
        print(f"[probe] {mk} / {ck}: NDS={res['NDS']} mAP={res['mAP']}")
        return 0

    # ---------------- driver ----------------
    mmdet3d_root = cfg_yaml['mmdet3d_root']
    if args.resume:
        run_dir = args.resume
    else:
        stamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        run_dir = osp.join(cfg_yaml['output_root'],
                           f"{stamp}_{cfg_yaml['name']}")
    os.makedirs(osp.join(run_dir, 'cells'), exist_ok=True)

    if not args.resume:
        meta = {
            'name': cfg_yaml['name'],
            'date': datetime.now().isoformat(timespec='seconds'),
            'seed': cfg_yaml.get('seed', 0),
            'host': os.uname().nodename,
            'python': sys.executable,
            'git': git_meta(cfg_yaml.get('repo_root', '.')),
            'git_mmdet3d': git_meta(mmdet3d_root),
            'config_path': osp.abspath(args.config),
        }
        try:
            meta['gpu'] = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                text=True).strip()
        except Exception:
            meta['gpu'] = None
        with open(osp.join(run_dir, 'meta.json'), 'w') as fh:
            json.dump(meta, fh, indent=2)
        with open(osp.join(run_dir, 'config.yaml'), 'w') as fh:
            yaml.safe_dump(cfg_yaml, fh, sort_keys=False)

    cells = [(m['key'], c['key'])
             for m in cfg_yaml['models'] for c in cfg_yaml['conditions']
             if c['key'] not in (m.get('skip_conditions') or [])]
    if args.only:
        cells = [c for c in cells if args.only in f'{c[0]}__{c[1]}']

    todo = [(m, c) for m, c in cells
            if not osp.exists(osp.join(cell_dir(run_dir, m, c), 'result.json'))]

    print(f'[probe] run dir : {run_dir}')
    print(f'[probe] cells   : {len(cells)} total, {len(todo)} to run')
    if args.dry_run:
        for m, c in todo:
            print(f'  would run {m} :: {c}')
        return 0

    python = cfg_yaml.get('python', sys.executable)
    failures = []
    for i, (m, c) in enumerate(todo, 1):
        out = cell_dir(run_dir, m, c)
        os.makedirs(out, exist_ok=True)
        log = osp.join(out, 'cell.log')
        print(f'[probe] ({i}/{len(todo)}) {m} :: {c} -> {log}', flush=True)
        cmd = [python, '-u', osp.abspath(__file__),
               '--config', osp.abspath(args.config),
               '--cell', f'{m}::{c}', '--run-dir', osp.abspath(run_dir)]
        with open(log, 'w') as fh:
            rc = subprocess.call(cmd, cwd=mmdet3d_root, stdout=fh,
                                 stderr=subprocess.STDOUT)
        if rc != 0:
            failures.append((m, c, rc))
            print(f'[probe]   FAILED rc={rc} (see {log})', flush=True)
        write_summary(run_dir)

    write_summary(run_dir)
    print(f'[probe] done. summary: {osp.join(run_dir, "summary.csv")}')
    if failures:
        print(f'[probe] {len(failures)} cell(s) failed:')
        for m, c, rc in failures:
            print(f'  {m} :: {c} (rc={rc})')
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
