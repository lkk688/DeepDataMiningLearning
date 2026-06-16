# P1a — VLM Ambiguity Ablation Runbook (HPC apptainer setup)

**Question:** does the optional Tier-2 VLM tier improve precision on
the A∪B − A∩B ambiguous candidate set (proposals dropped by the
strict geometric-fusion gate)?

**Backend:** apptainer-served vLLM with one of:
- **Qwen/Qwen3.6-27B-FP8** (default, served as `qwen-27b`)
- google/gemma-4-31b-it (served as `gemma-4-31b`)

Both SIF images live at `/data/rnd-liu/images/`. Local requests must
bypass the corporate proxy with `--noproxy "*"` (curl) or
`NO_PROXY=*` (env). Our `LocalVLMVoter` does this automatically.

---

## Step 1 — start the vLLM server

```bash
cd /fs/atipa/data/rnd-liu/MyRepo/DeepDataMiningLearning/DeepDataMiningLearning/detection3d/phase2a

# Default = Qwen3.6-27B-FP8 (recommended for visual grounding)
nohup bash start_vllm_server.sh > /tmp/vllm_server.log 2>&1 &

# Alternative: Gemma-4-31B
# MODEL=gemma nohup bash start_vllm_server.sh > /tmp/vllm_server.log 2>&1 &

# Wait for "Started server process" line
tail -f /tmp/vllm_server.log
```

VRAM: Qwen3.6-27B-FP8 at `--gpu-memory-utilization 0.6` uses
~48 GB on an 80 GB H100 (large context support); Gemma-4-31B at
0.9 util uses ~72 GB. **Do not launch while the B20 fullval / B21 chain
is still using GPU memory.**

Health-check with no-proxy curl:
```bash
curl --noproxy "*" -sS http://0.0.0.0:8000/v1/models
```

## Step 2 — generate ambiguous candidates (CPU + Faster R-CNN)

```bash
mkdir -p /tmp/p1a_ambig
conda run -n py310 python gen_ambiguous_candidates.py \
  --tfrec-dir /tmp/waymo143_train_tfrec_round2 \
  --out-dir /tmp/p1a_ambig \
  --max-frames 200 \
  --frame-stride 5 \
  --max-candidates 2000 \
  --frcnn-thresh 0.50
```

Output: `/tmp/p1a_ambig/ambiguous_candidates.jsonl`. Inspect with:

```bash
wc -l /tmp/p1a_ambig/ambiguous_candidates.jsonl
python -c "
import json, collections
c = collections.Counter()
for ln in open('/tmp/p1a_ambig/ambiguous_candidates.jsonl'):
    r = json.loads(ln)
    c[(r['source'], r['cls_B'] or r['cls_A'])] += 1
for k, v in sorted(c.items()): print(k, v)
"
```

## Step 3 — run VLM on candidates + audit against GT

```bash
conda run -n py310 python run_p1a_ambiguity.py \
  --candidates /tmp/p1a_ambig/ambiguous_candidates.jsonl \
  --out-json   /tmp/p1a_ambig/p1a_results.json \
  --cache-path /tmp/p1a_ambig/vlm_cache.json \
  --tfrec-dir  /tmp/waymo143_train_tfrec_round2 \
  --backend local \
  --server http://0.0.0.0:8000/v1 \
  --vlm-conf 0.7 \
  --max 2000
```

The `LocalVLMVoter` automatically sets `proxies={'http': None, 'https': None}`
and `trust_env=False` so the curl `--noproxy "*"` quirk is handled for
us; users do not need to manually export `NO_PROXY`.

To switch to Gemma-4-31B, change the started-server command in Step 1
and pass `--server http://0.0.0.0:8000/v1` plus
`--vlm-conf 0.7` (the default). The model name `qwen-27b` vs
`gemma-4-31b` is auto-resolved by vLLM if you keep the default; pass
explicit `model=...` to `LocalVLMVoter()` if you launch multiple
backends on different ports.

## Output

`/tmp/p1a_ambig/p1a_results.json` contains a `summary` array
with per-(source, class) precision-kept and precision-dropped against
Waymo v1.4.3 GT. Example structure:

```json
{
  "summary": [
    {"source": "A-only", "cls": "Cyclist", "n": 142,
     "n_vlm_kept": 47, "precision_kept": 0.681, "precision_drop": 0.092,
     "orig_precision": 0.246},
    ...
  ]
}
```

`precision_kept` ≥ `orig_precision` ≥ `precision_drop` is the "VLM
helps" pattern.

## Expected timeline

| Step | Wall-clock | GPU | Notes |
|---|---|---|---|
| Apptainer pull + start | ~2 min | ~48 GB (Qwen) | SIF cached locally |
| Model load | ~1 min | ~48 GB | weights mmap'd from SIF |
| Candidate gen | ~20 min | ~3 GB (Faster R-CNN) | 200 frames × 5-cam |
| Run VLM | ~1.5 h | ~48 GB | ~2000 calls at ~2.5 s each |

Total: ~2 h end-to-end, **after** the rest of the GPU pipeline frees up.

## Failure-mode checklist

| Symptom | Likely cause | Fix |
|---|---|---|
| HTTP timeout | corporate proxy intercepting localhost | use `--noproxy "*"` or `NO_PROXY=*`; `LocalVLMVoter` handles it |
| `Address already in use` | port 8000 still bound by previous serve | `PORT=8001 bash start_vllm_server.sh` |
| vLLM OOM | too high `--gpu-memory-utilization` | drop to `0.4` for shared GPU |
| Slow first responses | warmup of attention kernels | continue — caches after ~10 calls |
| `tfrecord not found` | candidate's tar wasn't extracted | extract the matching tar or `--max` cap |
| All `cls='None'` (parse fail) | model hallucination | check `vlm_server.log` for raw responses |
