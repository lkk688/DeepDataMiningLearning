#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, argparse, time, math, shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

def main():
    parser = argparse.ArgumentParser(description="MMDet3D nuScenes evaluation with attn_chunk sweeps, latency & memory tracking, and paper-ready plots.")
    parser.add_argument("--config", type=str, default="work_dirs/mybevfusion7_new/mybevfusion7_crossattnaux_painting.py", help="Path to MMDet3D config .py")
    parser.add_argument("--checkpoint", type=str, default="work_dirs/mybevfusion7_new/epoch_4.pth", help="Path to model checkpoint .pth")
    parser.add_argument("--chunks", type=str, default="4096,8192,16384", help="Comma-separated attn_chunk values to try")
    parser.add_argument("--outdir", type=str, default="eval_nuscenes_results", help="Output directory for metrics and figures")
    parser.add_argument("--device", type=str, default="cuda:0", help="Torch device (e.g., cuda:0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iters for timing")
    parser.add_argument("--max-iters", type=int, default=-1, help="Limit # test iterations (for quick debugging); -1 = all")
    parser.add_argument("--num-workers", type=int, default=-1, help="Override dataloader workers; -1 = config default")
    parser.add_argument("--batch-size", type=int, default=-1, help="Override test batch size; -1 = config default")
    parser.add_argument("--force-fp16", action="store_true", help="Cast model to fp16 for timing comparison (optional)")
    parser.add_argument("--cudnn-benchmark", action="store_true", help="Enable cuDNN benchmark")
    args = parser.parse_args()

    import torch
    from mmengine import Config
    from mmengine.runner import Runner
    from mmengine.hooks import Hook
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    chunks = [int(x.strip()) for x in args.chunks.split(",") if x.strip()]
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figs").mkdir(exist_ok=True, parents=True)
    (outdir / "json").mkdir(exist_ok=True, parents=True)

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    class EvalProfileHook(Hook):
        def __init__(self, warmup=5, max_iters=-1):
            self.warmup = max(0, int(warmup))
            self.max_iters = max_iters
            self.latencies_ms: List[float] = []
            self.iters = 0
            self.start_event = None
            self.end_event = None
            self.peak_mem_bytes = 0

        def before_test(self, runner):
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                self.start_event = torch.cuda.Event(enable_timing=True)
                self.end_event = torch.cuda.Event(enable_timing=True)
            self.latencies_ms.clear()
            self.iters = 0
            self.peak_mem_bytes = 0

        def before_test_iter(self, runner, batch_idx, data_batch=None):
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                self.start_event.record()

        def after_test_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            device = torch.device(args.device if torch.cuda.is_available() else "cpu")
            bs = None
            try:
                if isinstance(data_batch, dict):
                    if "inputs" in data_batch and isinstance(data_batch["inputs"], dict):
                        if "img" in data_batch["inputs"] and data_batch["inputs"]["img"] is not None:
                            bs = data_batch["inputs"]["img"].shape[0] if hasattr(data_batch["inputs"]["img"], "shape") else None
                        elif "points" in data_batch["inputs"] and data_batch["inputs"]["points"] is not None:
                            bs = len(data_batch["inputs"]["points"])
            except Exception:
                bs = None

            if device.type == "cuda":
                self.end_event.record()
                torch.cuda.synchronize(device)
                elapsed_ms = self.start_event.elapsed_time(self.end_event)
            else:
                elapsed_ms = None

            self.iters += 1
            if self.iters > self.warmup and elapsed_ms is not None:
                if bs is not None and bs > 0:
                    self.latencies_ms.append(float(elapsed_ms) / float(bs))
                else:
                    self.latencies_ms.append(float(elapsed_ms))

            if device.type == "cuda":
                peak = torch.cuda.max_memory_allocated(device)
                if peak > self.peak_mem_bytes:
                    self.peak_mem_bytes = int(peak)

            if self.max_iters > 0 and self.iters >= self.max_iters:
                runner.test_loop._epoch_length = self.iters

        def after_test(self, runner):
            pass

    def percentile(values: List[float], p: float) -> float:
        if not values:
            return float("nan")
        s = sorted(values)
        k = (len(s)-1) * (p/100.0)
        f = math.floor(k); c = math.ceil(k)
        if f == c:
            return s[int(k)]
        d0 = s[int(f)] * (c-k)
        d1 = s[int(c)] * (k-f)
        return d0 + d1

    def run_once(cfg: Config, ckpt: str, chunk: Optional[int], tag: str, override_workers: int, override_bs: int, force_fp16: bool):
        cfg = cfg.copy()
        if override_workers > -1:
            try:
                cfg.test_dataloader.num_workers = override_workers
            except Exception:
                pass
        if override_bs > -1:
            try:
                cfg.test_dataloader.batch_size = override_bs
            except Exception:
                pass

        cfg.auto_resume = False

        if not hasattr(cfg, "test_evaluator") or cfg.test_evaluator is None:
            raise RuntimeError("Config must define test_evaluator with a NuScenes metric.")

        # if chunk is not None:
        #     #cfg.set_item("model.view_transform.attn_chunk", int(chunk))
        #     cfg["model.view_transform.attn_chunk"] = int(chunk)

        runner = Runner.from_cfg(cfg)
        runner.load_checkpoint(ckpt)
        if force_fp16:
            runner.model.half()

        prof_hook = EvalProfileHook(warmup=args.warmup, max_iters=args.max_iters)
        runner.register_hook(prof_hook, priority="LOW")
        runner.test()#run the long evaluation process

        metrics: Dict[str, Any] = {}
        try:
            if hasattr(runner.test_loop, "evaluator") and hasattr(runner.test_loop.evaluator, "metrics"):
                metrics = runner.test_loop.evaluator.metrics  # type: ignore
        except Exception:
            pass

        result = {
            "tag": tag,
            "attn_chunk": chunk,
            "metrics": metrics,
            "latency_ms": {
                "mean": float(sum(prof_hook.latencies_ms) / max(1, len(prof_hook.latencies_ms))) if prof_hook.latencies_ms else None,
                "p50": float(percentile(prof_hook.latencies_ms, 50)) if prof_hook.latencies_ms else None,
                "p90": float(percentile(prof_hook.latencies_ms, 90)) if prof_hook.latencies_ms else None,
                "p95": float(percentile(prof_hook.latencies_ms, 95)) if prof_hook.latencies_ms else None,
                "count": len(prof_hook.latencies_ms),
            },
            "gpu_memory": {
                "peak_bytes": prof_hook.peak_mem_bytes,
                "peak_gb": float(prof_hook.peak_mem_bytes) / (1024**3) if prof_hook.peak_mem_bytes else None,
            }
        }
        return result

    cfg = Config.fromfile(args.config)
    cfg.launcher = "none"
    cfg.resume = False
    cfg.load_from = None

    if args.num_workers > -1:
        try:
            cfg.test_dataloader.num_workers = args.num_workers
        except Exception:
            pass
    if args.batch_size > -1:
        try:
            cfg.test_dataloader.batch_size = args.batch_size
        except Exception:
            pass

    all_results: List[Dict[str, Any]] = []
    for chunk in [int(x) for x in args.chunks.split(",")]:
        tag = f"attn_chunk_{chunk}"
        print(f"\n=== Running evaluation: {tag} ===")
        res = run_once(cfg, args.checkpoint, chunk, tag, args.num_workers, args.batch_size, args.force_fp16)
        all_results.append(res)
        with open(Path(args.outdir) / "json" / f"{tag}.json", "w") as f:
            json.dump(res, f, indent=2)

    with open(Path(args.outdir) / "json" / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    def extract_metric(res: Dict[str, Any], key: str, default=None):
        m = res.get("metrics", {})
        if key in m:
            return m[key]
        for k, v in m.items():
            if k.endswith(key):
                return v
        return default

    import matplotlib.pyplot as plt
    labels, nds, map_, mem_gb, lat_mean, lat_p90 = [], [], [], [], [], []
    for res in all_results:
        labels.append(str(res.get("attn_chunk")))
        nds.append(extract_metric(res, "NDS", None))
        map_.append(extract_metric(res, "mAP", None))
        mem_gb.append(res.get("gpu_memory", {}).get("peak_gb"))
        lat_mean.append(res.get("latency_ms", {}).get("mean"))
        lat_p90.append(res.get("latency_ms", {}).get("p90"))

    def plot_bar(xlabels, values, ylabel, title, fname, ylim=None):
        plt.figure(figsize=(6,3))
        xs = list(range(len(xlabels)))
        plt.bar(xs, values)
        plt.xticks(xs, xlabels)
        plt.ylabel(ylabel)
        plt.title(title)
        if ylim:
            plt.ylim(*ylim)
        plt.tight_layout()
        plt.savefig(Path(args.outdir) / "figs" / fname, format="pdf")
        plt.close()

    def plot_grouped(xlabels, values_a, values_b, ylabel, legend, title, fname):
        plt.figure(figsize=(6.5,3))
        xs = list(range(len(xlabels)))
        w = 0.35
        plt.bar([x-w/2 for x in xs], values_a, width=w, label=legend[0])
        plt.bar([x+w/2 for x in xs], values_b, width=w, label=legend[1])
        plt.xticks(xs, xlabels)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(Path(args.outdir) / "figs" / fname, format="pdf")
        plt.close()

    if any(v is not None for v in nds):
        plot_bar(labels, [v if v is not None else 0 for v in nds], "NDS ↑", "nuScenes NDS vs. attn_chunk", "nds_vs_chunk.pdf", ylim=(0,1))

    if any(v is not None for v in map_):
        plot_bar(labels, [v if v is not None else 0 for v in map_], "mAP ↑", "nuScenes mAP vs. attn_chunk", "map_vs_chunk.pdf", ylim=(0,1))

    if any(v is not None for v in mem_gb):
        plot_bar(labels, [v if v is not None else 0 for v in mem_gb], "Peak GPU Memory (GB) ↓", "Peak GPU memory vs. attn_chunk", "mem_vs_chunk.pdf")

    if any(v is not None for v in lat_mean) and any(v is not None for v in lat_p90):
        plot_grouped(labels,
                     [v if v is not None else 0 for v in lat_mean],
                     [v if v is not None else 0 for v in lat_p90],
                     "Latency per frame (ms) ↓",
                     ("mean", "p90"),
                     "Per-frame latency vs. attn_chunk",
                     "latency_vs_chunk.pdf")

    def safe_fmt(x, fmt="{:.4f}"):
        return ("-" if x is None else fmt.format(x))

    lines = ["attn_chunk\tNDS\tmAP\tpeak_mem_GB\tlat_mean_ms\tlat_p90_ms"]
    for i in range(len(labels)):
        lines.append(
            "{}\t{}\t{}\t{}\t{}\t{}".format(
                labels[i],
                safe_fmt(nds[i]),
                safe_fmt(map_[i]),
                safe_fmt(mem_gb[i], "{:.3f}"),
                safe_fmt(lat_mean[i], "{:.2f}"),
                safe_fmt(lat_p90[i], "{:.2f}")
            )
        )
    Path(args.outdir, "summary.tsv").write_text("\n".join(lines))

    print("\n=== Done. Outputs ===")
    print(f"JSON:   {Path(args.outdir) / 'json' / 'all_results.json'}")
    print(f"TSV:    {Path(args.outdir) / 'summary.tsv'}")
    print(f"FIGS:   {Path(args.outdir) / 'figs'} (PDF files)")

if __name__ == "__main__":
    main()
