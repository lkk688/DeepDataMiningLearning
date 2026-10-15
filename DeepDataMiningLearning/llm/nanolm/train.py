"""nanolm training entry point.

    python -m nanolm.train --config nanolm/configs/02_char_transformer.yaml
    python -m nanolm.train --config nanolm/configs/02_char_transformer.yaml model.n_layers=6 seed=2
    torchrun --nproc_per_node=4 -m nanolm.train --config nanolm/configs/04_golf_baseline.yaml   # multi-GPU

Every run creates  <out_dir>/<YYYY-mm-dd_HH-MM-SS>_<name>/  containing
    config.yaml        the fully-resolved config (re-run it with --config <this file>)
    meta.json          git commit/dirty flag, seed, date, host, device, torch version, command line
    train_log.jsonl    one JSON line per log_every steps (loss, lr, tokens/s, ...)
    eval_log.jsonl     one JSON line per eval_every steps (val loss / bpb on eval_tokens)
    metrics.json       final numbers (full-val bpb, sliding bpb, quantized round-trip bpb, size, time)
    sample.txt         generated text
    model.pt           checkpoint (weights + config)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .config import load_config, save_config, to_dict
from .data import TokenData, build_tokenizer, byte_lut_tensors, prepare_dataset
from .device import autocast_ctx, get_device, memory_mb, seed_everything, synchronize
from .evaluate import evaluate, generate
from .models import build_model, count_params
from .optim import EMA, build_optimizers, lr_multiplier, set_lr
from .quant import export_artifact

PKG_DIR = Path(__file__).resolve().parent


# ======================================================================== run bookkeeping
def git_info() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(["git", *a], cwd=PKG_DIR, stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    return {"commit": run("rev-parse", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(run("status", "--porcelain", "--", ".")) if run("rev-parse", "HEAD") else None}


def make_run_dir(cfg) -> Path:
    if cfg.run_dir:
        d = Path(cfg.run_dir)
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        d = Path(cfg.out_dir) / f"{stamp}_{cfg.name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


class JsonlLogger:
    def __init__(self, path: Path, enabled: bool):
        self.f = open(path, "w") if enabled else None     # a restarted run starts a fresh log

    def log(self, **kv):
        if self.f:
            self.f.write(json.dumps(kv) + "\n")
            self.f.flush()


def code_bytes() -> int:
    """Bytes of the model-defining source (golf counts code toward the 16 MB budget)."""
    files = [PKG_DIR / "models" / "gpt.py", PKG_DIR / "quant.py"]
    return sum(f.stat().st_size for f in files if f.exists())


# ================================================================================== train
def train(cfg) -> dict:
    # ---------------------------------------------------------------- distributed + device
    ddp = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if ddp:
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    rank = dist.get_rank() if ddp else 0
    world = dist.get_world_size() if ddp else 1
    master = rank == 0
    dinfo = get_device(cfg.device, cfg.precision)
    seed_everything(cfg.seed)

    # --------------------------------------------------------------------------- run dir
    run_dir = make_run_dir(cfg) if master else None
    if master:
        save_config(cfg, run_dir / "config.yaml")
        meta = {"name": cfg.name, "seed": cfg.seed, "date": datetime.now().isoformat(timespec="seconds"),
                "git": git_info(), "host": socket.gethostname(), "platform": platform.platform(),
                "python": sys.version.split()[0], "torch": torch.__version__, "device": str(dinfo),
                "world_size": world, "command": " ".join(sys.argv), "tags": cfg.tags}
        (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[run] {run_dir}\n[run] device: {dinfo} | world_size={world}")
    train_log = JsonlLogger(run_dir / "train_log.jsonl" if master else None, master)
    eval_log = JsonlLogger(run_dir / "eval_log.jsonl" if master else None, master)

    # ------------------------------------------------------------------------------ data
    if ddp and not master:
        dist.barrier()                         # let rank 0 tokenize/download first
    data_meta = prepare_dataset(cfg.data)
    if ddp and master:
        dist.barrier()
    data = TokenData(data_meta, sampler=cfg.data.sampler, seed=cfg.seed, rank=rank, world=world)
    tok = build_tokenizer(data_meta["spec"], meta=data_meta)
    luts = byte_lut_tensors(data_meta, dinfo.device)
    if master:
        print(f"[data] {data_meta['dataset']} | tokenizer={data_meta['spec']} vocab={data_meta['vocab_size']} | "
              f"train {data.n_train_tokens:,} tok | val {data.val.size:,} tok")

    # ----------------------------------------------------------------------------- model
    if cfg.model.vocab_size <= 0:
        cfg.model.vocab_size = data_meta["vocab_size"]
    cfg.model.max_seq_len = max(cfg.model.max_seq_len, cfg.train.seq_len, cfg.eval.eval_seq_len)
    model = build_model(cfg.model).to(dinfo.device)
    n_params = count_params(model)
    n_params_ne = count_params(model, exclude_embeddings=True)
    if master:
        print(f"[model] {cfg.model.arch}: {n_params:,} params ({n_params_ne:,} non-embedding)")
    optimizers = build_optimizers(model, cfg.train, dinfo)
    ema = EMA(model, cfg.train.ema_decay) if cfg.train.ema_decay > 0 else None
    fwd = model
    if cfg.train.compile and dinfo.compile_ok:
        fwd = torch.compile(model)
    if ddp:
        fwd = torch.nn.parallel.DistributedDataParallel(fwd, device_ids=[dinfo.device.index]
                                                        if dinfo.type == "cuda" else None)
    use_scaler = dinfo.type == "cuda" and dinfo.amp_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    # -------------------------------------------------------------------------- training
    tc = cfg.train
    B, T, A = tc.batch_size, tc.seq_len, tc.grad_accum
    tokens_per_step = B * T * A * world
    if master:
        print(f"[train] {tc.max_steps} steps x {tokens_per_step:,} tokens/step "
              f"({tc.max_steps * tokens_per_step / max(data.n_train_tokens, 1):.2f} passes over train) "
              f"| optimizer={tc.optimizer} schedule={tc.schedule}"
              + (f" | time budget {tc.max_time_sec:.0f}s" if tc.max_time_sec else ""))
    qat_on = False
    step, train_time, loss_acc, loss_n = 0, 0.0, 0.0, 0
    last_log = (0, 0.0)                        # (step, train_time) at the previous log line
    synchronize(dinfo)
    t_last = time.perf_counter()
    model.train()
    while True:
        progress = step / tc.max_steps
        if tc.max_time_sec > 0:
            progress = max(progress, train_time / tc.max_time_sec)
            if ddp:                            # all ranks must agree on when to stop
                p = torch.tensor(progress, device=dinfo.device)
                dist.all_reduce(p, op=dist.ReduceOp.MAX)
                progress = p.item()
        done = progress >= 1.0

        # ---- periodic cheap validation (on the first eval_tokens of val)
        if master and tc.eval_every > 0 and (step % tc.eval_every == 0 or done) and step > 0:
            synchronize(dinfo)
            train_time += time.perf_counter() - t_last
            sd_backup = None
            if ema is not None:
                sd_backup = {k: v.clone() for k, v in model.state_dict().items()}
                model.load_state_dict(ema.state_dict_for(model))
            r = evaluate(model, data.val_windows(T, tc.eval_tokens), T, cfg.eval.batch_size, luts, dinfo)
            if sd_backup is not None:
                model.load_state_dict(sd_backup)
            eval_log.log(step=step, train_tokens=step * tokens_per_step, time=round(train_time, 2), **r)
            print(f"  eval step {step:6d} | val_loss {r['loss']:.4f} | val_bpb {r['bpb']:.4f} | "
                  f"acc {r['acc'] * 100:.1f}% | {train_time:.0f}s")
            synchronize(dinfo)
            t_last = time.perf_counter()
        if done:
            break

        # ---- schedule-driven switches
        if cfg.model.recur_layers and not getattr(model, "recur_active", True) \
                and progress >= cfg.model.recur_start_frac:
            model.recur_active = True
            if master:
                print(f"  [step {step}] depth recurrence ON -> layer schedule {model.layer_schedule()}")
        if tc.qat_bits and not qat_on and progress >= tc.qat_start_frac and hasattr(model, "set_qat"):
            model.set_qat(tc.qat_bits)
            qat_on = True
            if master:
                print(f"  [step {step}] QAT int{tc.qat_bits} ON")
        set_lr(optimizers, lr_multiplier(tc, step, progress), step, tc)

        # ---- one optimizer step = A micro-batches
        for micro in range(A):
            x, y = data.train_batch(B, T, dinfo.device)
            sync_ctx = fwd.no_sync() if (ddp and micro < A - 1) else _null()
            with sync_ctx:
                with autocast_ctx(dinfo):
                    logits = fwd(x)
                if tc.loss_on == "last":           # teaching contrast: one target per window
                    loss = F.cross_entropy(logits[:, -1].float(), y[:, -1])
                else:                              # standard: every position predicts its next token
                    loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
                scaler.scale(loss / A).backward()
            loss_acc += loss.detach()
            loss_n += 1
        if tc.grad_clip > 0:
            for opt in optimizers:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
        for opt in optimizers:
            scaler.step(opt)
        scaler.update()
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if ema is not None:
            ema.update(model, step)
        step += 1

        if step % tc.log_every == 0:
            synchronize(dinfo)
            now = time.perf_counter()
            train_time += now - t_last
            t_last = now
            avg = (loss_acc / loss_n).item()
            if not torch.isfinite(torch.tensor(avg)):
                raise FloatingPointError(f"loss became {avg} at step {step} -- lower the LR or check data")
            lr0 = optimizers[-1].param_groups[0]["lr"]          # Muon lr when optimizer=muon
            tps = tokens_per_step * (step - last_log[0]) / max(train_time - last_log[1], 1e-9)
            last_log = (step, train_time)
            train_log.log(step=step, loss=avg, lr=lr0, time=round(train_time, 2), tok_per_s=round(tps))
            if master:
                print(f"step {step:6d} | loss {avg:.4f} | lr {lr0:.2e} | {tps / 1e3:.1f}k tok/s | {train_time:.0f}s")
            loss_acc, loss_n = 0.0, 0

    synchronize(dinfo)
    train_time += time.perf_counter() - t_last

    # ------------------------------------------------------------------------ final eval
    if ema is not None:
        model.load_state_dict(ema.state_dict_for(model))
    if hasattr(model, "set_qat"):
        model.set_qat(0)
    ev = cfg.eval
    eT = ev.eval_seq_len or T
    val = data.val_windows(eT, ev.final_eval_tokens)
    metrics = {"name": cfg.name, "seed": cfg.seed, "arch": cfg.model.arch, "params": n_params,
               "params_non_embedding": n_params_ne, "steps": step, "tokens_seen": step * tokens_per_step,
               "train_time_sec": round(train_time, 2), "tok_per_sec": round(step * tokens_per_step / max(train_time, 1e-9)),
               "peak_mem_mb": round(memory_mb(dinfo), 1), "device": str(dinfo), "world_size": world,
               "dataset": data_meta["dataset"], "tokenizer": data_meta["spec"]}
    t0 = time.perf_counter()
    r = evaluate(model, val, eT, ev.batch_size, luts, dinfo, rank=rank, world=world)
    metrics.update({f"val_{k}": v for k, v in r.items()})
    metrics["eval_time_sec"] = round(time.perf_counter() - t0, 2)
    if master:
        print(f"[final] val_loss {r['loss']:.4f} | val_bpb {r['bpb']:.4f} | ppl {r['ppl']:.2f} | "
              f"acc {r['acc'] * 100:.2f}% | {r['tokens']:,} tokens, {r['bytes_per_token']:.2f} bytes/token")
    if ev.sliding_stride > 0 and cfg.model.arch == "gpt":
        rs = evaluate(model, val, eT, ev.batch_size, luts, dinfo, stride=ev.sliding_stride, rank=rank, world=world)
        metrics.update({f"sliding_{k}": v for k, v in rs.items() if k in ("loss", "bpb", "tokens")})
        if master:
            print(f"[final] sliding(stride={ev.sliding_stride}) val_bpb {rs['bpb']:.4f}")

    if master:
        if cfg.export.quant_bits:
            blob, deq = export_artifact(model, cfg.export.quant_bits, cfg.export.compressor)
            (run_dir / f"model.int{cfg.export.quant_bits}.{cfg.export.compressor}").write_bytes(blob)
            sd_backup = {k: v.clone() for k, v in model.state_dict().items()}
            model.load_state_dict({k: v.to(dinfo.device) for k, v in deq.items()})
            rq = evaluate(model, val, eT, ev.batch_size, luts, dinfo,
                          stride=ev.sliding_stride if cfg.model.arch == "gpt" else 0)
            model.load_state_dict(sd_backup)
            cb = code_bytes()
            metrics.update({"quant_bits": cfg.export.quant_bits, "model_bytes": len(blob), "code_bytes": cb,
                            "artifact_bytes": len(blob) + cb,
                            "fits_budget": (len(blob) + cb) <= cfg.export.budget_mb * 1e6,
                            "quant_val_bpb": rq["bpb"], "quant_gap_bpb": rq["bpb"] - (
                                metrics.get("sliding_bpb", metrics["val_bpb"]))})
            print(f"[export] int{cfg.export.quant_bits}+{cfg.export.compressor}: {len(blob) / 1e6:.3f} MB model "
                  f"+ {cb / 1e3:.0f} KB code | round-trip val_bpb {rq['bpb']:.4f} "
                  f"(gap {metrics['quant_gap_bpb']:+.4f})")
        if ev.sample_tokens > 0:
            txt = generate(model, tok, dinfo, ev.sample_prompt, ev.sample_tokens, eT, ev.temperature, ev.top_k,
                           seed=cfg.seed, top_p=ev.top_p)
            (run_dir / "sample.txt").write_text(txt)
            print(f"[sample]\n{txt[:600]}\n")
        if cfg.train.save_checkpoint:
            torch.save({"model": model.state_dict(), "config": to_dict(cfg), "data_meta": data_meta},
                       run_dir / "model.pt")
        metrics["run_dir"] = str(run_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
        print(f"[done] metrics -> {run_dir / 'metrics.json'}")
    if ddp:
        dist.destroy_process_group()
    return metrics


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def main(argv=None):
    ap = argparse.ArgumentParser(description="nanolm trainer (config-driven)")
    ap.add_argument("--config", "-c", type=str, default=None, help="YAML experiment config")
    ap.add_argument("overrides", nargs="*", help="key.sub=value overrides, e.g. model.n_layers=6")
    args = ap.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)     # live logs when redirected to a file
    cfg = load_config(args.config, args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
