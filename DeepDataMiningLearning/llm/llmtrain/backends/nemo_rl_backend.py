"""nemo_rl backend: NVIDIA NeMo RL (Ray + vLLM generation + an AutoModel / Megatron policy), driven from llmtrain.

NeMo RL lives in its own checkout with its own uv environments (Python 3.13, pinned torch / vLLM / Ray), so we
never import it. We start from one of NeMo RL's OWN example configs and add Hydra overrides derived from the
llmtrain spec, then run its own entry point in its directory:

    cd $NEMO_RL_DIR && uv run examples/run_grpo.py --config examples/configs/grpo_math_1B.yaml  key=value ...

    grpo   -> examples/run_grpo.py          + grpo_math_1B.yaml     (math environment, verifiable reward)
    grpo + nemo_rl.gym_env (NeMo Gym, multi-step agents)
           -> examples/nemo_gym/run_grpo_nemo_gym.py + the Gym example config (default: Workplace Assistant)
    opd    -> examples/run_distillation.py  + distillation_math.yaml (on-policy distillation, reverse KL)
    recipe -> any NeMo RL script + config as is (nemo_rl.script / nemo_rl.config), with our run directory

Precedence: NeMo RL's example config  <  overrides derived from the spec  <  backend_overrides (a dict of
Hydra keys, e.g. {"loss_fn.use_cispo": true}). The command line is saved to <run_dir>/nemo_rl_command.txt.
"""
from __future__ import annotations

import json
import os
import re
import zlib
import shutil
import socket
import subprocess
import sys
from pathlib import Path

from ..data import read_jsonl, write_jsonl

SCRIPTS = {"grpo": ("examples/run_grpo.py", "examples/configs/grpo_math_1B.yaml"),
           "opd": ("examples/run_distillation.py", "examples/configs/distillation_math.yaml"),
           "gym": ("examples/nemo_gym/run_grpo_nemo_gym.py",
                   "examples/nemo_gym/grpo_workplace_assistant_nemotron_nano_v2_9b.yaml")}
GYM = "3rdparty/Gym-workspace/Gym"
DELETE = object()                                    # a Hydra "~key" override: remove the key from the base config
# Gym environments llmtrain knows how to wire into the Gym base config: NeMo Gym config file, the environment-server and
# agent names it defines, and where its data lives (train / validation JSONL; validation None = carve it from train).
GYM_ENVS = {
    "workplace_assistant": {"config": "resources_servers/workplace_assistant/configs/workplace_assistant.yaml",
                            "server": "workplace_assistant_environment_server", "agent": "workplace_assistant_simple_agent",
                            "train": "data/workplace_assistant/train.jsonl", "val": "data/workplace_assistant/validation.jsonl"},
    # PivotRL on coding-agent trajectories: one step at a "pivot" turn, scored by comparing the tool call with the
    # demonstrated one (nvidia/Nemotron-RL-Agentic-SWE-Pivot-v1, CC-BY-4.0; the recipe of grpo_qwen3_30ba3b_thinking_swe1.yaml)
    "swe_pivot": {"config": "resources_servers/single_step_tool_use_with_argument_comparison/configs/swe_pivot_single_step_tool_use_with_argument_comparison.yaml",
                  "server": "swe_pivot_single_step_tool_use_with_argument_comparison_environment_server",
                  "agent": "swe_pivot_single_step_tool_use_with_argument_comparison_agent",
                  "train": "resources_servers/single_step_tool_use_with_argument_comparison/data/train.jsonl", "val": None},
}
BASE_GYM_SERVER = "workplace_assistant_environment_server"      # the one the Gym base config already starts


def runtime_overrides(spec) -> dict:
    """Knobs that make NeMo RL run on a single, possibly SHARED GPU (see NemoRLSpec)."""
    ov = {"policy.sequence_packing.enabled": spec.nemo_rl.sequence_packing,
          "policy.generation.vllm_cfg.gpu_memory_utilization": spec.nemo_rl.gpu_mem_util}
    if spec.nemo_rl.kv_cache_gb > 0:
        ov["++policy.generation.vllm_kwargs.kv_cache_memory_bytes"] = int(spec.nemo_rl.kv_cache_gb * 2 ** 30)
    return ov


def keep_task(row: dict, n: "NemoRLSpec") -> bool:
    if n.gym_max_task_chars and len(json.dumps(row.get("responses_create_params", {}))) > n.gym_max_task_chars:
        return False
    pr = row.get("pass_rate")
    lo, hi = n.gym_pass_rate
    return pr is None or lo <= pr <= hi


def gym_data(spec, run_dir: Path, root: Path) -> dict:
    """Write the train / validation JSONL for a Gym environment into the run directory.

    Gym JSONL rows carry the task, the tools and the ground truth. They are streamed (some files are GBs), filtered
    (nemo_rl.gym_max_task_chars, gym_pass_rate), and re-stamped with the agent this config starts. Gym validates on the
    whole validation file, so it is capped at gym_val_limit. Without a validation file, ~10 % of trajectories (by a hash of
    trajectory_id, so pivots of one trajectory never straddle the split) become validation."""
    n, env = spec.nemo_rl, spec.nemo_rl.gym_env
    reg = GYM_ENVS.get(env, {"train": f"data/{env}/train.jsonl", "val": f"data/{env}/validation.jsonl", "agent": None})
    train = Path(n.gym_train or root / GYM / reg["train"])
    val = Path(n.gym_val) if n.gym_val else (root / GYM / reg["val"] if reg["val"] else None)
    for p in [train] + ([val] if val else []):
        if not p.exists():
            raise FileNotFoundError(f"{p} missing: download the environment's dataset (see the env's config 'datasets:')")
    out_dir = run_dir.resolve() / "data"; out_dir.mkdir(parents=True, exist_ok=True)

    def stamp(line):
        row = json.loads(line)
        if reg.get("agent"):
            row["agent_ref"] = {"type": "responses_api_agents", "name": reg["agent"]}
        return row

    tr, va = [], []
    with open(train) as f:
        for line in f:
            if not line.strip():
                continue
            row = stamp(line)
            if not keep_task(row, n):
                continue
            to_val = val is None and zlib.crc32(str(row.get("trajectory_id", len(tr))).encode()) % 10 == 0
            if to_val:
                if len(va) < n.gym_val_limit:
                    va.append(row)
            elif not n.gym_train_limit or len(tr) < n.gym_train_limit:
                tr.append(row)
            if n.gym_train_limit and len(tr) >= n.gym_train_limit and (val is not None or len(va) >= n.gym_val_limit):
                break
    if val is not None:
        with open(val) as f:
            va = [stamp(l) for l in f if l.strip()][: n.gym_val_limit]
    if not tr or not va:
        raise ValueError(f"gym env {env}: {len(tr)} train / {len(va)} validation tasks left after filtering")
    paths = {}
    for name, rows in (("train", tr), ("val", va)):
        p = out_dir / f"gym_{env}_{name}{len(rows)}.jsonl"
        p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        paths[name] = str(p)
    print(f"[nemo_rl] gym {env}: {len(tr)} train / {len(va)} validation tasks -> {out_dir}")
    return paths


def nemo_rl_dir(spec) -> Path:
    d = spec.nemo_rl.dir or os.environ.get("NEMO_RL_DIR", "")
    if not d or not (Path(d) / "examples").is_dir():
        raise ValueError("set nemo_rl.dir (or $NEMO_RL_DIR) to a NeMo RL checkout (git clone https://github.com/NVIDIA-NeMo/RL)")
    return Path(d).resolve()


def hydra_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(f"'{x}'" if isinstance(x, str) else hydra_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ",".join(f"{k}:" + (f"'{x}'" if isinstance(x, str) else hydra_value(x)) for k, x in v.items()) + "}"
    return str(v)


def write_data(run_dir: Path) -> dict:
    """qa records -> NeMo RL ResponseDataset JSONL: {"input": prompt, "output": verifiable answer}."""
    paths = {}
    for split in ("train", "val"):
        rows = [{"input": r["prompt"], "output": r["answer"]} for r in read_jsonl(run_dir / f"data/{split}.jsonl")]
        p = run_dir / f"data/nemo_rl_{split}.jsonl"
        write_jsonl(rows, p)
        paths[split] = str(p.resolve())
    return paths


def overrides(spec, run_dir: Path) -> dict:
    m, t, g, o = spec.model, spec.train, spec.grpo, spec.opd
    out = run_dir.resolve() / "nemo_rl"
    ov: dict = {
        "policy.model_name": m.name,
        "policy.max_grad_norm": t.max_grad_norm,
        "cluster.gpus_per_node": spec.launch.nproc_per_node,
        "cluster.num_nodes": 1,
        "checkpointing.enabled": True,
        "checkpointing.checkpoint_dir": str(out / "checkpoints"),
        "checkpointing.save_period": t.steps,
        "checkpointing.keep_top_k": 1,
        "checkpointing.metric_name": None,
        "policy.automodel_cfg.checkpoint.save_consolidated": "final",     # HF safetensors for the common evaluation
        "logger.log_dir": str(out / "logs"),
        **runtime_overrides(spec),
    }
    if m.peft == "lora":
        ov.update({"++policy.automodel_cfg.lora_cfg.enabled": True, "++policy.automodel_cfg.lora_cfg.match_all_linear": False,
                   "++policy.automodel_cfg.lora_cfg.target_modules": [t_ if ("*" in t_ or "." in t_) else f"*.{t_}" for t_ in m.lora_targets],
                   "++policy.automodel_cfg.lora_cfg.dim": m.lora_r, "++policy.automodel_cfg.lora_cfg.alpha": m.lora_alpha,
                   "++policy.automodel_cfg.lora_cfg.dropout": m.lora_dropout,
                   "++policy.automodel_cfg.lora_cfg.use_triton": False,      # the Triton LoRA kernel needs flash-attn
                   # the full block: some example configs (distillation_math) have no lora_cfg, and the schema wants all
                   "++policy.automodel_cfg.lora_cfg.exclude_modules": [], "++policy.automodel_cfg.lora_cfg.dropout_position": "post",
                   "++policy.automodel_cfg.lora_cfg.lora_A_init": "xavier", "++policy.automodel_cfg.lora_cfg.restore_from": None})
    gym = spec.method == "grpo" and bool(spec.nemo_rl.gym_env)
    if gym:   # the Gym example config is Megatron-only (policy.optimizer: null): give the AutoModel policy its optimizer
        ov["policy.optimizer"] = {"name": "torch.optim.AdamW",
                                  "kwargs": {"lr": t.lr, "weight_decay": t.weight_decay, "betas": [0.9, 0.999], "eps": 1e-8}}
    else:
        ov.update({"policy.optimizer.kwargs.lr": t.lr, "policy.optimizer.kwargs.weight_decay": t.weight_decay})
    if gym:
        paths = gym_data(spec, run_dir, nemo_rl_dir(spec))
        ov.update({"data.train.data_path": paths["train"], "data.validation.data_path": paths["val"],
                   "logger.wandb_enabled": False,
                   # the Gym example config is for 8 GPUs + Megatron; one GPU: the AutoModel (DTensor) policy
                   "policy.megatron_cfg.enabled": False, "policy.automodel_cfg.enabled": True,
                   "policy.automodel_cfg.tensor_parallel_size": spec.nemo_rl.tp_size,
                   "policy.automodel_cfg.context_parallel_size": 1,
                   "policy.max_total_sequence_length": spec.nemo_rl.gym_max_seq_len,
                   "policy.generation.vllm_cfg.http_server_serving_chat_kwargs.tool_parser": spec.nemo_rl.gym_tool_parser})
        reg = GYM_ENVS.get(spec.nemo_rl.gym_env)
        if reg and reg["server"] != BASE_GYM_SERVER:       # swap the Gym environment the base config starts
            ov.update({"env.nemo_gym.config_paths": ["responses_api_models/vllm_model/configs/vllm_model_for_training.yaml", reg["config"]],
                       f"~env.nemo_gym.{BASE_GYM_SERVER}": DELETE,
                       f"++env.nemo_gym.{reg['server']}": {"environment_servers": {"legacy_agent": {
                           "entrypoint": "app.py", "agent_server": {"type": "responses_api_agents", "name": reg["agent"]}}}}})
    elif spec.method in ("grpo", "opd"):
        paths = write_data(run_dir)
        # "++" = add-or-override: these keys are not in NeMo RL's example config (plain keys must already exist,
        # so a typo in any other override is still an error)
        ov.update({"++data.train.data_path": paths["train"], "data.validation": None,   # we evaluate with evaluate.py
                   "++data.train.dataset_name": "ResponseDataset", "++data.default.dataset_name": "ResponseDataset",
                   "++data.default.input_key": "input", "++data.default.output_key": "output",
                   "++data.default.prompt_file": None, "++data.train.split_validation_size": 0})
    if spec.method == "grpo":
        if g.loss_type not in ("grpo", "dapo", "cispo"):
            raise ValueError(f"nemo_rl backend maps grpo.loss_type grpo | dapo | cispo, not {g.loss_type!r} "
                             "(set the NeMo RL loss_fn keys through backend_overrides)")
        n = t.batch_size * t.grad_accum                                    # completions per optimizer step
        ov.update({"grpo.num_prompts_per_step": n // g.num_generations, "grpo.num_generations_per_prompt": g.num_generations,
                   "grpo.max_num_steps": t.steps, "grpo.seed": spec.seed,
                   "grpo.normalize_rewards": g.scale_rewards != "none",
                   "policy.train_global_batch_size": n, "policy.train_micro_batch_size": t.batch_size,
                   "policy.generation.temperature": g.temperature,
                   "loss_fn.reference_policy_kl_penalty": g.beta,
                   "loss_fn.ratio_clip_min": g.epsilon, "loss_fn.ratio_clip_max": g.epsilon_high,
                   "loss_fn.token_level_loss": g.loss_type != "grpo"})
        if g.loss_type == "cispo":    # older example configs (e.g. the Gym ones) have no use_cispo key
            ov["++loss_fn.use_cispo"] = True
        if gym:       # the environment's own verifier scores the agent: validate before and after training
            ov.update({"grpo.val_period": t.steps, "grpo.val_at_start": True, "++grpo.val_at_end": True,
                       "grpo.val_num_generations_per_prompt": spec.nemo_rl.val_generations,
                       "grpo.max_num_epochs": 1000,          # train.steps decides the length (the Gym config stops after 1 epoch)
                       "policy.train_micro_batch_size": min(t.batch_size, 2), "policy.logprob_batch_size": min(t.batch_size, 2)})
        else:         # single-turn: we evaluate with llmtrain/evaluate.py
            ov.update({"grpo.val_period": -1, "policy.max_total_sequence_length": g.max_prompt_len + g.max_new_tokens})
    elif spec.method == "opd":
        n = t.batch_size * t.grad_accum
        ov.update({"teacher.model_name": m.teacher, "distillation.max_num_steps": t.steps, "distillation.val_period": -1,
                   "distillation.seed": spec.seed, "distillation.num_prompts_per_step": n,
                   "loss_fn.kl_type": "reverse" if o.beta == 1.0 else ("forward" if o.beta == 0.0 else "mixed"),
                   "policy.train_global_batch_size": n, "policy.train_micro_batch_size": t.batch_size,
                   "policy.max_total_sequence_length": spec.grpo.max_prompt_len + o.max_new_tokens,
                   "teacher.max_total_sequence_length": spec.grpo.max_prompt_len + o.max_new_tokens,
                   "policy.generation.temperature": o.temperature})
        tp = spec.nemo_rl.tp_size                                          # the example config assumes 8 GPUs (TP x CP)
        for who in ("policy", "teacher"):
            ov.update({f"{who}.automodel_cfg.tensor_parallel_size": tp, f"{who}.automodel_cfg.context_parallel_size": 1})
    ov.update(spec.backend_overrides)
    return ov


def find_final(run_dir: Path) -> tuple[str, bool]:
    """-> (HF model dir, is_adapter): the consolidated weights or LoRA adapter of the last step."""
    ckpts = sorted((run_dir / "nemo_rl" / "checkpoints").glob("step_*"), key=lambda p: int(p.name.split("_")[1]))
    if not ckpts:
        return "", False
    w = ckpts[-1] / "policy" / "weights"
    for cand in (w / "model" / "consolidated", w / "consolidated", w / "model", w):    # full model or LoRA adapter
        if (cand / "adapter_config.json").exists():
            return str(cand), True
        if (cand / "config.json").exists():
            return str(cand), False
    return "", False


def parse_log(text: str) -> dict:
    """NeMo RL's stdout -> metrics: per-step training reward and loss, and every validation pass.

    With val_at_start and val_at_end the first validation is the model before training and the last is after it;
    "accuracy" there is the mean environment reward over the validation rollouts (for Gym, the task verifier).
    """
    num = r"(-?[0-9.]+(?:e-?[0-9]+)?|nan)"
    m = {"train_reward": [float(x) for x in re.findall(r"• Avg Reward: " + num, text)],
         "train_loss": [float(x) for x in re.findall(r"• Loss: " + num, text)]}
    vals = [{"accuracy": float(a), "avg_length": float(n), "samples": int(k)} for a, n, k in re.findall(
        r"Validation Results:\s*• Accuracy: " + num + r"\s*• Average response length: " + num
        + r" tokens\s*• Samples processed: (\d+)", text)]
    if vals:
        m["validation"] = vals
        m["before_val_reward"], m["after_val_reward"] = vals[0]["accuracy"], vals[-1]["accuracy"]
    return m


def val_groups(run_dir: Path, g: int) -> dict:
    """NeMo RL's per-sample validation dumps (logs/exp_*/val_data_step<N>.jsonl, prompt-major: the g samples of a
    task are consecutive) -> per validation step: accuracy, and with g > 1 pass@g and the share of informative
    groups (0 < successes < g), the only groups whose GRPO advantage is nonzero."""
    out = {}
    for f in sorted((run_dir / "nemo_rl" / "logs").glob("exp_*/val_data_step*.jsonl")):
        r = [float(x["rewards"][0] if isinstance(x["rewards"], list) else x["rewards"]) for x in read_jsonl(f)]
        if not r:
            continue
        st = {"n": len(r), "accuracy": round(sum(r) / len(r), 4)}
        if g > 1 and len(r) % g == 0:
            groups = [r[i:i + g] for i in range(0, len(r), g)]
            st["pass_at_g"] = round(sum(max(x) > 0 for x in groups) / len(groups), 4)
            st["informative_groups"] = round(sum(min(x) < max(x) for x in groups) / len(groups), 4)
        out[f.stem.replace("val_data_", "")] = st
    return out


def train(spec, run_dir: Path, kind: str) -> dict:
    if spec.method not in ("grpo", "opd", "recipe"):
        raise ValueError("nemo_rl backend: grpo | opd | recipe")
    root = nemo_rl_dir(spec)
    if spec.method == "recipe":
        if not (spec.nemo_rl.script and spec.nemo_rl.config):
            raise ValueError("method=recipe on nemo_rl needs nemo_rl.script and nemo_rl.config (paths inside the checkout)")
        script, base = spec.nemo_rl.script, spec.nemo_rl.config
        ov = {"logger.log_dir": str(run_dir.resolve() / "nemo_rl" / "logs"),
              "checkpointing.checkpoint_dir": str(run_dir.resolve() / "nemo_rl" / "checkpoints"), **runtime_overrides(spec)}
        ov.update(spec.backend_overrides)
    else:
        script, base = SCRIPTS["gym" if (spec.method == "grpo" and spec.nemo_rl.gym_env) else spec.method]
        base = spec.nemo_rl.config or base
        ov = overrides(spec, run_dir)
    uv = shutil.which("uv") or str(Path(sys.executable).parent / "uv")
    cmd = [uv, "run", script, "--config", base] + [k if v is DELETE else f"{k}={hydra_value(v)}" for k, v in ov.items()]
    (run_dir / "nemo_rl_command.txt").write_text(f"cd {root}\n" + " \\\n  ".join(cmd) + "\n")
    (run_dir / "nemo_rl_overrides.json").write_text(json.dumps(ov, indent=1, default=lambda v: "<delete>"))
    if spec.launch.dry_run:
        print("[nemo_rl] dry run:", " ".join(cmd[:5]), f"... ({len(ov)} overrides)")
        return {"final_model": "", "is_adapter": False, "dry_run": True}
    print("[nemo_rl]", " ".join(cmd[:5]), f"... ({len(ov)} overrides; see nemo_rl_command.txt)")
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", str(Path.home() / ".cache" / "uv"))
    extra = {k: str(v) for k, v in spec.nemo_rl.env.items()}
    if "CUDA_HOME" in extra:                                   # nvcc of that toolkit first on PATH, for this process only
        extra["PATH"] = f"{extra['CUDA_HOME']}/bin" + os.pathsep + extra.get("PATH", env.get("PATH", ""))
    env.update(extra)
    # Ray talks gRPC to its own node; behind an HTTP proxy that traffic must bypass the proxy (else "403" at step 1)
    local = [socket.gethostname(), socket.gethostname().split(".")[0]]
    try:
        local.append(socket.gethostbyname(socket.gethostname()))
    except OSError:
        pass
    for key in ("no_proxy", "NO_PROXY"):
        env[key] = ",".join(x for x in [env.get(key, "localhost,127.0.0.1"), *local] if x)
    with open(run_dir / "nemo_rl.log", "w") as log:
        proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"NeMo RL exited with code {rc}; see {run_dir / 'nemo_rl.log'}")
    final, is_adapter = find_final(run_dir)
    out = {"final_model": final, "is_adapter": is_adapter,
           "backend_metrics": parse_log((run_dir / "nemo_rl.log").read_text(errors="replace"))}
    g = int(ov.get("grpo.val_num_generations_per_prompt", 1) or 1)
    if groups := val_groups(run_dir, g):
        out["backend_metrics"]["val_groups"] = groups
    for k in ("before_val_reward", "after_val_reward"):           # promote to the top level, next to our own eval
        if k in out["backend_metrics"]:
            out[k] = out["backend_metrics"].pop(k)
    return out
