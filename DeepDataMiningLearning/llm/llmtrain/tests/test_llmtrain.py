"""CPU-only tests (no downloads, no GPU):   python -m pytest llmtrain/tests -q   (from DeepDataMiningLearning/llm)"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from llmtrain import data as D
from llmtrain import rewards
from llmtrain.backends import automodel_backend as AM
from llmtrain.backends.trl_backend import to_trl
from llmtrain.config import load_spec
from llmtrain.run import SUPPORT, validate

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("cfg", sorted(p.name for p in CONFIGS.glob("*.yaml") if p.name != "base.yaml"))
def test_example_configs_load_and_validate(cfg):
    spec = load_spec(str(CONFIGS / cfg))
    validate(spec)


def test_typo_is_an_error():
    with pytest.raises(KeyError):
        load_spec(str(CONFIGS / "sft_alpaca.yaml"), ["train.step=10"])


def test_method_backend_matrix():
    for backend, methods in SUPPORT.items():
        for m in ("sft", "dpo", "grpo", "opd", "kd"):
            spec = load_spec(str(CONFIGS / "sft_alpaca.yaml"), [f"backend={backend}", f"method={m}",
                                                                 "model.teacher=Qwen/Qwen2.5-1.5B-Instruct"])
            if m in methods:
                validate(spec)
            else:
                with pytest.raises(ValueError):
                    validate(spec)


def test_records_from_presets():
    spec = load_spec(str(CONFIGS / "sft_alpaca.yaml"))
    d = spec.data
    r = D.to_record("alpaca", {"instruction": "Add", "input": "2+2", "output": "4"}, d)
    assert r == {"prompt": "Add\n\n2+2", "response": "4"} and D.kind_of(r) == "sft"
    g = D.to_record("gsm8k", {"question": "Q?", "answer": "3 + 4 = <<3+4=7>>7 apples\n#### 1,007"}, d)
    assert g["answer"] == "1007" and g["response"].startswith("3 + 4 = 7 apples") and D.kind_of(g) == "qa"
    d.min_score_gap = 2.0
    row = {"chosen": [{"content": "p"}, {"content": "good"}], "rejected": [{"content": "p"}, {"content": "bad"}]}
    assert D.to_record("ultrafeedback", {**row, "score_chosen": 9, "score_rejected": 8}, d) is None
    assert D.kind_of(D.to_record("ultrafeedback", {**row, "score_chosen": 9, "score_rejected": 6}, d)) == "pref"


def test_jsonl_preset_with_column_mapping(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps({"q": f"question {i}", "a": f"answer {i}"}) for i in range(30)))
    spec = load_spec(str(CONFIGS / "sft_alpaca.yaml"),
                     ["data.preset=jsonl", f"data.path={p}", "data.columns={prompt: q, response: a}", "data.n_val=3"])
    train, val, kind = D.build(spec.data, tok=None)
    assert kind == "sft" and len(val) == 3 and len(train) == 30 - 15     # first 5*n_val rows reserved for val
    assert not {r["prompt"] for r in train} & {r["prompt"] for r in val}


@pytest.mark.parametrize("text,ref,ok", [("so 3+4 = 7.\n#### 7", "7", 1.0), ("#### 1,007", "1007", 1.0),
                                         ("the answer is \\boxed{12}", "12", 1.0), ("I think 5, maybe 6", "5", 0.0),
                                         ("no number", "5", 0.0)])
def test_answer_reward(text, ref, ok):
    assert rewards.answer_correct(None, [text], [ref]) == [ok]
    assert rewards.answer_correct(None, [[{"role": "assistant", "content": text}]], [ref]) == [ok]


def test_format_reward():
    assert rewards.answer_format(None, ["... #### 42", "42"]) == [0.1, 0.0]


class FakeTok:
    def __init__(self, chat):
        self.chat_template = "x" if chat else None
        self.eos_token = "</s>"


def test_trl_formats():
    recs = [{"prompt": "hi", "response": "hello", "answer": "1"}]
    sft = to_trl("sft", recs, FakeTok(True))[0]
    assert sft["prompt"] == [{"role": "user", "content": "hi"}] and sft["completion"][0]["role"] == "assistant"
    base = to_trl("sft", recs, FakeTok(False))[0]
    assert base["prompt"].startswith("### Instruction:") and base["completion"].endswith("</s>")
    assert to_trl("grpo", recs, FakeTok(True))[0]["answer"] == "1"
    assert to_trl("opd", recs, FakeTok(True))[0]["messages"][1]["content"] == "hello"


def _run_dir_with_data(tmp_path):
    D.write_jsonl([{"prompt": "p1", "response": "r1"}], tmp_path / "data/train.jsonl")
    D.write_jsonl([{"prompt": "p2", "response": "r2"}], tmp_path / "data/val.jsonl")
    return tmp_path


def test_automodel_sft_lora_yaml(tmp_path):
    rd = _run_dir_with_data(tmp_path)
    spec = load_spec(str(CONFIGS / "sft_alpaca.yaml"), ["backend=automodel", "launch.nproc_per_node=2"])
    y = AM.build_config(spec, rd, chat=True)
    assert y["recipe"] == "TrainFinetuneRecipeForNextTokenPrediction"
    assert y["step_scheduler"]["max_steps"] == spec.train.steps
    assert y["step_scheduler"]["global_batch_size"] == spec.train.batch_size * spec.train.grad_accum * 2
    assert y["peft"]["dim"] == spec.model.lora_r and y["peft"]["alpha"] == spec.model.lora_alpha
    assert "*.q_proj" in y["peft"]["target_modules"]                     # PEFT short names -> AutoModel globs
    assert y["dataset"]["use_hf_chat_template"] is True and y["dataset"]["answer_only_loss_mask"] is True
    assert y["rng"]["seed"] == spec.seed and y["checkpoint"]["model_save_format"] == "safetensors"
    rows = D.read_jsonl(Path(y["dataset"]["path_or_dataset_id"]))
    assert rows == [{"question": "p1", "answer": "r1"}]
    yaml.safe_dump(y)                                                     # serializable


def test_automodel_base_model_rendering_and_kd(tmp_path):
    rd = _run_dir_with_data(tmp_path)
    spec = load_spec(str(CONFIGS / "kd_gsm8k_automodel.yaml"))
    y = AM.build_config(spec, rd, chat=False)
    assert y["recipe"] == "KnowledgeDistillationRecipeForNextTokenPrediction" and "peft" not in y
    assert y["teacher_model"]["pretrained_model_name_or_path"] == spec.model.teacher and y["kd_ratio"] == spec.kd.ratio
    assert D.read_jsonl(Path(y["dataset"]["path_or_dataset_id"]))[0]["question"].startswith("### Instruction:")


def test_automodel_recipe_precedence(tmp_path):
    rd = _run_dir_with_data(tmp_path)
    recipe = {"recipe": "TrainFinetuneRecipeForNextTokenPrediction", "optimizer": {"lr": 1e-5, "_target_": "torch.optim.Adam"},
              "dataset": {"_target_": "official.Dataset"}, "step_scheduler": {"global_batch_size": 64, "local_batch_size": 8},
              "checkpoint": {"model_save_format": "torch_save"}, "ci": {"recipe_owner": "x"}}
    (rd / "r.yaml").write_text(yaml.safe_dump(recipe))
    spec = load_spec(str(CONFIGS / "recipe_automodel_qwen3_hellaswag.yaml"),
                     [f"automodel.recipe_yaml={rd / 'r.yaml'}", "backend_overrides={optimizer: {lr: 3.0e-5}}"])
    y = AM.build_config(spec, rd, chat=True)
    assert y["dataset"]["_target_"] == "official.Dataset"                 # use_our_data=false keeps the recipe data
    assert y["optimizer"] == {"lr": 3e-5, "_target_": "torch.optim.Adam"}  # backend_overrides win
    assert y["checkpoint"]["model_save_format"] == "safetensors"          # spec wins over the recipe
    assert y["step_scheduler"]["max_steps"] == spec.train.steps and "ci" not in y


def test_nemo_rl_grpo_overrides(tmp_path):
    from llmtrain.backends import nemo_rl_backend as NR
    D.write_jsonl([{"prompt": "1+1?", "answer": "2", "response": "#### 2"}], tmp_path / "data/train.jsonl")
    D.write_jsonl([{"prompt": "2+2?", "answer": "4", "response": "#### 4"}], tmp_path / "data/val.jsonl")
    spec = load_spec(str(CONFIGS / "grpo_gsm8k_nemo_rl.yaml"), ["backend_overrides={loss_fn.use_cispo: true}"])
    validate(spec)
    ov = NR.overrides(spec, tmp_path)
    g, t = spec.grpo, spec.train
    assert ov["grpo.num_generations_per_prompt"] == g.num_generations
    assert ov["grpo.num_prompts_per_step"] * g.num_generations == ov["policy.train_global_batch_size"] == t.batch_size * t.grad_accum
    assert ov["loss_fn.ratio_clip_max"] == g.epsilon_high and ov["loss_fn.reference_policy_kl_penalty"] == g.beta
    assert ov["loss_fn.use_cispo"] is True                                   # backend_overrides win
    assert "*.q_proj" in ov["++policy.automodel_cfg.lora_cfg.target_modules"]
    assert D.read_jsonl(Path(ov["++data.train.data_path"])) == [{"input": "1+1?", "output": "2"}]
    assert NR.hydra_value(["*.q_proj", 3, None, True]) == "['*.q_proj',3,null,true]"


def test_nemo_rl_opd_single_gpu(tmp_path):
    from llmtrain.backends import nemo_rl_backend as NR
    D.write_jsonl([{"prompt": "p", "answer": "1", "response": "r"}], tmp_path / "data/train.jsonl")
    D.write_jsonl([{"prompt": "p", "answer": "1", "response": "r"}], tmp_path / "data/val.jsonl")
    ov = NR.overrides(load_spec(str(CONFIGS / "opd_gsm8k_nemo_rl.yaml")), tmp_path)
    assert ov["loss_fn.kl_type"] == "reverse" and ov["teacher.model_name"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert ov["teacher.automodel_cfg.tensor_parallel_size"] == 1 and ov["policy.automodel_cfg.context_parallel_size"] == 1


def test_nemo_rl_parse_log():
    from llmtrain.backends.nemo_rl_backend import parse_log
    log = ("\n📊 Validation Results:\n    • Accuracy: 0.1250\n    • Average response length: 812.3 tokens\n"
           "    • Samples processed: 64\n"
           "\n📊 Training Results:\n  • Loss: 0.0123\n  • Generation KL Error: 0.0006\n  • Avg Reward: 0.2000\n"
           "\n📊 Training Results:\n  • Loss: -0.0040\n  • Generation KL Error: not reported\n  • Avg Reward: 0.3500\n"
           "\n📊 Validation Results:\n    • Accuracy: 0.2031\n    • Average response length: 640.0 tokens\n"
           "    • Samples processed: 64\n")
    m = parse_log(log)
    assert m["train_reward"] == [0.2, 0.35] and m["train_loss"] == [0.0123, -0.004]
    assert (m["before_val_reward"], m["after_val_reward"]) == (0.125, 0.2031)
    assert m["validation"][0]["samples"] == 64 and len(m["validation"]) == 2


def test_nemo_rl_val_groups(tmp_path):
    from llmtrain.backends.nemo_rl_backend import val_groups
    from llmtrain.data import write_jsonl
    r = [0, 0, 0, 0,  1, 1, 1, 1,  0, 1, 0, 0]                               # 3 tasks x 4 samples, prompt-major
    write_jsonl([{"content": [], "rewards": [x], "idx": i} for i, x in enumerate(r)],
                tmp_path / "nemo_rl" / "logs" / "exp_001" / "val_data_step0.jsonl")
    st = val_groups(tmp_path, 4)["step0"]
    assert st["accuracy"] == round(5 / 12, 4) and st["pass_at_g"] == round(2 / 3, 4)
    assert st["informative_groups"] == round(1 / 3, 4)                        # only the mixed group has advantage


def test_gym_pivots_select(tmp_path):
    from llmtrain.gym_pivots import select
    from llmtrain.data import write_jsonl
    write_jsonl([{"id": i} for i in range(3)], tmp_path / "data" / "gym_swe_pivot_val3.jsonl")
    r = [0, 0, 0, 0,  1, 0, 1, 0,  1, 1, 1, 1]                                  # 3 tasks x 4 samples, task-major
    write_jsonl([{"rewards": [x]} for x in r], tmp_path / "nemo_rl" / "logs" / "exp_001" / "val_data_step0.jsonl")
    st = select(tmp_path, tmp_path / "out.jsonl")
    assert st["kept"] == 1 and st["successes_histogram"] == {0: 1, 2: 1, 4: 1}
    from llmtrain.data import read_jsonl
    assert read_jsonl(tmp_path / "out.jsonl")[0] == {"id": 1, "pass_rate_policy": 0.5}


def test_gym_env_swap_overrides(tmp_path):
    from llmtrain.backends.nemo_rl_backend import GYM_ENVS, keep_task
    from llmtrain.config import load_spec
    spec = load_spec(str(CONFIGS / "gym_swe_pivot_nemo_rl.yaml"), [])
    assert spec.nemo_rl.gym_env in GYM_ENVS and GYM_ENVS["swe_pivot"]["val"] is None
    n = spec.nemo_rl
    assert keep_task({"responses_create_params": {"input": "x" * 100}, "pass_rate": 0.5}, n)
    assert not keep_task({"responses_create_params": {"input": "x" * (n.gym_max_task_chars + 1)}}, n)


# ---- launch.target=dli (NVIDIA DLI workshop VM), with a fake `workshop` package and a fake vLLM server ----
def _fake_workshop(monkeypatch, tmp_path, served):
    import sys
    import types
    calls = []
    env = types.ModuleType("workshop.env")
    env.vllm_url = lambda: "http://fake"
    env.stop_vllm = lambda: calls.append("stop_vllm")
    env.start_vllm = lambda: calls.append("start_vllm")
    env.load_adapter = lambda name, path: (calls.append(("load_adapter", name)), served.append(name))
    env.swap_vllm_to = lambda path: calls.append(("swap", path))
    tr = types.ModuleType("workshop.train")

    def train(config_path, run_id):
        calls.append(("train", run_id))
        ck = yaml.safe_load(open(config_path))["checkpoint"]["checkpoint_dir"]
        m = Path(ck) / "epoch_0_step_3" / "model"
        m.mkdir(parents=True)
        (m / "adapter_config.json").write_text("{}")
        return {"elapsed_s": 1.0, "final_loss": 0.5, "n_steps": 3}
    tr.train = train
    pkg = types.ModuleType("workshop")
    pkg.env, pkg.train = env, tr
    for name, mod in (("workshop", pkg), ("workshop.env", env), ("workshop.train", tr)):
        monkeypatch.setitem(sys.modules, name, mod)
    return calls


def test_dli_end_to_end_with_fake_workshop(tmp_path, monkeypatch):
    from llmtrain import dli, serve
    from llmtrain.run import run
    served = ["base-model"]
    calls = _fake_workshop(monkeypatch, tmp_path, served)
    monkeypatch.setattr(serve, "models", lambda url: [{"id": m, "parent": None if m == "base-model" else "base-model"} for m in served])
    # the adapter knows the answers, the base model does not
    monkeypatch.setattr(serve, "chat", lambda url, model, messages, **kw: {
        "completions": [("#### 7" if model != "base-model" else "no idea")], "usage": {"completion_tokens": 2}})
    base = {"model": {"pretrained_model_name_or_path": "/workspace/checkpoints/m"},
            "dataset": {"_target_": dli.CHAT_T, "path_or_dataset_id": "/workspace/data/x.jsonl", "seq_length": 2048},
            "validation_dataset": {"_target_": dli.CHAT_T, "path_or_dataset_id": "/workspace/data/v.jsonl"},
            "rng": {"_target_": "R", "seed": 1}, "optimizer": {"_target_": "torch.optim.Adam", "lr": 1e-5},
            "step_scheduler": {"num_epochs": 2}, "peft": {"dim": 8}, "checkpoint": {"checkpoint_dir": "/scratch/s/checkpoints"}}
    (tmp_path / "ws.yaml").write_text(yaml.safe_dump(base))
    src = tmp_path / "qa.jsonl"
    src.write_text("".join(json.dumps({"prompt": f"q{i}", "answer": "7", "response": "#### 7"}) + "\n" for i in range(30)))
    monkeypatch.setattr(dli, "tokenizer", lambda model: None)
    spec = load_spec(str(CONFIGS / "dli_sft_lora.yaml"), [
        f"data.path={src}", "data.n_train=20", "data.n_val=5", "train.steps=3", "dli.match=number",
        f"dli.root={tmp_path / 'scratch'}", f"dli.base_yaml={tmp_path / 'ws.yaml'}", "dli.system=Be brief.",
        "eval.prompts=[hello]"])
    m = run(spec)
    assert [c if isinstance(c, str) else c[0] for c in calls] == ["stop_vllm", "train", "start_vllm", "load_adapter"]
    assert m["model"] == "/workspace/checkpoints/m" and m["is_adapter"]
    assert m["before_val_answer_acc"] == 0.0 and m["after_val_answer_acc"] == 1.0
    rd = next((tmp_path / "scratch").iterdir())
    cfg = yaml.safe_load((rd / "automodel_config.yaml").read_text())
    assert cfg["dataset"]["path_or_dataset_id"].endswith("chat_train.jsonl") and cfg["dataset"]["seq_length"] == 2048
    assert cfg["validation_dataset"]["path_or_dataset_id"].endswith("chat_val.jsonl")
    assert cfg["step_scheduler"]["max_steps"] == 3 and cfg["optimizer"] == {"_target_": "torch.optim.Adam", "lr": 1e-4}
    assert cfg["rng"]["seed"] == spec.seed and cfg["peft"] == {"dim": 8}
    assert cfg["model"]["pretrained_model_name_or_path"] == "/workspace/checkpoints/m"      # dli:base keeps the workshop's
    row = json.loads((rd / "data/chat_train.jsonl").read_text().splitlines()[0])
    assert [x["role"] for x in row["messages"]] == ["system", "user", "assistant"]
    for f in ("config.yaml", "meta.json", "metrics.json", "train_summary.json", "samples_after.txt", "val_generations_before.jsonl"):
        assert (rd / f).exists(), f


@pytest.mark.parametrize("text,label,how,ok", [
    ("Category: Billing.", "billing", "contains", True), (" billing ", "Billing", "exact", True),
    ("billing issue", "billing", "exact", False), ('{"label": "SAFE"} done', "safe", "json:label", True),
    ("{bad json", "safe", "json:label", False), ("so #### 1,234", "1234", "number", True)])
def test_serve_match(text, label, how, ok):
    from llmtrain.serve import match
    assert match(text, label, how) is ok


def test_serve_chat_messages():
    from llmtrain.serve import to_messages
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
    assert [m["role"] for m in to_messages(row)] == ["system", "user"]
    assert to_messages({"q": "x", "c": "y"}, template="{q}/{c}", system="S") == [
        {"role": "system", "content": "S"}, {"role": "user", "content": "x/y"}]


def test_dli_target_needs_automodel_sft():
    spec = load_spec(str(CONFIGS / "dli_sft_lora.yaml"), ["method=kd", "model.teacher=t"])
    with pytest.raises(ValueError):
        validate(spec)
