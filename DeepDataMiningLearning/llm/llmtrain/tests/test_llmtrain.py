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
