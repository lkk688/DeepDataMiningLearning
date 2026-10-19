"""Where does the memory of a served LLM go? Weights, KV cache, recurrent state, and prefill scratch.

Loads a model with mlx-lm, then for each context length N prefills N tokens into a fresh cache and records
MLX's own counters (active / peak memory) and the exact bytes held by every cache entry.

python memory_probe.py --model mlx-community/Qwen3.8-27B-4bit --sizes 1024 4096 16384 32768 --out memory.jsonl
"""
import argparse, json, time

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

GB = 1e9


def cache_bytes(cache):
    """Bytes per cache kind: KVCache (full attention) vs ArraysCache (Gated DeltaNet conv + recurrent state)."""
    def arrays(x):                                   # ArraysCache.state is (list of arrays, padding, lengths)
        if isinstance(x, mx.array):
            yield x
        elif isinstance(x, (list, tuple)):
            for y in x:
                yield from arrays(y)
    out = {}
    for c in cache:
        kind = type(c).__name__
        out[kind] = out.get(kind, 0) + sum(a.nbytes for a in arrays(c.state))
    return out


def prefill(model, cache, tokens, step):
    for s in range(0, tokens.size, step):
        model(tokens[None, s: s + step], cache=cache)
        mx.eval([c.state for c in cache])
    mx.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    ap.add_argument("--sizes", type=int, nargs="+", default=[1024, 4096, 16384, 32768])
    ap.add_argument("--step", type=int, default=2048, help="prefill chunk (mlx-lm's default prefill step)")
    ap.add_argument("--out", default="memory.jsonl")
    a = ap.parse_args()
    model, tok = load(a.model)
    mx.eval(model.parameters())
    weights = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
    base = mx.get_active_memory()
    print(f"weights {weights / GB:.2f} GB | active after load {base / GB:.2f} GB")
    rows = [{"model": a.model, "size": 0, "weights_gb": round(weights / GB, 3), "active_gb": round(base / GB, 3)}]
    rng = mx.random.key(0)
    for n in a.sizes:
        cache = make_prompt_cache(model)
        mx.clear_cache(); mx.reset_peak_memory()
        tokens = mx.random.randint(1000, 200000, (n,), key=rng)
        t0 = time.perf_counter(); prefill(model, cache, tokens, a.step); dt = time.perf_counter() - t0
        cb = cache_bytes(cache)
        r = {"model": a.model, "size": n, "prefill_s": round(dt, 1), "prefill_tok_s": round(n / dt, 1),
             "active_gb": round(mx.get_active_memory() / GB, 3), "peak_gb": round(mx.get_peak_memory() / GB, 3),
             **{f"cache_{k}_gb": round(v / GB, 4) for k, v in cb.items()},
             "kv_bytes_per_token": round(cb.get("KVCache", 0) / n)}
        print(json.dumps(r), flush=True); rows.append(r)
        del cache; mx.clear_cache()
    with open(a.out, "a") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)


if __name__ == "__main__":
    main()
