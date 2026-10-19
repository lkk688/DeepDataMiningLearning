#!/bin/zsh
cd "$(dirname "$0")"
export CLIENT=prefill_client.py OUT=prefill.jsonl
M=mlx-community/Qwen3.8-27B-4bit; MTP=mlx-community/Qwen3.8-27B-MTP-4bit
bash serve_bench.sh pf_mlx_lm ~/venvs/mlxlm $M -- mlx_lm.server --model $M --port 8000 --chat-template-args '{"enable_thinking": false}'
bash serve_bench.sh pf_mlxvlm_mtp ~/venvs/mlxvlm $M -- python -m mlx_vlm server --model $M --draft-model $MTP --draft-kind mtp --port 8000
bash serve_bench.sh pf_lithos_dspark ~/venvs/lithos $M -- lithos-metal serve --model $M --max-context 40960 --local-files-only
bash serve_bench.sh pf_lithos_nodraft ~/venvs/lithos $M -- lithos-metal serve --model $M --no-draft --max-context 40960 --local-files-only
bash serve_bench.sh pf_lithos_dspark_c512 ~/venvs/lithos $M -- lithos-metal serve --model $M --max-context 40960 --prefill-chunk-size 512 --local-files-only
echo ALLDONE_PREFILL
