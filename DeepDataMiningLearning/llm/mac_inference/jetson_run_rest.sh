#!/bin/bash
# The Jetson Thor configurations of TUTORIAL_mac_inference.md §10. Usage: jetson_run_rest.sh fn|27mtp|27 (run_one.sh = jetson_run_one.sh)
VLLM=(docker run -e HF_HUB_OFFLINE=1 --runtime=nvidia --gpus all --network host --ipc=host -v $HOME/.cache/huggingface:/root/.cache/huggingface -v $HOME/.cache/vllm:/root/.cache/vllm ghcr.io/nvidia-ai-iot/vllm:qwen3.8-next-jetson-thor)
case "$1" in
 fn) ~/llmbench/run_one.sh thor_vllm_fn_nvfp4_mtp 8000 Qwen3.8-Flash-Next -- "${VLLM[@]}" local-inference-lab/Qwen3.8-Flash-Next-NVFP4 --revision ada4da32a583a78aa47299f45a70603c950490b8 --served-model-name Qwen3.8-Flash-Next --gpu-memory-utilization 0.93 --max-num-seqs 1 --mamba-ssm-cache-dtype bfloat16 --hf-overrides "{\"architectures\":[\"Qwen4ExpForConditionalGeneration\"],\"model_type\":\"qwen4_exp\"}" --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3,\"model\":\"/opt/qwen38-mtp-model\"}" ;;
 27mtp) ~/llmbench/run_one.sh thor_vllm_27b_nvfp4_mtp 8000 Qwen3.8-27B -- "${VLLM[@]}" nvidia/Qwen3.8-27B-NVFP4 --served-model-name Qwen3.8-27B --gpu-memory-utilization ${UTIL:-0.4} --max-num-seqs 1 --max-model-len 65536 --mamba-ssm-cache-dtype bfloat16 --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":3}" ;;
 27) ~/llmbench/run_one.sh thor_vllm_27b_nvfp4 8000 Qwen3.8-27B -- "${VLLM[@]}" nvidia/Qwen3.8-27B-NVFP4 --served-model-name Qwen3.8-27B --gpu-memory-utilization ${UTIL:-0.4} --max-num-seqs 1 --max-model-len 65536 --mamba-ssm-cache-dtype bfloat16 ;;
esac
