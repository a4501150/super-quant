.PHONY: all setup download convert calibrate recalibrate imatrix sensitivity quantize quantize-nvfp4 convert-nvfp4 benchmark bench-spec bench-concurrent bench-kv bench-sglang compare serve serve-sglang stop-sglang serve-vllm stop-vllm clean help

SHELL := /bin/bash
PROJECT_DIR := $(shell pwd)
SCRIPTS := $(PROJECT_DIR)/scripts
SRC := $(PROJECT_DIR)/src
UV := uv run --project $(PROJECT_DIR)

# Load model config from model.env
MODEL_ID := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$MODEL_ID')
MODEL_NAME := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$MODEL_NAME')
MODEL_CONFIG_DIR := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$MODEL_CONFIG_DIR')
F16_GGUF := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$F16_GGUF')
LLAMA_PERPLEXITY := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$LLAMA_PERPLEXITY')
LLAMA_QUANTIZE := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$LLAMA_QUANTIZE')
GGUF_ARCH_KEY := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$GGUF_ARCH_KEY')
LLAMACPP_DIR := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$LLAMACPP_DIR')
MODELS_DIR := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$MODELS_DIR')

help:
	@echo "Super-Quant: Advanced GGUF Quantization Pipeline"
	@echo ""
	@echo "Full pipeline:  make all"
	@echo ""
	@echo "Individual steps:"
	@echo "  make setup        Build llama.cpp + install Python deps"
	@echo "  make download     Download BF16 model from HuggingFace"
	@echo "  make convert      Convert to F16 GGUF (with MTP tensors)"
	@echo "  make calibrate    Build multi-domain calibration dataset"
	@echo "  make imatrix      Generate DI-MATRIX importance matrices"
	@echo "  make sensitivity  Per-tensor KLD analysis (optional, refines overrides)"
	@echo "  make quantize     Quantize to all target levels"
	@echo "  make benchmark    Run perplexity + throughput benchmarks"
	@echo "  make compare      Print comparison table"
	@echo "  make serve        Launch llama-server (DSpark/MTP/none)"
	@echo "  make quantize-nvfp4  AWQ+GPTQ NVFP4 quantization (llm-compressor)"
	@echo "  make convert-nvfp4   Convert NVFP4 checkpoint to GGUF"
	@echo "  make serve-sglang    Launch SGLang server (DFlash2, FP8 KV)"
	@echo "  make stop-sglang     Stop SGLang server"
	@echo "  make bench-sglang    Benchmark SGLang throughput"
	@echo ""
	@echo "Serving options:"
	@echo "  make serve                                          # Q6_K, DSpark, 5 slots"
	@echo "  make serve QUANT=UD-Q8_0 CTX=262144 PARALLEL=3     # Q8_0, native ctx, 3 slots"
	@echo "  make serve SPEC_TYPE=mtp                            # MTP instead of DSpark"
	@echo "  make serve SPEC_TYPE=none                           # No speculative decoding"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove .venv and build artifacts (keeps models)"

all: setup download convert calibrate imatrix quantize benchmark compare

setup:
	@bash $(SCRIPTS)/00_setup.sh

download:
	@$(UV) bash $(SCRIPTS)/01_download_model.sh

bench-spec:
	@$(UV) bash $(SCRIPTS)/07_bench_spec.sh

convert:
	@$(UV) bash $(SCRIPTS)/02_convert_to_gguf.sh

calibrate:
	@$(UV) python3 $(SRC)/prepare_calibration.py --output-dir $(PROJECT_DIR)/calibration --model-id $(MODEL_ID)

recalibrate:
	@$(UV) python3 $(SRC)/prepare_calibration.py --output-dir $(PROJECT_DIR)/calibration --model-id $(MODEL_ID) --force

imatrix:
	@$(UV) bash $(SCRIPTS)/03_generate_imatrix_gpu.sh

sensitivity:
	@$(UV) python3 $(SRC)/sensitivity_analysis.py \
		--model-id $(MODEL_ID) \
		--test-file $(PROJECT_DIR)/calibration/combined.txt \
		--gguf-arch-key $(GGUF_ARCH_KEY) \
		--llamacpp-dir $(LLAMACPP_DIR) \
		--output-json $(PROJECT_DIR)/results/sensitivity.json
	@$(UV) python3 $(SRC)/generate_hybrid_overrides.py \
		--model $(F16_GGUF) \
		--sensitivity $(PROJECT_DIR)/results/sensitivity.json \
		--output $(MODEL_CONFIG_DIR)/tensor_overrides.txt

quantize:
	@$(UV) bash $(SCRIPTS)/04_quantize.sh

quantize-nvfp4:
	@$(UV) python3 $(SRC)/quantize_nvfp4.py \
		--model-id $(MODEL_ID) \
		--calibration-dir $(PROJECT_DIR)/calibration \
		--output-dir $(HOME)/models/nvfp4/$(MODEL_NAME)-NVFP4

NVFP4_CHECKPOINT := $(HOME)/models/nvfp4/$(MODEL_NAME)-NVFP4
NVFP4_GGUF := $(MODELS_DIR)/$(MODEL_NAME)-NVFP4.gguf
CONVERT_SCRIPT := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$CONVERT_SCRIPT')

convert-nvfp4:
	@$(UV) python3 $(CONVERT_SCRIPT) $(NVFP4_CHECKPOINT) \
		--outfile $(NVFP4_GGUF) \
		--outtype auto \
		--fp8-as-q8

benchmark:
	@$(UV) bash $(SCRIPTS)/05_benchmark.sh

bench-concurrent:
	@$(UV) bash $(SCRIPTS)/08_bench_concurrent.sh

bench-kv:
	@$(UV) bash $(SCRIPTS)/10_bench_kv_types.sh

compare:
	@$(UV) python3 $(SRC)/compare_results.py --results-dir $(PROJECT_DIR)/results --model-name $(MODEL_NAME)

compare-md:
	@$(UV) python3 $(SRC)/compare_results.py --results-dir $(PROJECT_DIR)/results --model-name $(MODEL_NAME) --markdown

QUANT     ?= UD-Q6_K
CTX       ?= 524288
PORT      ?= 8000
PARALLEL  ?= 3
SPEC_TYPE ?= dspark
serve:
	@SPEC_TYPE=$(SPEC_TYPE) $(UV) bash $(SCRIPTS)/06_serve.sh $(QUANT) $(CTX) $(PORT) $(PARALLEL) $(KV_TYPE_K) $(KV_TYPE_V)

serve-sglang:
	@$(UV) bash $(SCRIPTS)/serve_sglang.sh

stop-sglang:
	@bash $(SCRIPTS)/stop_sglang.sh

serve-vllm:
	@bash $(SCRIPTS)/serve_vllm.sh

stop-vllm:
	@bash $(SCRIPTS)/stop_sglang.sh

bench-sglang:
	@$(UV) bash $(SCRIPTS)/11_bench_sglang.sh

MODELS_DIR := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$MODELS_DIR')

clean:
	rm -rf $(PROJECT_DIR)/.venv
	rm -f $(PROJECT_DIR)/calibration/*.dat
	rm -f $(PROJECT_DIR)/calibration/*.txt
	rm -rf $(PROJECT_DIR)/results/*
	@echo "Cleaned. Models preserved in $(MODELS_DIR)"
