.PHONY: all setup setup-sglang setup-vllm download convert calibrate-corpus recalibrate-corpus calibrate recalibrate package-calibration-corpus imatrix sensitivity quantize quantize-nvfp4 convert-nvfp4 bench-llamacpp bench-sglang compare compare-md serve serve-llama stop serve-vllm stop-vllm awq-prescale awq-convert awq-imatrix awq-sensitivity awq-quantize awq-all test clean help

SHELL := /bin/bash
PROJECT_DIR := $(shell pwd)
SCRIPTS := $(PROJECT_DIR)/scripts
SRC := $(PROJECT_DIR)/src
UV := uv run --project $(PROJECT_DIR)

# Every target reads the active model through configs/model.env. Override it with
# MODEL_DIR=<model> on the make command; keep model-specific choices out of here.
REQUESTED_MODEL_DIR := $(MODEL_DIR)
MODEL_CONFIG_OK := $(shell MODEL_DIR='$(REQUESTED_MODEL_DIR)' bash -c 'source "$(PROJECT_DIR)/configs/model.env" >/dev/null && printf yes')
ifneq ($(MODEL_CONFIG_OK),yes)
$(error Failed to load MODEL_DIR=$(REQUESTED_MODEL_DIR))
endif
model_var = $(shell MODEL_DIR='$(REQUESTED_MODEL_DIR)' bash -c 'source "$(PROJECT_DIR)/configs/model.env" && printf "%s" "$${!1}"' _ $(1))
MODEL_DIR := $(call model_var,MODEL_DIR)
MODEL_ID := $(call model_var,MODEL_ID)
MODEL_NAME := $(call model_var,MODEL_NAME)
CALIBRATION_MODEL_ID := $(call model_var,CALIBRATION_MODEL_ID)
MODEL_CONFIG_DIR := $(call model_var,MODEL_CONFIG_DIR)
QUANT_CONFIG := $(MODEL_CONFIG_DIR)/quantize.json
F16_GGUF := $(call model_var,F16_GGUF)
LLAMA_PERPLEXITY := $(call model_var,LLAMA_PERPLEXITY)
LLAMA_QUANTIZE := $(call model_var,LLAMA_QUANTIZE)
GGUF_ARCH_KEY := $(call model_var,GGUF_ARCH_KEY)
LLAMACPP_DIR := $(call model_var,LLAMACPP_DIR)
MODELS_DIR := $(call model_var,MODELS_DIR)
NATIVE_CTX := $(call model_var,NATIVE_CTX)
AWQ_CHECKPOINT := $(call model_var,AWQ_CHECKPOINT)
AWQ_F16_GGUF := $(call model_var,AWQ_F16_GGUF)
AWQ_CALIBRATION_DIR := $(call model_var,AWQ_CALIBRATION_DIR)
AWQ_IMATRIX_MERGED := $(call model_var,AWQ_IMATRIX_MERGED)
AWQ_TENSOR_OVERRIDES := $(call model_var,AWQ_TENSOR_OVERRIDES)
AWQ_SENSITIVITY := $(call model_var,AWQ_SENSITIVITY)
CALIBRATION_CORPUS_DIR := $(call model_var,CALIBRATION_CORPUS_DIR)
CALIBRATION_DIR := $(call model_var,CALIBRATION_DIR)
CALIBRATION_DOMAINS := $(call model_var,CALIBRATION_DOMAINS)
IMATRIX_WEIGHTS := $(call model_var,IMATRIX_WEIGHTS)
CORPUS_ARCHIVE ?= $(PROJECT_DIR)/calibration-corpus-v1.tar.gz
IMATRIX_CONTEXT_SIZE := $(call model_var,IMATRIX_CONTEXT_SIZE)
QUANT_TYPES_VAR := $(call model_var,QUANT_TYPES)

help:
	@echo "Super-Quant: Config-Driven Quantization and Serving"
	@echo ""
	@echo "All targets use MODEL_DIR=<model>. The configured default is Qwen3.8-Flash-Next."
	@echo ""
	@echo "Full GGUF pipeline:"
	@echo "  MODEL_DIR=<model> make all"
	@echo "  Note: make all uses existing tensor overrides; run sensitivity after model or policy changes."
	@echo ""
	@echo "Pipeline stages:"
	@echo "  make download        Download the active model to the Hugging Face cache"
	@echo "  make convert         Convert the active model to source-precision GGUF"
	@echo "  make calibrate-corpus Build the shared tokenizer-neutral source corpus"
	@echo "  make recalibrate-corpus Rebuild the shared source corpus from pinned datasets"
	@echo "  make calibrate       Build model-specific calibration from the shared corpus"
	@echo "  make recalibrate     Rebuild model-specific calibration from the shared corpus"
	@echo "  make package-calibration-corpus Build the deterministic corpus release archive"
	@echo "  make imatrix         Generate and merge per-domain importance matrices"
	@echo "  make sensitivity     Measure tensor sensitivity and regenerate overrides"
	@echo "  make quantize        Build configured GGUF quantization types"
	@echo "  make bench-llamacpp  Run GGUF quality and llama.cpp throughput benchmarks"
	@echo "  make compare         Print the benchmark comparison"
	@echo "  make compare-md      Print the comparison as Markdown"
	@echo ""
	@echo "AWQ and native NVFP4 stages (requires: uv sync --extra quantize):"
	@echo "  make awq-all         Run AWQ pre-scaling through AWQ-UD GGUF quantization"
	@echo "  make awq-prescale    Save a plain BF16 checkpoint with AWQ channel scaling"
	@echo "  make awq-convert     Convert the AWQ checkpoint to GGUF"
	@echo "  make awq-imatrix     Generate an importance matrix in the AWQ channel basis"
	@echo "  make awq-sensitivity Regenerate AWQ-specific tensor overrides"
	@echo "  make awq-quantize    Build AWQ-UD GGUF quantization types"
	@echo "  make quantize-nvfp4  Build an AWQ+GPTQ NVFP4 checkpoint"
	@echo "  make convert-nvfp4   Convert a native NVFP4 checkpoint to GGUF"
	@echo ""
	@echo "Serving:"
	@echo "  make serve           Start SGLang with the active model config"
	@echo "  make serve-vllm      Start vLLM with the active model config"
	@echo "  make serve-llama     Start llama.cpp with a configured GGUF"
	@echo "  make stop            Stop SGLang"
	@echo "  make stop-vllm       Stop vLLM"
	@echo "  Examples: make serve-llama SPEC_TYPE=none"
	@echo "            make serve-llama SPEC_TYPE=mtp QUANT=UD-Q6_K CTX=262144 PARALLEL=3"
	@echo ""
	@echo "Benchmarks:"
	@echo "  make bench-llamacpp  GGUF quality and llama.cpp throughput"
	@echo "  make bench-sglang    SGLang performance and HiCache correctness"
	@echo "  Optional: BENCH_PHASES=performance or BENCH_PHASES=correctness"
	@echo "  Request options: uv run python scripts/bench_sglang_request.py --help"
	@echo ""
	@echo "Environment and maintenance:"
	@echo "  make setup           Validate CUDA/llama.cpp and sync project dependencies"
	@echo "  make setup-sglang    Rebuild the pinned SGLang environment"
	@echo "  make setup-vllm      Rebuild the pinned vLLM environment"
	@echo "  make test            Run unit tests"
	@echo "  make clean           Remove project artifacts; preserve generated models"

# Sensitivity is intentionally excluded: established model configs reuse their
# reviewed override file. Run `make sensitivity` when the model or policy changes.
all: setup download convert calibrate imatrix quantize bench-llamacpp compare

setup:
	@bash $(SCRIPTS)/00_setup.sh

setup-sglang:
	@bash $(SCRIPTS)/setup_sglang_env.sh

setup-vllm:
	@bash $(SCRIPTS)/setup_vllm_env.sh

download:
	@$(UV) bash $(SCRIPTS)/01_download_model.sh

convert:
	@$(UV) bash $(SCRIPTS)/02_convert_to_gguf.sh

calibrate-corpus:
	@$(UV) python3 $(SRC)/prepare_calibration.py \
		--corpus-dir $(CALIBRATION_CORPUS_DIR) \
		--corpus-only

recalibrate-corpus:
	@$(UV) python3 $(SRC)/prepare_calibration.py \
		--corpus-dir $(CALIBRATION_CORPUS_DIR) \
		--corpus-only \
		--force-corpus

calibrate:
	@$(UV) python3 $(SRC)/prepare_calibration.py \
		--corpus-dir $(CALIBRATION_CORPUS_DIR) \
		--output-dir $(CALIBRATION_DIR) \
		--model-id $(CALIBRATION_MODEL_ID)

recalibrate:
	@$(UV) python3 $(SRC)/prepare_calibration.py \
		--corpus-dir $(CALIBRATION_CORPUS_DIR) \
		--output-dir $(CALIBRATION_DIR) \
		--model-id $(CALIBRATION_MODEL_ID) \
		--force

package-calibration-corpus:
	@$(UV) python3 $(SCRIPTS)/package_calibration_corpus.py \
		$(CALIBRATION_CORPUS_DIR) \
		$(CORPUS_ARCHIVE)

imatrix:
	@$(UV) bash $(SCRIPTS)/03_generate_imatrix_gpu.sh

sensitivity:
	@$(UV) python3 $(SRC)/sensitivity_analysis.py \
		--model-id $(MODEL_ID) \
		--test-file $(CALIBRATION_DIR)/holdout.txt \
		--gguf-arch-key $(GGUF_ARCH_KEY) \
		--llamacpp-dir $(LLAMACPP_DIR) \
		--output-json $(PROJECT_DIR)/results/sensitivity.json
	@$(UV) python3 $(SRC)/generate_hybrid_overrides.py \
		--model $(F16_GGUF) \
		--sensitivity $(PROJECT_DIR)/results/sensitivity.json \
		--output $(MODEL_CONFIG_DIR)/tensor_overrides.txt

quantize:
	@$(UV) bash $(SCRIPTS)/04_quantize.sh

# Requires the quantize extra, a per-model quantize.json, and enough memory.
quantize-nvfp4:
	@test -f "$(QUANT_CONFIG)" || { echo "ERROR: quantization recipe not found: $(QUANT_CONFIG)" >&2; exit 1; }
	@$(UV) python3 $(SRC)/quantize_nvfp4.py \
		--config $(QUANT_CONFIG) \
		--calibration-dir $(CALIBRATION_DIR) \
		--output-dir $(HOME)/models/nvfp4/$(MODEL_NAME)-NVFP4

NVFP4_CHECKPOINT := $(HOME)/models/nvfp4/$(MODEL_NAME)-NVFP4
NVFP4_GGUF := $(MODELS_DIR)/$(MODEL_NAME)-NVFP4.gguf
CONVERT_SCRIPT := $(call model_var,CONVERT_SCRIPT)

convert-nvfp4:
	@$(UV) python3 $(CONVERT_SCRIPT) $(NVFP4_CHECKPOINT) \
		--outfile $(NVFP4_GGUF) \
		--outtype auto \
		--fp8-as-q8

bench-llamacpp:
	@$(UV) bash $(SCRIPTS)/05_benchmark.sh

compare:
	@$(UV) python3 $(SRC)/compare_results.py --results-dir $(PROJECT_DIR)/results --model-name $(MODEL_NAME)

compare-md:
	@$(UV) python3 $(SRC)/compare_results.py --results-dir $(PROJECT_DIR)/results --model-name $(MODEL_NAME) --markdown

serve:
	@MODEL_DIR=$(MODEL_DIR) $(UV) bash $(SCRIPTS)/serve_sglang.sh

stop:
	@bash $(SCRIPTS)/stop_sglang.sh

QUANT     ?= UD-Q6_K
CTX       ?= $(NATIVE_CTX)
PORT      ?= 8000
PARALLEL  ?= 3
# Empty means use LLAMA_SPEC_TYPE_DEFAULT from the active model config.
SPEC_TYPE ?=
serve-llama:
	@MODEL_DIR=$(MODEL_DIR) SPEC_TYPE=$(SPEC_TYPE) $(UV) bash $(SCRIPTS)/06_serve.sh $(QUANT) $(CTX) $(PORT) $(PARALLEL) $(KV_TYPE_K) $(KV_TYPE_V)


serve-vllm:
	@bash $(SCRIPTS)/serve_vllm.sh

stop-vllm:
	@bash $(SCRIPTS)/stop_vllm.sh

bench-sglang:
	@MODEL_DIR=$(MODEL_DIR) $(UV) bash $(SCRIPTS)/11_bench_sglang.sh

test:
	@$(UV) --extra quantize python -m unittest discover -s tests -v

# --- AWQ pre-scaled pipeline ---
# These targets require `uv sync --extra quantize` and enough host/GPU memory
# for the selected model. Resource limits belong to the host, not MODEL_DIR.
awq-prescale:
	@test -f "$(QUANT_CONFIG)" || { echo "ERROR: quantization recipe not found: $(QUANT_CONFIG)" >&2; exit 1; }
	@$(UV) python3 $(SRC)/awq_prescale.py \
		--config $(QUANT_CONFIG) \
		--calibration-dir $(CALIBRATION_DIR) \
		--output-dir $(AWQ_CHECKPOINT)

awq-convert:
	@$(UV) python3 $(CONVERT_SCRIPT) $(AWQ_CHECKPOINT) \
		--outfile $(AWQ_F16_GGUF) \
		--outtype f16

awq-imatrix:
	@mkdir -p $(AWQ_CALIBRATION_DIR)
	@$(UV) python3 $(SRC)/generate_imatrix.py \
		--model-id $(AWQ_CHECKPOINT) \
		--calibration-dir $(CALIBRATION_DIR) \
		--domains $(CALIBRATION_DOMAINS) \
		--output-dir $(AWQ_CALIBRATION_DIR) \
		--gguf-arch-key $(GGUF_ARCH_KEY) \
		--llamacpp-dir $(LLAMACPP_DIR) \
		--context-size $(IMATRIX_CONTEXT_SIZE) \
		--dtype bfloat16 \
		--force
	@$(UV) python3 $(SRC)/merge_imatrix.py \
		$(foreach d,$(CALIBRATION_DOMAINS),$(AWQ_CALIBRATION_DIR)/imatrix_$(d).dat) \
		--weights $(IMATRIX_WEIGHTS) \
		-o $(AWQ_IMATRIX_MERGED)

awq-sensitivity:
	@$(UV) python3 $(SRC)/sensitivity_analysis.py \
		--model-id $(AWQ_CHECKPOINT) \
		--test-file $(CALIBRATION_DIR)/holdout.txt \
		--gguf-arch-key $(GGUF_ARCH_KEY) \
		--llamacpp-dir $(LLAMACPP_DIR) \
		--output-json $(AWQ_SENSITIVITY)
	@$(UV) python3 $(SRC)/generate_hybrid_overrides.py \
		--model $(AWQ_F16_GGUF) \
		--sensitivity $(AWQ_SENSITIVITY) \
		--output $(AWQ_TENSOR_OVERRIDES)

awq-quantize:
	@bash -c '\
		IMATRIX="$(AWQ_IMATRIX_MERGED)" && \
		OVERRIDES_CLEAN=$$(mktemp) && \
		grep -v "^\s*#" "$(AWQ_TENSOR_OVERRIDES)" | grep -v "^\s*$$" > "$$OVERRIDES_CLEAN" && \
		for TYPE in $(QUANT_TYPES_VAR); do \
			OUTPUT="$(MODELS_DIR)/$(MODEL_NAME)-AWQ-UD-$$TYPE.gguf" && \
			if [ -f "$$OUTPUT" ]; then echo "Skipping AWQ-UD-$$TYPE -- already exists"; continue; fi && \
			echo "--- AWQ-UD-$$TYPE ---" && \
			$(LLAMA_QUANTIZE) \
				--imatrix "$$IMATRIX" \
				--tensor-type-file "$$OVERRIDES_CLEAN" \
				"$(AWQ_F16_GGUF)" "$$OUTPUT" "$$TYPE" && \
			echo "  -> $$OUTPUT ($$(du -sh "$$OUTPUT" | cut -f1))" ; \
		done && \
		rm -f "$$OVERRIDES_CLEAN" \
	'

awq-all: awq-prescale awq-convert awq-imatrix awq-sensitivity awq-quantize

clean:
	rm -rf $(PROJECT_DIR)/.venv
	rm -rf $(CALIBRATION_DIR)
	rm -rf $(PROJECT_DIR)/results/*
	@echo "Cleaned. Models preserved in $(MODELS_DIR)"
