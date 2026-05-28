.PHONY: all setup download convert calibrate recalibrate imatrix sensitivity quantize benchmark compare serve clean help

SHELL := /bin/bash
PROJECT_DIR := $(shell pwd)
SCRIPTS := $(PROJECT_DIR)/scripts
SRC := $(PROJECT_DIR)/src
UV := uv run --project $(PROJECT_DIR)

# Load model config from model.env
MODEL_ID := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$MODEL_ID')
MODEL_NAME := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$MODEL_NAME')
F16_GGUF := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$F16_GGUF')
LLAMA_PERPLEXITY := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$LLAMA_PERPLEXITY')
LLAMA_QUANTIZE := $(shell bash -c 'source $(PROJECT_DIR)/configs/model.env && echo $$LLAMA_QUANTIZE')

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
	@echo "  make serve        Launch llama-server with MTP"
	@echo ""
	@echo "Serving options:"
	@echo "  make serve QUANT=Q4_K_M CTX=65536 PORT=8080"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean        Remove .venv and build artifacts (keeps models)"

all: setup download convert calibrate imatrix quantize benchmark compare

setup:
	@bash $(SCRIPTS)/00_setup.sh

download:
	@$(UV) bash $(SCRIPTS)/01_download_model.sh

convert:
	@$(UV) bash $(SCRIPTS)/02_convert_to_gguf.sh

calibrate:
	@$(UV) python3 $(SRC)/prepare_calibration.py --output-dir $(PROJECT_DIR)/calibration --model-id $(MODEL_ID)

recalibrate:
	@$(UV) python3 $(SRC)/prepare_calibration.py --output-dir $(PROJECT_DIR)/calibration --model-id $(MODEL_ID) --force

imatrix:
	@bash $(SCRIPTS)/03_generate_imatrix.sh

sensitivity:
	@$(UV) python3 $(SRC)/sensitivity_analysis.py \
		--model $(F16_GGUF) \
		--test-file $(PROJECT_DIR)/calibration/combined.txt \
		--llama-perplexity $(LLAMA_PERPLEXITY) \
		--llama-quantize $(LLAMA_QUANTIZE) \
		--output $(PROJECT_DIR)/configs/tensor_overrides.txt \
		--output-json $(PROJECT_DIR)/results/sensitivity.json

quantize:
	@bash $(SCRIPTS)/04_quantize.sh

benchmark:
	@bash $(SCRIPTS)/05_benchmark.sh

compare:
	@$(UV) python3 $(SRC)/compare_results.py --results-dir $(PROJECT_DIR)/results --model-name $(MODEL_NAME)

compare-md:
	@$(UV) python3 $(SRC)/compare_results.py --results-dir $(PROJECT_DIR)/results --model-name $(MODEL_NAME) --markdown

QUANT ?= Q5_K_M
CTX ?= 32768
PORT ?= 8080
serve:
	@bash $(SCRIPTS)/06_serve.sh $(QUANT) $(CTX) $(PORT)

clean:
	rm -rf $(PROJECT_DIR)/.venv
	rm -f $(PROJECT_DIR)/calibration/*.dat
	rm -f $(PROJECT_DIR)/calibration/*.txt
	rm -rf $(PROJECT_DIR)/results/*
	@echo "Cleaned. Models preserved in $(PROJECT_DIR)/models/"
