.PHONY: setup download calibrate-corpus recalibrate-corpus calibrate recalibrate package-calibration-corpus quantize-nvfp4 serve-infernix stop-infernix infernix-placement infernix-gaming infernix-full test clean help

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
CALIBRATION_CORPUS_DIR := $(call model_var,CALIBRATION_CORPUS_DIR)
CALIBRATION_DIR := $(call model_var,CALIBRATION_DIR)
CORPUS_ARCHIVE ?= $(PROJECT_DIR)/calibration-corpus-v1.tar.gz

help:
	@echo "Super-Quant: Config-Driven Quantization and Serving"
	@echo ""
	@echo "All targets use MODEL_DIR=<model>. The configured default is Qwen3.8-Flash-Next."
	@echo ""
	@echo "Pipeline stages:"
	@echo "  make download        Download the active model to the Hugging Face cache"
	@echo "  make calibrate-corpus Build the shared tokenizer-neutral source corpus"
	@echo "  make recalibrate-corpus Rebuild the shared source corpus from pinned datasets"
	@echo "  make calibrate       Build model-specific calibration from the shared corpus"
	@echo "  make recalibrate     Rebuild model-specific calibration from the shared corpus"
	@echo "  make package-calibration-corpus Build the deterministic corpus release archive"
	@echo ""
	@echo "Native NVFP4 quantization (requires: uv sync --extra quantize):"
	@echo "  make quantize-nvfp4  Build an AWQ+GPTQ NVFP4 checkpoint"
	@echo ""
	@echo "Serving:"
	@echo "  make serve-infernix  Start Infernix (background; survives the shell)"
	@echo "  make stop-infernix   Stop Infernix"
	@echo "  make infernix-placement RESERVE_VRAM_GB=32 RESERVE_RAM_GB=24"
	@echo "  make infernix-gaming  Switch Infernix to the gaming profile (reserves in model.env)"
	@echo "  make infernix-full    Switch Infernix to every expert in VRAM"
	@echo ""
	@echo "Environment and maintenance:"
	@echo "  make setup           Validate CUDA and sync project dependencies"
	@echo "  make test            Run unit tests"
	@echo "  make clean           Remove project artifacts; preserve generated models"

setup:
	@bash $(SCRIPTS)/00_setup.sh

download:
	@$(UV) bash $(SCRIPTS)/01_download_model.sh

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

# Requires the quantize extra, a per-model quantize.json, and enough memory.
quantize-nvfp4:
	@test -f "$(QUANT_CONFIG)" || { echo "ERROR: quantization recipe not found: $(QUANT_CONFIG)" >&2; exit 1; }
	@$(UV) python3 $(SRC)/quantize_nvfp4.py \
		--config $(QUANT_CONFIG) \
		--calibration-dir $(CALIBRATION_DIR) \
		--output-dir $(HOME)/models/nvfp4/$(MODEL_NAME)-NVFP4

serve-infernix:
	@MODEL_DIR=$(MODEL_DIR) bash $(SCRIPTS)/serve_infernix.sh

stop-infernix:
	@MODEL_DIR=$(MODEL_DIR) bash $(SCRIPTS)/stop_infernix.sh

infernix-placement:
	@MODEL_DIR=$(MODEL_DIR) bash -c 'source configs/model.env && python3 $(SCRIPTS)/infernix_placement.py --reserve-vram-gb "$${RESERVE_VRAM_GB:-$$INFERNIX_GAMING_RESERVE_VRAM_GB}" --reserve-ram-gb "$${RESERVE_RAM_GB:-$$INFERNIX_GAMING_RESERVE_RAM_GB}"'

infernix-gaming:
	@MODEL_DIR=$(MODEL_DIR) bash -c 'source configs/model.env && python3 $(SCRIPTS)/infernix_placement.py --reserve-vram-gb "$$INFERNIX_GAMING_RESERVE_VRAM_GB" --reserve-ram-gb "$$INFERNIX_GAMING_RESERVE_RAM_GB"'

infernix-full:
	@python3 $(SCRIPTS)/infernix_placement.py --shares vram=all

test:
	@$(UV) --extra quantize python -m unittest discover -s tests -v

clean:
	rm -rf $(PROJECT_DIR)/.venv
	rm -rf $(CALIBRATION_DIR)
	rm -rf $(PROJECT_DIR)/results/*
	@echo "Cleaned. Quantized models and corpus releases outside the repository are preserved"
