# help:
# 	@echo "Downloads the ProcTHOR environment related data:"
# 	@echo "  download-procthor-10k			Download procthor-10k data."
# 	@echo "  download-ai2thor-simulator		Download ai2thor simulator."
# 	@echo "  download-sbert-model			Download SBERT model."
# 	@echo "  download-procthor-all			Download both procthor-10k data and ai2thor simulator."


PROCTHOR_DATA_DIR ?= procthor-10k
AI2THOR_SIM_DIR ?= ai2thor
SBERT_MODEL_DIR ?= sentence_transformers
RESOURCES_BASE_DIR ?= ./resources
DOCKER_PYTHON ?= uv run python

.PHONY: download-procthor-10k-data
download-procthor-10k-data: $(RESOURCES_BASE_DIR)/$(PROCTHOR_DATA_DIR)/download_complete.tmp
$(RESOURCES_BASE_DIR)/$(PROCTHOR_DATA_DIR)/download_complete.tmp:
	@mkdir -p $(RESOURCES_BASE_DIR)/$(PROCTHOR_DATA_DIR)
	@$(DOCKER_PYTHON) -m procthor.scripts.download_data \
		--save_dir /resources/$(PROCTHOR_DATA_DIR)
	@touch $(RESOURCES_BASE_DIR)/$(PROCTHOR_DATA_DIR)/download_complete.tmp

.PHONY: download-ai2thor-simulator
download-ai2thor-simulator: $(RESOURCES_BASE_DIR)/$(AI2THOR_SIM_DIR)/download_complete.tmp
$(RESOURCES_BASE_DIR)/$(AI2THOR_SIM_DIR)/download_complete.tmp:
	@mkdir -p $(RESOURCES_BASE_DIR)/$(AI2THOR_SIM_DIR)
	@$(DOCKER_PYTHON) -m procthor.scripts.download_simulator
	@touch $(RESOURCES_BASE_DIR)/$(AI2THOR_SIM_DIR)/download_complete.tmp

.PHONY: download-sbert-model
download-sbert-model: $(RESOURCES_BASE_DIR)/$(SBERT_MODEL_DIR)/model.safetensors
$(RESOURCES_BASE_DIR)/$(SBERT_MODEL_DIR)/model.safetensors:
	@mkdir -p $(RESOURCES_BASE_DIR)/$(SBERT_MODEL_DIR)
	@$(DOCKER_PYTHON) -m procthor.scripts.download_sbert \
		--save_dir /resources/$(SBERT_MODEL_DIR)

.PHONY: download-procthor-all
download-procthor-all: download-procthor-10k-data download-ai2thor-simulator download-sbert-model
