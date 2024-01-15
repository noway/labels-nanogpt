.PHONY: deploy.zip all

include .env

all: scp_to_lambda

deploy.zip:
	zip -r deploy.zip train.py train_simple.py train_all.sh checkpoint.sh tokens-*.json labels-*.json

scp_to_lambda: deploy.zip
	scp deploy.zip $(USER)@$(IP_ADDR):$(FILESYSTEM_PATH)
	ssh $(USER)@$(IP_ADDR) "cd $(FILESYSTEM_PATH); unzip -o deploy.zip; rm deploy.zip"