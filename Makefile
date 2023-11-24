.PHONY: deploy.zip all

include .env

all: scp_to_lambda

deploy.zip:
	zip -r deploy.zip input.txt train.py

scp_to_lambda: deploy.zip
	scp deploy.zip ubuntu@$(IP_ADDR):$(FILESYSTEM_PATH)
	ssh ubuntu@$(IP_ADDR) "cd $(FILESYSTEM_PATH); unzip -o deploy.zip; rm deploy.zip"