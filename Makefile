SHELL := /bin/bash

init:
	git submodule update --init --recursive
	make setup

setup:
	virtualenv venv
	source venv/bin/activate;\
	pip install -e panda-gym;\
	pip install -e rlkit;\
	pip install -r requirements.txt
	
format:
	black . --exclude rlkit --line-length 104
	isort . --sg rlkit

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

.PHONY : test
test:
	source venv/bin/activate;\
	python test/test_env.py