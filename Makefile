SHELL := /bin/bash

init:
	git submodule update --init --recursive
	make setup

setup:
	virtualenv venv
	source venv/bin/activate;\
	python3 -m pip install -e panda-gym;\
	python3 -m pip install -r requirements.txt
	
format:
	black . --exclude external --line-length 104
	isort . --sg external

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

.PHONY : test
test:
	source venv/bin/activate;\
	python test/test_env.py