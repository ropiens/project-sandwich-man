SHELL := /bin/bash

init:
	git submodule update --init --recursive
	make setup

setup:
	virtualenv venv
	source venv/bin/activate;\
	python3 -m pip install -e panda-gym;\
	python3 -m pip install -r requirements.txt
	python setup.py install
	
format:
	black . --exclude venv --line-length 128
	isort . --sg venv

clean:
	rm -rf venv
	find -iname "*.pyc" -delete

.PHONY : test
test:
	source venv/bin/activate;\
	python test/test_env.py