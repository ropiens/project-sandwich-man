SHELL := /bin/bash

init:
	git submodule update --init --recursive
	make setup

setup:
	virtualenv venv
	source venv/bin/activate;\
	python3 -m pip install panda-gym;\
	python3 -m pip install -r requirements.txt
	
format:
	black . --exclude external --line-length 104
	isort . --sg external

clean:
	rm -rf venv
	find -iname "*.pyc" -delete