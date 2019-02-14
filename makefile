clean:
	rm -rf build
	rm -f deepnovo_cython_modules.c
	rm -f deepnovo_cython_modules*.so

pull:
	git pull

.PHONY: build
build: clean
	python deepnovo_cython_setup.py build_ext --inplace

.PHONY: train
train: pull
	python main.py --train

.PHONY: denovo
denovo:
	python main.py --search_denovo

.PHONY: test
denovo_test:
	python main.py --test
