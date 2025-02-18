SHELL          = /bin/bash
PROJECT_NAME   = jrystal
PROJECT_FOLDER = $(PROJECT_NAME) third_party
PYTHON_FILES   = $(shell find . -type f -name "*.py" -not -path '*/.venv/*')
COMMIT_HASH    = $(shell git log -1 --format=%h)
COPYRIGHT      = "Garena Online Private Limited"
DATE           = $(shell date "+%Y-%m-%d")
PATH           := $(HOME)/go/bin:$(PATH)

# installation

check_install = python3 -c "import $(1)" || (cd && pip3 install $(1) --upgrade && cd -)
check_install_extra = python3 -c "import $(1)" || (cd && pip3 install $(2) --upgrade && cd -)

flake8-install:
	$(call check_install, flake8)
	$(call check_install_extra, bugbear, flake8_bugbear)

py-format-install:
	$(call check_install, isort)
	$(call check_install, yapf)

mypy-install:
	$(call check_install, mypy)

addlicense-install: go-install
	command -v addlicense || go install github.com/google/addlicense@latest

doc-install:
	$(call check_install, pydocstyle)
	$(call check_install_extra, doc8, "doc8<1")
	$(call check_install, sphinx)
	$(call check_install, sphinx_book_theme)
	$(call check_install, readthedocs-sphinx-search)
	$(call check_install, myst-parser)
	$(call check_install_extra, sphinxcontrib.spelling, sphinxcontrib.spelling pyenchant)

auditwheel-install:
	$(call check_install_extra, auditwheel, auditwheel typed-ast patchelf)

# python linter. -fix version applies the fixes

flake8: flake8-install
	flake8 $(PYTHON_FILES) --count --show-source --statistics

flake8-fix: flake8-install
	flake8 $(PYTHON_FILES)

py-format: py-format-install
	isort --check $(PYTHON_FILES) && yapf -r -d $(PYTHON_FILES)

py-format-fix: py-format-install
	isort $(PYTHON_FILES) && yapf -ir $(PYTHON_FILES)

mypy: mypy-install
	mypy $(PROJECT_NAME)

# documentation

addlicense: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache -check $(PROJECT_FOLDER)

addlicense-fix: addlicense-install
	addlicense -c $(COPYRIGHT) -l apache $(PROJECT_FOLDER)

docstyle: doc-install
	pydocstyle $(PROJECT_NAME) && doc8 docs && cd docs && make html SPHINXOPTS="-W"

doc: doc-install
	cd docs && make html && cd _build/html && python3 -m http.server

spelling: doc-install
	cd docs && make spelling SPHINXOPTS="-W"

doc-clean:
	cd docs && make clean

lint: buildifier flake8 py-format # docstyle

format: py-format-install buildifier-install addlicense-install py-format-fix clang-format-fix buildifier-fix addlicense-fix

pypi-wheel: auditwheel-install bazel-release
	ls dist/*.whl -Art | tail -n 1 | xargs auditwheel repair --plat manylinux_2_24_x86_64
