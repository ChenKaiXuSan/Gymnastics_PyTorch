TEMPLATE_DIR := $(abspath template/els-cas-templates)
TEMPLATE_ARCHIVE := $(abspath template/els-cas-templates.zip)
TEMPLATE_CLASS := $(TEMPLATE_DIR)/cas-sc.cls
export TEXINPUTS := $(TEMPLATE_DIR):
export BSTINPUTS := $(TEMPLATE_DIR):
export BIBINPUTS := $(CURDIR):
export MPLCONFIGDIR := $(CURDIR)/.mplconfig
export XDG_CACHE_HOME := $(CURDIR)/.cache

.PHONY: all assets check pdf clean template

all: assets check pdf

template: $(TEMPLATE_CLASS)

$(TEMPLATE_CLASS): $(TEMPLATE_ARCHIVE)
	unzip -q -o $(TEMPLATE_ARCHIVE) -d template

assets:
	conda run -n gymnastic python scripts/generate_paper_assets.py

check:
	conda run -n gymnastic python scripts/check_manuscript.py

pdf: template
	mkdir -p build
	latexmk -pdf -bibtex -interaction=nonstopmode -halt-on-error -outdir=build manuscript.tex

clean:
	latexmk -C -outdir=build manuscript.tex
