PYTHON ?= python
LATEXMK ?= latexmk
BUILD := build

.PHONY: all manuscript supplement check submission-check package

all: manuscript supplement check

manuscript:
	$(LATEXMK) -pdf -bibtex -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) manuscript.tex

supplement:
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error -outdir=$(BUILD) online_resource_1.tex

check:
	$(PYTHON) scripts/check_sports_engineering.py

submission-check:
	$(PYTHON) scripts/check_sports_engineering.py --submission-ready

package: all
	mkdir -p submission
	cp $(BUILD)/manuscript.pdf submission/manuscript.pdf
	cp $(BUILD)/online_resource_1.pdf submission/online_resource_1.pdf
	cp cover_letter.md submission/cover_letter.md
	zip -j -q submission/sports_engineering_source.zip manuscript.tex online_resource_1.tex references.bib cohort_cycle_analysis.pdf sn-jnl.cls sn-mathphys-num.bst appendix.sty threeparttable.sty vruler.sty
