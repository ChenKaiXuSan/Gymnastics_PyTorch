PYTHON ?= python
LATEXMK ?= latexmk
BUILD := build

.PHONY: all tables manuscript supplement check submission-check package

all: tables manuscript supplement check

tables:
	$(PYTHON) scripts/generate_comparison_tables.py --reuse-person-cache

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
	zip -q submission/sports_engineering_source.zip manuscript.tex online_resource_1.tex references.bib cohort_cycle_analysis.pdf sn-jnl.cls sn-mathphys-num.bst appendix.sty threeparttable.sty vruler.sty generated/*.tex generated/*.csv scripts/generate_comparison_tables.py
