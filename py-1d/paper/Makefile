# Build the paper PDF.
#
# Requires pdflatex. Runs two passes to resolve references.

PAPER = paper
TEX = $(PAPER).tex
PDF = $(PAPER).pdf
LATEX = pdflatex -interaction=nonstopmode -halt-on-error

.PHONY: all clean

all: $(PDF)

$(PDF): $(TEX) $(wildcard figs/*.png)
	$(LATEX) $(TEX)
	$(LATEX) $(TEX)

clean:
	rm -f $(PAPER).aux $(PAPER).log $(PAPER).out $(PAPER).toc
	rm -f $(PAPER).fls $(PAPER).fdb_latexmk $(PAPER).synctex.gz
