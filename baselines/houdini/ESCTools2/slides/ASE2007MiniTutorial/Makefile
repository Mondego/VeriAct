# $Id: Makefile 2771 2007-07-05 10:16:42Z leavens $

FILES = JMLTutorial
PSFILES = $(FILES:%=%Presentation.ps)
PDFFILES = $(FILES:%=%Presentation.pdf)
HANDOUTS = $(FILES:%=%Handouts.ps4.pdf)

LATEX=latex
PDFLATEX=pdflatex
# mlatex is the miktex version of latex
HLATEX=mlatex
HPDFLATEX=mpdflatex
DVIPS=dvips  
PSNUP=psnup
PS2PDF=ps2pdf

pdf: $(PDFFILES)

handouts: $(HANDOUTS)

both: pdf handouts

ps: $(PSFILES)

%.pdf: %.tex $(FILES:%=%.tex)
	$(PDFLATEX) $<
	$(PDFLATEX) $<

%.ps4.pdf: %.tex $(FILES:%=%.tex)
	$(HPDFLATEX) $<
	$(HPDFLATEX) $<
	mv $(subst .tex,.pdf,$<) $@

%.dvi: %.tex $(FILES:%=%.tex)
	$(LATEX) $<
	$(LATEX) $<

%.ps: %.dvi 
	$(DVIPS) -o $@ $<

%.ps4.ps: %.ps 
	$(PSNUP) -2 $< >$<4.ps

cleanall: clean
	rm -f JMLTutorial*.pdf

clean:
	rm -f *.dvi *.ps *.log *.out *.vrb *~
