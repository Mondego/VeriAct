#
# $Id$
#

export TOP = ..
include ../Makefile.defs

texfiles =	$(wildcard *.tex)

dvifiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.dvi,$(texfile)))

psfiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.ps,$(texfile)))

pdffiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.pdf,$(texfile)))

SLIDES =	escjava_demo.pdf introjml.pdf

all:	build

build:	$(SLIDES)
	export TEXINPUTS=$(ESCTOOLS_ROOT)/slides//:$$(TEXINPUTS) && \
	make -C ETAPSTutorial all && \
	make -C ECOOPTutorial-cok all

# latex

%.pdf: %.ps
	ps2pdf $<

%.pdf.gz: %.pdf
	gzip -c $< > $@

%.ps: %.dvi
	dvips -Pcmz -Pamz -f < $< > $@

%.ps.gz: %.ps
	gzip -c $< > $@

%.dvi: %.tex
	latex $* && latex $*;

# texinfo

%.info:	%.texinfo
	makeinfo $<

%.txt:	%.texinfo
	makeinfo --no-headers $< -o $@

%.html:	%.texinfo
	makeinfo --html	$<

%.dvi:	%.texinfo
	texi2dvi $<

# cleanup

cleanup:
	rm -f _region_* *.aux *.log *# *~

clean:
	make -C ETAPSTutorial clean;
	make -C ECOOPTutorial-cok clean;
#	$(MAKE) -C fmco05 clean;
	rm -rf auto _region_* \
		$(SLIDES) \
		*.aux *.log *~ *.dvi *.ps *.pdf *.brf *.txt \
		*.bbl *.blg *.toc *.thm *.out *.cb \
		*.lof *.lot *.gz *.Z #*

# End of Makefile
