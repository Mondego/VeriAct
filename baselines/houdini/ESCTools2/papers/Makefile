#
# $Id$
#

paper =		

talk =		

texfiles =	$(wildcard *.tex)

dvifiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.dvi,$(texfile)))

psfiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.ps,$(texfile)))

pdffiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.pdf,$(texfile)))

javafiles =		$(wildcard problems/*.java)
javatexinfofiles =	$(addsuffix .texinfo,$(javafiles))

PROBLEMS =	problems
PAPERS =	CASSIS2004.pdf FTFJP2004.pdf PASTE2004.pdf NIII-R0413.pdf
DIAGRAMS =	koa_state_diagram.eps koa_state_diagram.pdf
HOMEWORK =	jml_homework.dvi jml_homework.ps jml_homework.pdf jml_homework.txt

PERL=perl
CODE2TEXI=$(PERL) -e 'BEGIN: while (<>) { s/\r//g; s/@/@@/g; s/\{/@\{/g; s/\}/@\}/g; } continue { print; }'

.PHONY: all build view
all:	build

build:	$(PAPERS) $(HOMEWORK)

view:	$(HOMEWORK)
	open -a Skim.app jml_homework.pdf

$(PROBLEMS)/%.java.texinfo : $(PROBLEMS)/%.java
	@$(CODE2TEXI) $< > $@

%.pdf: %.aux %.bbl
	export BIBINPUTS="~/Sandboxes/Writing/Bibliography//:"; \
	export BSTINPUTS="~/LaTeX//:"; \
	export TEXINPUTS="~/LaTeX//:"; \
	pdflatex $* && pdflatex $*;

%.dvi: %.aux %.bbl
	export BIBINPUTS="~/Sandboxes/Writing/Bibliography//:"; \
	export BSTINPUTS="~/LaTeX//:"; \
	export TEXINPUTS="~/LaTeX//:"; \
	latex $* && latex $*;

%.aux %.log: %.tex
	export BIBINPUTS="~/Sandboxes/Writing/Bibliography//:"; \
	export BSTINPUTS="~/LaTeX//:"; \
	export TEXINPUTS="~/LaTeX//:"; \
	latex $*;

%.bbl %.blg: %.aux
	export BIBINPUTS="~/Sandboxes/Writing/Bibliography//:"; \
	export BSTINPUTS="~/LaTeX//:"; \
	export TEXINPUTS="~/LaTeX//:"; \
	bibtex $*;

%.ps2pdf: %.ps
	ps2pdf $<

%.pdf.gz: %.pdf
	gzip -c $< > $@

%.ps: %.dvi
	dvips -Pcmz -Pamz -f < $< > $@

%.ps.gz: %.ps
	gzip -c $< > $@

# texinfo

.svnversion:
	svnversion > .svnversion

%.info:	%.texinfo .svnversion
	makeinfo $<

%.txt:	%.texinfo .svnversion
	makeinfo --no-headers $< -o $@

%.html:	%.texinfo .svnversion
	makeinfo --html	$<

%.dvi:	%.texinfo .svnversion
	texi2dvi $<

%.pdf:	%.texinfo .svnversion
	texi2pdf $<

jml_homework.dvi:	jml_homework.texinfo $(javatexinfofiles)

PASTE2004.pdf:	PASTE2004.tex
	pdflatex PASTE2004; pdflatex PASTE2004;
	make cleanup

CASSIS2004.pdf:	CASSIS2004.tex koa_state_diagram.pdf
	(unset BSTINPUTS; \
	pdflatex CASSIS2004; bibtex CASSIS2004; pdflatex CASSIS2004; pdflatex CASSIS2004;)
	make cleanup

CASSIS2004.dvi:	CASSIS2004.tex koa_state_diagram.eps
	(unset BSTINPUTS; \
	latex CASSIS2004; bibtex CASSIS2004; latex CASSIS2004; latex CASSIS2004;)

NIII-R0413.pdf:	NIII-R0413.tex koa_state_diagram.pdf
	(unset BSTINPUTS; \
	pdflatex NIII-R0413; bibtex NIII-R0413; pdflatex NIII-R0413; pdflatex NIII-R0413;)
	make cleanup

cleanup:
	rm -f _region_* *.aux *.log \
		*.cp *.cps *.fn *.ky *.pg *.tp *.vr \
		*# *~

clean:
	rm -rf auto _region_* \
		$(PAPERS) $(HOMEWORK) \
		*.aux *.log *~ *.dvi *.brf \
		*.cp *.cps *.fn *.ky *.pg *.tp *.vr \
		*.bbl *.blg *.toc *.thm *.out *.cb \
		*.lof *.lot *.gz *.Z #*
	rm -rf jml_homework
	rm -f problems/*.texinfo
