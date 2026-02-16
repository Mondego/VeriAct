#
# $Id$
#

paper =		bringingjml

texfiles =	$(wildcard *.tex)

dvifiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.dvi,$(texfile)))

psfiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.ps,$(texfile)))

pdffiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.pdf,$(texfile)))

all:	${paper}.dvi ${paper}.ps ${paper}.pdf

%.pdf: %.tex
	pdflatex $< $@

%.pdf.gz: %.pdf
	gzip -c $< > $@

%.ps: %.dvi
	dvips -Pcmz -Pamz -f < $< > $@

%.ps.gz: %.ps
	gzip -c $< > $@

# ${paper}.dvi: ${paper}.tex
# 	latex ${paper}; latex ${paper}; bibtex ${paper}; \
# 	latex ${paper}; latex ${paper}; \

${paper}.dvi: ${paper}.tex
	latex ${paper}; latex ${paper};

cleanup:
	rm -f _region_* *.aux *.log *# *~

clean:
	rm -rf auto _region_* \
		*.aux *.log *~ *.dvi ${paper}.ps ${paper}.pdf *.brf *.txt \
		*.bbl *.blg *.toc *.thm *.out *.cb \
		*.lof *.lot *.gz *.Z #*
