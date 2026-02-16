# -*- makefile -*- mode for emacs
# @(#) $Id$

# Copyright (C) 2002 Iowa State University

# This file is part of JML

# JML is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.

# JML is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with JML; see the file COPYING.  If not, write to
# the Free Software Foundation, 675 Mass Ave, Cambridge, MA 02139, USA.

## TOPDIR = ../..

## INFO_DIR = /usr/unsup/info
## WWW_DIR = /home/bambam/leavens/WWW/JML
## FTP_DIR = /home/ftp/pub/leavens/JML
## ANONFTPDIR = /home/ftp/pub/techreports/TR98-06

## SAMPLESDIR = $(TOPDIR)/org/jmlspecs/samples
## THISDOWN = docs/prelimdesign

# Included JML files that are automatically generated
# from corresponding files in other directories

include ../../Makefile.defs

ROOT = Escjava2-ImplementationNotes

TEXINFOS = 

# Implicit Rules (for GNU Make)


# Target Definitions.
#
TEXIBASE = $(ROOT)

.PHONY: all clean html postscript pdf info texinfos publish ftp-docs
.PHONY: docs-for-ftp just-ftp-docs www clean-this cleanlocal clean-classes-this
.PHONY: distclean-this distclean-local cleanhtmldir

all : html postscript pdf

postscript : $(TEXIBASE).ps

$(TEXIBASE).ps  : $(TEXIBASE).dvi
	dvips -o $(TEXIBASE).ps $(TEXIBASE).dvi

dvi: $(TEXIBASE).dvi

$(TEXIBASE).dvi : $(TEXIBASE).texinfo $(TEXINFOS)
	rm -f $(TEXIBASE).aux
	$(TEXI2DVI) $(TEXIBASE).texinfo

pdf: $(TEXIBASE).pdf

$(TEXIBASE).pdf : $(TEXIBASE).texinfo $(TEXIBASE).ps
	if type -type $(TEXI2PDF) > /dev/null; then \
		$(TEXI2PDF) $(TEXIBASE).texinfo; \
	else \
		ps2pdf $(TEXIBASE).ps $(TEXIBASE).pdf; \
	fi

$(TEXIBASE).dvi.gz : $(TEXIBASE).dvi
	gzip -f --best $(TEXIBASE).dvi

$(TEXIBASE).ps.gz : $(TEXIBASE).ps
	gzip -f --best $(TEXIBASE).ps

info : $(TEXIBASE).info

$(TEXIBASE).info : $(TEXIBASE).texinfo $(TEXINFOS)
	$(MAKEINFO) $(TEXIBASE).texinfo

texinfos: $(TEXINFOS)


html: $(TEXIBASE)/$(TEXIBASE)_toc.html

$(TEXIBASE)/$(TEXIBASE)_toc.html: $(TEXIBASE).texinfo $(TEXINFOS) \
		adjust-html-file Makefile
	$(DELTREE) $(TEXIBASE)
	$(MKDIR) $(TEXIBASE)
	$(TEXI2HTML) $(TEXI2HTMLOPTS) $(TEXIBASE).texinfo
	for f in *.html ; \
	do \
		$(PERL) -p adjust-html-file $$f > $(TEXIBASE)/$$f ; \
	done
	#-$(CP) $(TEXIBASE)/$(TEXIBASE)_toc.html $(TEXIBASE)/index.html
	$(RM) $(TEXIBASE)*.html

publish: ftp-docs www

ftp-docs: docs-for-ftp just-ftp-docs

docs-for-ftp: $(TEXIBASE).ps.gz $(TEXIBASE).pdf html

just-ftp-docs:
	$(RM) $(FTP_DIR)/$(TEXIBASE).ps.gz
	$(RM) $(FTP_DIR)/$(TEXIBASE).pdf
	$(CP) $(TEXIBASE).ps.gz $(TEXIBASE).pdf $(FTP_DIR)
	chmod 664 $(FTP_DIR)/$(TEXIBASE).*.gz $(FTP_DIR)/*.pdf
	chgrp jml $(FTP_DIR)/$(TEXIBASE).*.gz $(FTP_DIR)/*.pdf
	$(CP) $(TEXIBASE).ps.gz $(ANONFTPDIR)/TR.ps.gz
	$(CP) $(TEXIBASE).pdf $(ANONFTPDIR)/TR.pdf
	$(CP) TR.abs README $(ANONFTPDIR)

www: html
	chmod ug+w $(TEXIBASE) $(TEXIBASE)/*.html
	chgrp jml $(TEXIBASE) $(TEXIBASE)/*.html
	chmod a+r $(TEXIBASE)/*.html
	chmod a+rx $(TEXIBASE)
	$(DELTREE) $(WWW_DIR)/$(TEXIBASE)
	cp -pr $(TEXIBASE) $(WWW_DIR)/

# Clean up stuff
#
clean: clean-this

clean-this: cleanlocal

cleanlocal:
	$(DELTREE) *.log *~

clean-classes-this: remove-dot-class-files

distclean-this: distclean-local

distclean-local: clean-this cleanhtmldir
	##$(RM) $(LOCAL)/*.texinfo

cleanhtmldir:
	$(DELTREE) $(TEXIBASE)

# include standard project targets

RM = rm -f
DELTREE = rm -rf
CP = cp -f
MKDIR = mkdir -p

TEXI2DVI = texi2dvi
TEXI2HTML = texi2html
TEXI2HTMLOPTS = -menu -number
DVIPS = dvips -P pdf -t letter
DVIPDF = dvipdf
PERL = perl
