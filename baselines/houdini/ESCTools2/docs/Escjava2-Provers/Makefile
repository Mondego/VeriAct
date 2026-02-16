#
# $Id$
#

include ../../Makefile.defs

paper =		Escjava2-Provers

talk =		

texfiles =	$(wildcard *.tex)

texifiles =	$(wildcard *.texinfo)

dvifiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.dvi,$(texfile)))

psfiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.ps,$(texfile)))

pdffiles =	$(foreach texfile,$(texfiles),\
		$(subst .tex,.pdf,$(texfile)))

java_files =		$(wildcard *.java)
java_texinfo_files =	$(addsuffix .texinfo,$(java_files))

OSX_CFLAGS = -I/System/Library/Frameworks/JavaVM.framework/Versions/1.4.2/Headers
OSX_LFLAGS = -framework JavaVM
LINUX_CFLAGS = -I$(JAVA_HOME)/include -I$(JAVA_HOME)/include/linux
LINUX_LFLAGS = 

CC = gcc
LIBTOOL = libtool

#OSX_DYNAMIC_LIB_SWITCH = -dynamiclib
#LINUX_DYNAMIC_LIB_SWITCH = -shared

OSX_DYNAMIC_LIB_EXTENSION = dylib
LINUX_DYNAMIC_LIB_EXTENSION = so

#OSX_DYNAMIC_LIB_PREFIX = lib
#LINUX_DYNAMIC_LIB_PREFIX = lib

os = $(shell uname -s)
ifeq ("${os}","Darwin")
	LIBTOOL = glibtool
	CFLAGS = $(OSX_CFLAGS)
	LFLAGS = $(OSX_LFLAGS)
	#DYNAMIC_LIB_SWITCH = $(OSX_DYNAMIC_LIB_SWITCH)
	DYNAMIC_LIB_EXTENSION = $(OSX_DYNAMIC_LIB_EXTENSION)
	#DYNAMIC_LIB_PREFIX = $(OSX_DYNAMIC_LIB_PREFIX)
else
	CFLAGS = $(LINUX_CFLAGS)
	LFLAGS = $(LINUX_LFLAGS)
	#DYNAMIC_LIB_SWITCH = $(LINUX_DYNAMIC_LIB_SWITCH)
	DYNAMIC_LIB_EXTENSION = $(LINUX_DYNAMIC_LIB_EXTENSION)
	#DYNAMIC_LIB_PREFIX = $(LINUX_DYNAMIC_LIB_PREFIX)
endif

PERL=perl
CODE2TEXI=$(PERL) -e 'BEGIN: while (<>) { s/\r//g; s/@/@@/g; s/\{/@\{/g; s/\}/@\}/g; } continue { print; }'

%.java.texinfo: %.java
	$(CODE2TEXI) $< > $@

%.pdf:	%.texinfo
	${TEXI2PDF} $<
	texindex `basename $< .texinfo`.??
	${TEXI2PDF} $<

%.info:	%.texinfo
	makeinfo $<

%.txt:	%.texinfo
	makeinfo --no-headers $< -o $@

%.html:	%.texinfo
	makeinfo --html	$<

%.dvi:	%.texinfo
	texi2dvi $<
	texindex `basename $< .texinfo`.??
	texi2dvi $<

%.eps: %.fig
	fig2dev -L eps $< $@

%.pdf: %.fig
	fig2dev -L pdf $< $@

%.class: %.java
	javac -source 1.4 $<

%.h:	%.class
	javah `basename $< .class`

.PHONY: all build cleanup interface java jni test clean

all:	java jni interface build

build:	Escjava2-Provers.pdf

Escjava2-Provers.pdf:	Escjava2-Provers.texinfo $(java_texinfo_files)

interface : ProverResponse.class Signature.class ProverInterface.class Formula.class

java:	ProverInterface.class DemonstrationProver.class

jni:	DemonstrationProver.h libdemoprover.la demoprover

test:	java jni

demoprover:	DemonstrationProver.c
	$(CC) $(CFLAGS) \
	  -o demoprover $<

demoprover.o:	DemonstrationProver.c
	$(LIBTOOL) --mode=compile $(CC) $(CFLAGS) -c \
	  -o demoprover.o $<

libdemoprover.la:	demoprover.o
	$(LIBTOOL) --mode=link $(CC) $(CFLAGS) \
		-o libdemoprover.la -rpath $(PWD) demoprover.lo $(LFLAGS) \
		&& mv -f .libs/*.$(DYNAMIC_LIB_EXTENSION) .

cleanup:
	rm -f _region_* *.aux *.log \
		*.cp *.cps *.fn *.fns *.ky *.kys \
		*.pg *.pgs *.tp *.tps *.vr *.vrs *.p *.k \
		*# *~

clean:
	rm -rf auto _region_* \
	  $(paper).pdf $(paper).ps \
	  demoprover \
	  *.class \
	  *.o *.la *.lo *.so *.dylib .libs \
	  *.java.texinfo \
	  *.aux *.log *~ *.dvi *.brf \
	  *.cp *.cps *.fn *.fns *.ky *.kys \
	  *.pg *.pgs *.tp *.tps *.vr *.vrs *.p *.k \
	  *.bbl *.blg *.toc *.thm *.out *.cb \
	  *.lof *.lot *.gz *.Z #*
