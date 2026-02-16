# $Id$
#
# Copyright 2006, DSRG.org
#

export TOP = ../..
include ${TOP}/Makefile.defs

SUBDIRS = jdktests	\
	  junittests

clean:  clean_escjava
	for d in $(SUBDIRS) ; do \
	    $(MAKE) -C $$d $@ || exit 1; \
	done

clean_escjava:
	$(MAKE) -C escjava/test clean

# End of Makefile
