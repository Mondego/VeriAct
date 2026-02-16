export TOP  = ../../../..
include ${TOP}/Makefile.defs

.PHONY: runtests check

runtests:  check

check:
	$(ESCJ) -quiet -classpath . *.java
