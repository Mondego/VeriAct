# Copyright Hewlett-Packard, 2002

all: runtest

class:
	@echo "No .class files need to be built"

source:
	@echo "No .java files need to be built"

runtest: 
	rtestall

clean: 
	-rm -rf test*/out test*/diff
	find . -name \*~ -exec rm {} \;
	find . -name \*.class -exec rm {} \;
	find . -name core -exec rm {} \;

