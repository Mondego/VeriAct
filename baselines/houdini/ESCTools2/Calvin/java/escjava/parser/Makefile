# Copyright Hewlett-Packard, 2002

all: source
	javac -d ${CLASSDIRECTORY} *.java

full: source
	$(JAVAC) -d ${CLASSDIRECTORY} *.java

source:
	@echo "Nothing to build."

clean: 
	cd test; ${MAKE} clean
	rm -f *~
	rm -rf *.class ${CLASSDIRECTORY}/escjava/parser/*.class

runtest:
	cd test; ${MAKE} runtest
