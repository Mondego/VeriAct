# Copyright Hewlett-Packard, 2002

all: source
	$(JAVAC) -d ${CLASSDIRECTORY} *.java

full: source
	$(JAVAC) -d ${CLASSDIRECTORY} *.java

source:
	@echo "Nothing to build."

clean: 
	rm -f *~
	rm -rf *.class ${CLASSDIRECTORY}/escjava/translate/*.class

runtest:
	@echo "No tests to run."
