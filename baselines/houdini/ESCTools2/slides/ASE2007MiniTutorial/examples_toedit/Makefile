# $Id: Makefile 2764 2007-07-03 09:22:14Z leavens $

# A simple makefile for working with jmlunit

CLASSPATH='.;c:\cygwin\usr\local\junit\junit.jar'
# Flags below are for JML common tools release 5.4, looking to the future
JML = jml -Q -e
# JMLC = jmlc -Q -e -o
JMLC = jmlc -Q --universes --neutralContext
JMLRAC = jmlrac
JMLUNIT = jmlunit -i
JAVAC = javac
# Flags below are for ESC/Java2 release 2.0b1
ESCJAVA = escjava2 -nonNullByDefault

TYPES = Alias BagOfInt BagOfIntMain \
	BoundedStack ModularityDemo Ex

all: compile runtests

check checkall:
	$(JML) $(addsuffix .java,$(TYPES))

esc ESC: $(addsuffix .esc,$(TYPES))

compile: $(addsuffix .class,$(TYPES))

runtests: $(addsuffix _JML_TestData.class,$(TYPES)) \
          $(addsuffix _JML_Test.class,$(TYPES))
	$(JMLC) $(addsuffix .java,$(TYPES))
	for f in $(addsuffix _JML_Test,$(TYPES)); \
	do \
		$(JMLRAC) $f; \
	done

%.esc: %.java
	$(ESCJAVA) $< 

generate: $(addsuffix _JML_TestData.java,$(TYPES)) \
          $(addsuffix _JML_Test.java,$(TYPES))

%_JML_TestData.class: %_JML_TestData.java
	javac $<

%_JML_Test.class: %_JML_Test.java 
	javac $<

%.class: $(addsuffix .java,%)
	$(JMLC) $<

%_JML_TestData.java: %.java
	$(JMLUNIT) $<

%_JML_Test.java: %.java
	$(JMLUNIT) $<

clean: cleanclasses

cleanclasses:
	rm -f *.class
