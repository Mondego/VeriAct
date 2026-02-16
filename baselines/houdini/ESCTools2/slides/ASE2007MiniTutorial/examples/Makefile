# $Id: Makefile 2763 2007-07-03 09:19:55Z leavens $

# A simple makefile for working with jmlunit

CLASSPATH='.;c:\cygwin\usr\local\junit\junit.jar;c:\cygwin\usr\local\JML\bin\jml-release.jar'
# Flags below are for JML common tools release 5.5_rc3, looking to the future
JML = jml -Q
# JMLC = jmlc -Q -o
JMLC = jmlc -Q
JMLRAC = jmlrac
JMLUNIT = jmlunit -i
JAVAC = javac
# Flags below are for ESC/Java2 release 2.0b3
ESCJAVA = escjava2 -nonNullByDefault

TYPES = ArrayOps Interval IntervalRemove \
	BagOfInt BagOfIntMain \
        Actor ActorUnderspecified Aging \
	BoundedStack Counter Counter2 \
	Gendered GenderedEqualsBad GenderedEqualsGood \
	Animal Patient FemalePatient Females \
        Person Staff Doctor Nurse HeadTest \
	ScreenPoint ScreenPoint2 Hailstone Hailstone2 \
	Ex Factor SortedInts SortedInts2 SortedInts3 \
	ModularityDemo

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
