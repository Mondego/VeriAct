# Copyright 2000, 2001, Compaq Computer Corporation

all: build man1/jls.1 man1/jls.html

build:
	${JAVAC} -d $(call canonicalize,${CLASSDIRECTORY}) *.java

clean: 
	rm -f *.class
	rm -f man1/*.1 man1/*.html

.SUFFIXES: .class .java .1 .mtex .html

.mtex.1:
	mtex2man < $*.mtex > $*.1

.mtex.html:
	mtex2html < $*.mtex > $*.html
