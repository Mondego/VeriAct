# Copyright Hewlett-Packard, 2002

class:	source
	@echo !!!!! ${JAVAC} -d ${CLASSDIRECTORY} Root.java
	${JAVAC} -d ${CLASSDIRECTORY} Root.java

full:	source
	cd ast; ${MAKE} full
	cd backpred; ${MAKE} full
	cd parser; ${MAKE} full
	cd prover; ${MAKE} full
	cd tc; ${MAKE} full
	cd translate; ${MAKE} full
	${JAVAC} -d ${CLASSDIRECTORY} *.java

source:
	cd ast; ${MAKE} source
	cd backpred; ${MAKE} source
	cd parser; ${MAKE} source
	cd prover; ${MAKE} source
	cd tc; ${MAKE} source
	cd translate; ${MAKE} source

clean:
	cd ast; ${MAKE} clean
	cd backpred; ${MAKE} clean
	cd parser; ${MAKE} clean
	cd prover; ${MAKE} clean
	cd tc; ${MAKE} clean
	cd test; ${MAKE} clean
	cd translate; ${MAKE} clean
	rm -f *~
	rm -rf *.class ${CLASSDIRECTORY}/escjava/*.class

runtest:
	cd ast; ${MAKE} runtest
	cd backpred; ${MAKE} runtest
	cd parser; ${MAKE} runtest
	cd prover; ${MAKE} runtest
	cd test; ${MAKE} runtest
	cd tc; ${MAKE} runtest
	cd translate; ${MAKE} runtest

doc:	escjava.mtex
	cc -C -E escjava.mtex | grep -v "^#" | mtex2man \
		> ${ESCJAVA_ROOT}/doc/man1/escjava.1
	cc -C -E escjava.mtex | grep -v "^#" | mtex2html \
		> ${ESCJAVA_ROOT}/doc/escjava.html
	cc -C -E -DPUBLIC escjava.mtex | grep -v "^#" | mtex2man \
		> ${ESCJAVA_ROOT}/doc/man1/escjava-public.1
	cc -C -E -DPUBLIC escjava.mtex | grep -v "^#" | mtex2html \
		> ${ESCJAVA_ROOT}/doc/escjava-public.html


# Counting size of source

JAVASOURCEFILES = ast/*.java parser/*.java prover/*.java tc/*.java translate/*.java
OTHERSOURCEFILES = 
TESTHARNESSFILES = 

wc:
	# --- Non-java Source files ---
	wc ${OTHERSOURCEFILES}
	# --- Java Source files ---
	wc ${JAVASOURCEFILES}
	jrun javafe.parser.test.TestParse print ${JAVASOURCEFILES} | wc
	# --- Test harness files ---
	wc ${TESTHARNESSFILES}
	jrun javafe.parser.test.TestParse print ${TESTHARNESSFILES} | wc
