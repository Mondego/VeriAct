# $Id$
# Modified 2007, Mobius Project, University College Dublin

export TOP = ../../..
include ${TOP}/Makefile.defs

############################################################################
# ESCJAVA is (only) used for standalone tests
ESCJAVA = ${ESCJ_ROOT}/escj 

_ECLASSPATH = $(call canonicalize,${ESC_CLASSPATH}:${JUNIT_UTILS_PATH}:.${EXTRA_CP})
_CLASSPATH = $(call canonicalize,:${JUNIT_UTILS_PATH}:.${EXTRA_CP})
_TMP = $(call canonicalize,/tmp)
_SIMP = $(call canonicalize,${ESCJ_SIMPLIFY})

_SPECS = $(call canonicalize,${ESC_SPECS})
_ESCROOT = $(call canonicalize,${ESCTOOLS_ROOT})

# Used to invoke text-based JUnit tests
export JUNIT = ${JAVA} -Xmx256M -classpath "${_CLASSPATH}" \
	-Djava.io.tmpdir=${_TMP} -Dsimplify=${_SIMP} \
	junit.textui.TestRunner

# Used to invoke GUI-based JUnit tests
export JUNITSW = $(JAVA) -Xmx256M -classpath "${_CLASSPATH}" \
	-Djava.io.tmpdir=${_TMP} -Dsimplify=${_SIMP} \
	junit.swingui.TestRunner

##############################################################################

## The root targets to run the tests
.PHONY: test runtests
test runtests: extests skipped separate_escjava_tests

## Builds the lists that drive the tests as well as the JUnit tests themselves
.PHONY: buildtests
buildtests: lists
	touch testfile  ## Required by TestSystem1.java
	$(JAVAC) -classpath "${_ECLASSPATH}" -source 1.4 `cat listfiles` \
		AllTests.java Suite.java LocalTestCase.java

## This runs the tests as Junit tests.  AllTests runs everything that is
## listed in listfiles.  These tests actually execute the Test*.java files.
.PHONY: extests
extests: buildtests
	-${JUNIT} AllTests

## This runs all the JUnit tests in a Junit GUI
.PHONY: swtests
swtests: buildtests
	-$(JUNITSW) AllTests

DEFAULTFLAGS = " "
ALWAYSFLAGS = -quiet -nowarn Deadlock -specs "${_SPECS}" -classpath "${_CLASSPATH}"

CHUNK = 40


## This target runs all the escjava tests, but in batches to avoid resource
## overloads
#### FIXME - this target is not working
.PHONY: part_escjava_tests
part_escjava_tests: lists
	@echo "TESTING escjava on jdktests (in parts)"
	cp listtests listteststemp
	cp listteststemp templist
	while test -s templist ; do \
	    head -n ${CHUNK} templist > listtests ;\
	    ${JUNIT} Suite ;\
	    tail +${CHUNK} templist | tail +2 > templista ;\
	    mv templista templist; \
	done 
	rm -f templist templista
	mv listteststemp listtests


## This target runs all the escjava tests, but individually
.PHONY: separate_escjava_tests
separate_escjava_tests: lists
	@echo "TESTING escjava on jdktests (individually)"
	-@for f in `cat listfiles ` ; do \
		$(MAKE) --no-print-directory $${f}-test; \
	done

## This target runs a single test, both as a JUnit test and as a escjava test
%.test:
	@echo Testing $*
	$(MAKE) $*.ex
	$(MAKE) $*-test

## This target runs a single test as a JUnit test
%.ex:
	$(JAVAC) -classpath "${_CLASSPATH}" $*.java
	-$(JUNIT) $*

## This target (X-test) runs escjava against X.java, using any flags specified
## in X.java and comparing the output against the contents of X.java-expected.
## The actual output is stored in X.java-ckd if it is different.
%-test: .sedScript
	@echo Testing $* 
	@if [ ! -e $* ] ; then \
		echo No such file $* ;\
	else \
	(${ESCJAVA} ${ALWAYSFLAGS} \
		-classpath "${_CLASSPATH}" \
		-specs "${_SPECS}"\
		`(grep '//#FLAGS:' $* || echo ${DEFAULTFLAGS}) | \
		sed -e "sx//#FLAGS:xx" -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" ` \
		$* 2>&1 \
		| tr '\\' '/'  \
		| sed -f .sedScript \
	        > $*-ckd ) || exit 0; \
	if [ -e $*-expected ] ; then \
		${DIFF} $*-ckd $*-expected  \
		  && rm $*-ckd ;\
	else \
		echo ... Creating $*-expected ;\
		mv $*-ckd $*-expected ;\
	fi ;\
	fi ;

## This target (X-run) runs escjava against X.java, using any flags specified
## in X.java; the output is simply output, and not stored in -ckd or 
## compared against -expected.  This is useful for debugging.
%-run:
	@echo Running $*  \
		${ALWAYSFLAGS} -classpath "${_CLASSPATH}" -specs "${_SPECS}"\
		`grep '//#FLAGS:' $* | \
		sed -e "sx//#FLAGS:xx" -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" ` \
		${ARGS} 
	@if [ ! -e $* ] ; then \
		echo No such file $* ;\
	else \
	${ESCJAVA} ${ALWAYSFLAGS} -classpath "${_CLASSPATH}" -specs "${_SPECS}"\
		`grep '//#FLAGS:' $* | \
		sed -e "sx//#FLAGS:xx" -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" ` \
		${ARGS} $* ; \
	fi


## Generates the -expected file for the test.  
## HAND INSPECT IT TO BE SURE IT IS CORRECT!!!!
%-expected: %
	escjava ${ALWAYSFLAGS} \
		`grep '//#FLAGS:' $* | sed -e "sx//#FLAGS:xx"` $* 2>&1 > $@

## Builds the lists that drive the tests
.PHONY: lists
lists: Test*.java skip
	@-rm -f listtests listfiles
	@touch listtests
	@touch skip
	@(ls Test*.java | grep -v -f skip > listfiles)
ifdef COMSPEC
	@for f in `cat listfiles `; do \
	        echo -n ${ALWAYSFLAGS} " " >> listtests ; \
	        ( grep FLAGS $$f || echo ${DEFAULTFLAGS} ) \
	        | sed -e 's?//#FLAGS:??' -e "s/\$$/ $$f/"  \
	        | sed -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" \
	        | sed -e "s@.:..@.;..@" \
	        >> listtests ; \
	done
else
	@for f in `cat listfiles `; do \
	        echo -n ${ALWAYSFLAGS} " " >> listtests ; \
	        ( grep FLAGS $$f || echo ${DEFAULTFLAGS} ) \
	        | sed -e 's?//#FLAGS:??' -e "s/\$$/ $$f/"  \
		| sed -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" \
	        >> listtests ; \
	done
endif

## Lists the skipped tests
.PHONY: skipped
skipped:
	@echo SKIPPING: `(echo Test*.java | grep -f skip )`

.sedScript:
	@echo "s@${_ESCROOT}@ESCTOOLS_ROOT@" | tr '\\' '/' > .sedScript

## Cleans all temporary files
.PHONY: clean
clean:
	@rm -f core* .sedScript

# End of file
