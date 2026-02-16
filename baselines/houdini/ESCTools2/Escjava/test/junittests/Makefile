# $Id$

# Modified 2007, Mobius Project, Systems Research Group, University College Dublin

TOP = ../../..
include ${TOP}/Makefile.defs

############################################################################
# ESCJAVA is (only) used for standalone tests
ESCJAVA = ${ESCJ_ROOT}/escj 

_CLASSPATH = $(call canonicalize,${ESC_CLASSPATH}:${JUNIT_UTILS_PATH}:.${EXTRA_CP})
_TMP = $(call canonicalize,/tmp)
_SIMP = $(call canonicalize,${ESCJ_SIMPLIFY})
_SPECS = $(call canonicalize,.:${ESC_SPECS})

# Used to invoke text-based JUnit tests
export JUNIT = ${JAVA} -Xmx256M -classpath "${_CLASSPATH}" \
	-Djava.io.tmpdir=${_TMP} -Dsimplify=${_SIMP} junit.textui.TestRunner

# Used to invoke GUI-based JUnit tests
export JUNITSW = $(JAVA) -Xmx256M -classpath "${_CLASSPATH}" \
	-Djava.io.tmpdir=${_TMP} -Dsimplify=${_SIMP} junit.swingui.TestRunner

##############################################################################
.PHONY: buildtests
buildtests:
	$(MAKE)  lists
	$(JAVAC) -classpath "${_CLASSPATH}" TestSuite.java
	cp Binary.javax Binary.java
	javac Binary.java
	rm Binary.java

.PHONY: test runtests
test runtests: buildtests
	$(MAKE) skips
	$(MAKE) dotests

.PHONY: dotests
dotests: 
	-$(MAKE) nonjunittests
	-$(MAKE) partjunittests

.PHONY: junittests
junittests:
	@echo "TESTING Escjava: junittests"
	$(JUNIT) TestSuite

.PHONY: quiet-junittests
quiet-junittests:
	@$(JUNIT) TestSuite

.PHONY: swtests
swtests: listtests
	$(JUNITSW) TestSuite

ALWAYSFLAGS = -nowarn Deadlock -verboseTrace -testMode -classpath "${_SPECS}"

DEFAULTFLAGS = " "

## This lists files without .java suffixes that need to be added to the
## set of junittests
OTHERTESTS = \
	RefineT.t SpecFile.jml RefineNoJava.refines-java \
	Binary.refines-spec NoPublic.refines-spec

## These have output that depends on initialization (e.g. what has already 
## happened in the previous junittests), so the output is more stable when
## run separately
NONJUNITTESTS = 
ifdef PDTESTS
## These are platform-dependent tests or tests whose output is a bit random
NONJUNITTESTS += $(shell cat notjunittests | grep -v -f skip)
endif

.PHONY: lists 
lists: listtests

listtests:  ${wildcard *.java} ${OTHERTESTS} skip notjunittests
	@echo Recreating test list in junittests
	@rm -f listtests
	@touch listtests
ifdef COMSPEC
	@for f in `ls *.java ${OTHERTESTS} | grep -v -f skip -f notjunittests`; do \
		echo -n ${ALWAYSFLAGS} " " >> listtests ; \
		( grep '//#FLAGS:' $$f || echo ${DEFAULTFLAGS} )  \
		| sed -e 's?//#FLAGS:??' -e "s/\$$/ $$f/"  \
		| sed -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" \
		| sed -e "s@.:..@.;..@" \
		>> listtests ; \
	done
else
	@for f in `ls *.java ${OTHERTESTS} | grep -v -f skip -f notjunittests`; do \
		echo -n ${ALWAYSFLAGS} " " >> listtests ; \
		( grep '//#FLAGS:' $$f || echo ${DEFAULTFLAGS} )  | tr -d "\r" \
		| sed -e 's?//#FLAGS:??' -e "s/\$$/ $$f/"  \
		| sed -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" \
		>> listtests ; \
	done
endif


.PHONY: separate_tests
separate_tests:
	$(MAKE) skips
	$(MAKE) othertests
	$(MAKE) separate_junit_tests
	$(MAKE) nonjunittests


.PHONY: nonjunittests
nonjunittests: notjunittests skip
	@echo "TESTING Escjava: nonjunittests"
	@-for f in ${NONJUNITTESTS} ; do \
		$(MAKE) --no-print-directory $${f}-test; \
	done


## Runs the tests listed in OTHERTESTS, as individual tests
.PHONY: othertests
othertests: $(addsuffix -test,${OTHERTESTS})


## Runs the JUnit tests n one fell swoop, using the JUnit infrastructure
## This can exhaust resources and get slower and slower, so we don't use it
.PHONY: alljunittests
alljunittests: listtests
	@echo "TESTING Escjava: junittests (all at once)"
	${MAKE} --no-print-directory quiet-junittests


CHUNK = 40

## Runs the JUnit tests as groups, using the JUnit infrastructure
.PHONY: partjunittests
partjunittests: listtests
	@echo "TESTING Escjava: junittests (in parts)"
	@cp listtests listteststemp
	@cp listteststemp templist
	@while test -s templist ; do \
		head -n ${CHUNK} templist > listtests ;\
		${MAKE} --no-print-directory quiet-junittests ;\
		tail +${CHUNK} templist | tail +2 > templista ;\
		mv templista templist; \
	done 
	@rm -f templist templista
	@mv listteststemp listtests

## Runs the junit tests as individual tests, not using the JUnit infrastructure
.PHONY: separate_junit_tests
separate_junit_tests: skip
	@for f in `ls *.java | grep -v -f skip -f nonjunittests` ; do \
		$(MAKE) --no-print-directory $${f}-test; \
	done

## Prints out the tests that are skipped, to remind the reader to fix them
.PHONY: skips
skips: skip
	@echo Skipped: `cat skip | grep -v "^#" `

## Creates an empty skip file, if needed.  Many targets presume one exists.
skip:
	@touch skip

## Runs an individual test, e.g. make X.java-test runs the X.java test, just
## as it is run as part of the test suite
%-test: 
	@echo Testing $* 
	@if [ ! -e $* ] ; then \
		echo No such file $* ;\
	else \
	(${ESCJAVA} ${ALWAYSFLAGS} \
		`(grep '//#FLAGS:' $* || echo ${DEFAULTFLAGS}) | tr -d "\r\n" | \
		sed -e "sx//#FLAGS:xx" -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" ` \
		$* 2>&1 | sed -e s/x/x/ > $*-ckd ) || exit 0; \
	if [ -e $*-expected ] ; then \
		${DIFF} $*-ckd $*-expected  \
		  && rm $*-ckd ;\
	else \
		echo ... Creating $*-expected ;\
		mv $*-ckd $*-expected ;\
	fi ;\
	fi ;\

## Runs an individual test, simply outputing the result of the test
%-run: 
	@echo Running $*  \
		${ALWAYSFLAGS} \
		`(grep '//#FLAGS:' $* || echo ${DEFAULTFLAGS}) | tr -d "\r\n" | \
		sed -e "sx//#FLAGS:xx" -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" `\
		${ARGS} 
	if [ ! -e $* ] ; then \
		echo No such file $* ;\
	else \
	${ESCJAVA} ${ALWAYSFLAGS} \
		`(grep '//#FLAGS:' $* || echo ${DEFAULTFLAGS}) | tr -d "\r\n" | \
		sed -e "sx//#FLAGS:xx" -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" ` \
		${ARGS} $* ; \
	fi 


## Runs an individual test, putting the result into the -expected file
%-expected: %
	${ESCJAVA} ${ALWAYSFLAGS} \
		`(grep '//#FLAGS:' $* || echo ${DEFAULTFLAGS}) | tr -d "\r\n" | \
		sed -e "sx//#FLAGS:xx" -e "s@ESCTOOLS_ROOT@${ESCTOOLS_ROOT}@" ` \
		  $* 2>&1 > $@

## Deletes all temporary files
.PHONY: clean
clean:
	@rm -f core*
	@rm -f listtests
	@touch listtests

# End of file
