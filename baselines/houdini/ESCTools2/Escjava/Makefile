# $Id$
# Copyright 2000, 2001, Compaq Computer Corporation
# Copyright 2003 University of Nijmegen
# Copyright 2003 David R. Cok
# This file is extensively revised in the 2003 ESC revision.
# Modified 2007, Mobius Project, Systems Research Group, 
#                University College Dublin, Ireland

export TOP = ..
include ${TOP}/Makefile.defs

# Makefile.defs and the parent Makefile provide definitions of the 
# following variables needed here:
# JAVAC, JAVAC_FLAGS, ESCJAVA_ROOT, 
# ESCJAVA_BUILD_CLASSPATH, ESCJAVA_CLASSFILES, 
# (Building Escjava relies on the AST building tools)

JAVA_VM_FLAGS ?=-Xmx512m

# Various conventional UNIX tools are used in the following, including
# cc in preprocess only mode.  Also, the "astgen" and "astfilelist"
# programs from Javafe are used here.

###########################################################################
# Defining relevant directory locations and other variables.

ESCTOOLS_ROOT_REL = ..
ESCTOOLS_UTILS_ROOT_REL = $(ESCTOOLS_ROOT_REL)/Utils
ESCJAVA_SOURCE_DIR = ${ESCJAVA_ROOT}/java
ESCJAVA_SOURCE_DIR_REL = java


# The following definitions all use =, not :=, so that the wildcard functions
# are evaluated at the time of use, rather than at the time of definition.
# Some .java files do not exist at the time the Makefile is read, but are
# created in the course of executing the Makefile.
ESCJAVA_SOURCE_FILES_RELATIVE = \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/ant/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/ast/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/dfa/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/dfa/cfd/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/dfa/buildcfd/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/dfa/daganalysis/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/backpred/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/gui/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/parser/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/prover/jniw/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/prover/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/reader/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sp/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/tc/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/translate/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/coq/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/coq/visitor/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/cvc3/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/pvs/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/sammy/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/simplify/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/vcGeneration/xml/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sortedProver/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sortedProver/simplify/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sortedProver/z3/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sortedProver/fx7/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sortedProver/cvc3/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sortedProver/auflia/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/sortedProver/fx7pg/*.java) \
	$(wildcard $(ESCJAVA_SOURCE_DIR_REL)/escjava/*.java)

ESCJAVA_SOURCE_FILES = \
	$(foreach javafile,$(ESCJAVA_SOURCE_FILES_RELATIVE),\
	$(subst escjava,$(ESCJAVA_SOURCE_DIR)/escjava,$(javafile)))

#	${ESCJAVA_SOURCE_DIR}/escjava/Version.java

OTHER_SOURCE_FILES_RELATIVE = \
	$(wildcard $(ESCTOOLS_UTILS_ROOT_REL)/junitutils/*.java)

OTHER_SOURCE_FILES = \
	$(foreach javafile,$(OTHER_SOURCE_FILES_RELATIVE),\
	$(subst Utils,$(ESCTOOLS_ROOT)/Utils,$(javafile)))

# The executable to run escjava
export ESCJ = ${ESCJ_ROOT}/escj

## These definitions are used in subsidiary makes, hence are exported
export CLASSDIRECTORY=${ESCJAVA_CLASSFILES}
export SOURCEDIRECTORY=${ESCJAVA_SOURCE_DIR}
export TESTSOURCEDIRECTORY=${ESCJAVA_ROOT}/test

###########################################################################
.PHONY: all build buildall test alltests

all: clean build docs test

build: version compile

buildall: version compileall

test: test_escjava

alltests: test_escjava 
	-$(MAKE) self_test

###########################################################################
# Documentation targets.

# mtex2man and mtex2html are not generally available so we turn this
# off for now - FIXME David
# mtex2man is a Modula-III tool, thus if one is building Simplify, one
# should have it as well.  We'll just conditionalize this rule on the
# existence of the binary. -Joe

.PHONY: docs docs_escjava

docs: docs_escjava

ESCJAVA_CORE_DOC = ${ESCJAVA_SOURCE_DIR}/escjava/escjava.mtex

docs_escjava:	${ESCJAVA_CORE_DOC}
	if type -type mtex2man > /dev/null && type -type mtex2html > /dev/null; then \
		cc -C -E ${ESCJAVA_CORE_DOC} | grep -v "^#" | mtex2man \
			> ${ESCJAVA_ROOT}/doc/man1/escjava.1; \
		cc -C -E ${ESCJAVA_CORE_DOC} | grep -v "^#" | mtex2html \
			> ${ESCJAVA_ROOT}/doc/escjava.html; \
		cc -C -E -DPUBLIC ${ESCJAVA_CORE_DOC} | grep -v "^#" | mtex2man \
			> ${ESCJAVA_ROOT}/doc/man1/escjava-public.1; \
		cc -C -E -DPUBLIC ${ESCJAVA_CORE_DOC} | grep -v "^#" | mtex2html \
			> ${ESCJAVA_ROOT}/doc/escjava-public.html; \
	fi


###########################################################################
# Compilation targets.

STAMPFILE = classes.stamp

## These targets must be in the given order.  We use classes.stamp
## instead of allclasses here to enable incremental builds.  This
## incremental compilation is not robust - removing individual .class
## files will cause not everything to be built.  Also sometimes .java
## files ought to be recompiled if dependencies change, even if the
## file has not been modified.  If in doubt do a clean before a build
## - it doesn't take long.

.PHONY: compile compileall version generate_files really-compile rm-stampfile

compile: ${CLASSDIRECTORY} generate_files mocha really-compile

# compileall: removing the stamp file makes all Java files out of date so that
# we can reuse the compile target.
compileall: rm-stampfile compile

really-compile: ${STAMPFILE}  build_jniw

${STAMPFILE}: ${ESCJAVA_SOURCE_FILES_RELATIVE}
	@#./dl-xml-rpc-lib.sh # for future use
	./generate-functions-vcGenerator.sh #fixme, this is lame
	echo $? | xargs ${JAVAC} ${JAVAC_FLAGS} \
		-classpath $(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}) \
		-sourcepath $(call canonicalize,${SOURCEPATH}) \
		-d $(call canonicalize,${CLASSDIRECTORY}) 
	cp ${ESCJAVA_SOURCE_DIR}/escjava/gui/*.html ${CLASSDIRECTORY}/escjava/gui
	touch $@

build_jniw:
ifdef JNIW
	${MAKE} -C java/escjava/prover/jniw all
endif

VERSION_FILE = ${ESCJAVA_SOURCE_DIR}/escjava/Version.java
VERSION_TEMP = ${VERSION_FILE}-temp

version:
	@rm -f ${VERSION_TEMP}
	@touch ${VERSION_TEMP}
	@echo "package escjava;" >> ${VERSION_TEMP}
	@echo "public class Version {" >> ${VERSION_TEMP}
	@echo "    public static final /*@ non_null */ String VERSION = " \"${ESC_PROJECT}-${ESC_VERSION}\" ";" >> ${VERSION_TEMP}
	@echo "}" >> ${VERSION_TEMP}
	@if [ ! -e ${VERSION_FILE} ]; then \
		cp ${VERSION_TEMP} ${VERSION_FILE} ; fi
	@cmp -s ${VERSION_FILE} ${VERSION_TEMP} || cp ${VERSION_TEMP} ${VERSION_FILE}
	@rm -f ${VERSION_TEMP}

${CLASSDIRECTORY}:
	mkdir -p $@

generate_files: generate_escjava_ast_source generate_escjava_backpred

${ESCJAVA_SOURCE_DIR}/escjava/Main.java: ${ESCJAVA_SOURCE_DIR}/escjava/Version.java
	touch $@

rm-stampfile:
	-rm -f ${STAMPFILE}

###########################################################################
## mocha

MOCHA_DIR = ./mochalib
MOCHA_SOURCES = ${MOCHA_DIR}/java/mocha/*/*/*.java
MOCHA_CLASSES_DIR = ${MOCHA_DIR}/classes

.PHONY: mocha clean_mocha

mocha: ${MOCHA_CLASSES_DIR} ${MOCHA_LIB}

${MOCHA_LIB}: ${MOCHA_SOURCES}
	${JAVAC} -d $(call canonicalize,${MOCHA_CLASSES_DIR}) `find ${MOCHA_DIR} -name "*.java"`
	(cd ${MOCHA_CLASSES_DIR}; jar cf $(call canonicalize,${MOCHA_LIB}) mocha)

${MOCHA_CLASSES_DIR}:
	mkdir $@

clean_mocha:
	rm -rf ${MOCHA_CLASSES_DIR}
	rm -f ${MOCHA_LIB}

###########################################################################
# source file generation for escjava.ast

AST_DIR = ${ESCJAVA_SOURCE_DIR}/escjava/ast

# Generating Java files:

VECTEMPLATE = ${AST_DIR}/_TYPE_Vec.j

# The component type for these (e.g. GuardedCmd.java) is in Escjava.
TypeVec = \
	${AST_DIR}/GuardedCmdVec.java  \
	${AST_DIR}/ConditionVec.java  \
	${AST_DIR}/ExprModifierPragmaVec.java \
	${AST_DIR}/CondExprModifierPragmaVec.java \
	${AST_DIR}/ModifiesGroupPragmaVec.java \
	${AST_DIR}/ExprDeclPragmaVec.java \
	${AST_DIR}/ExprStmtPragmaVec.java \
	${AST_DIR}/VarExprModifierPragmaVec.java \
	${AST_DIR}/DefPredVec.java \
	${AST_DIR}/DecreasesInfoVec.java

.PHONY: source generate_escjava_ast_source clean_escjava_ast

source: generate_escjava_ast_source

generate_escjava_ast_source: ${AST_DIR}/GeneratedTags.java ${TypeVec}

# FIXME - ought to make these depend on the component type.
${AST_DIR}/%Vec.java: ${VECTEMPLATE}
	@if [ -e ${ESCJAVA_SOURCE_DIR}/escjava/ast/$*.java ]; then \
		sed s/_TYPE_/$*/g ${VECTEMPLATE} | \
		sed s/_PKG_/escjava.ast/g              | \
		sed s/_COMPONENTPKG_/escjava.ast/g > $@ ;\
	else \
		echo "Could not find component for $@" ;\
	fi


# This also generates all the .java files implied by "hierarchy.j".
${AST_DIR}/GeneratedTags.java ${AST_DIR}/AnOverview.java: ${AST_DIR}/hierarchy.j
	@(cd ${AST_DIR}; ${ESC_ASTGEN} < hierarchy.j; \
		grep -v "//@" AnOverview.java > AnOverview.tmp; \
		mv AnOverview.tmp AnOverview.java; \
		grep -v "//@" GCExpr.java > GCExpr.tmp; \
		mv GCExpr.tmp GCExpr.java; \
		grep -v "//@" GuardedCmd.java > GuardedCmd.tmp; \
		mv GuardedCmd.tmp GuardedCmd.java)
## The above grep's etc. are a kludge and ought to be fixed. - FIXME David
## These three classes are the three root abstract classes of 
## escjava.ast.  Why do we need to strip out their specs? -Joe

clean_escjava_ast:
	(cd ${AST_DIR}; rm -f `${ESC_ASTFILELIST} < ${AST_DIR}/hierarchy.j`)
	rm -f *~ ${TypeVec} ${AST_DIR}/ASTGEN* ${AST_DIR}/zASTGEN* *.tmp
	rm -f ${AST_DIR}/*.class ${CLASSDIRECTORY}/escjava/ast/*.class

###########################################################################
## Generating the source files in backpred.

BACKPRED_DIR = ${ESCJAVA_SOURCE_DIR}/escjava/backpred
DEFAULT_UNIV_BACKPRED = ${BACKPRED_DIR}/DefaultUnivBackPred.java

.PHONY: generate_escjava_backpred clean_escjava_backpred

generate_escjava_backpred: ${DEFAULT_UNIV_BACKPRED}

${DEFAULT_UNIV_BACKPRED}: ${BACKPRED_DIR}/UnivBackPred.ax
	@rm -f ${DEFAULT_UNIV_BACKPRED}
	@echo 'package escjava.backpred;' > ${DEFAULT_UNIV_BACKPRED}
	@echo 'class DefaultUnivBackPred {' >> ${DEFAULT_UNIV_BACKPRED}
	@echo '  static String s =' >> ${DEFAULT_UNIV_BACKPRED}
	@sed -e 's/"/\\"/g' \
	  -e 's/^/    "/' \
	  -e 's/$$/\\n" +/' \
	  ${BACKPRED_DIR}/UnivBackPred.ax >> ${DEFAULT_UNIV_BACKPRED}
	@echo '    "";' >> ${DEFAULT_UNIV_BACKPRED}
	@echo '}' >> ${DEFAULT_UNIV_BACKPRED}

clean_escjava_backpred:
	rm -f ${DEFAULT_UNIV_BACKPRED}

###########################################################################
# Clean targets.
## FIXME - Check all these

.PHONY: clean clean_escjava clean_tests clean_classfiles \
	clean_docs clean_misc clean_archives clean_vcGeneration

clean: clean_classfiles clean_docs clean_escjava clean_tests \
	clean_misc clean_archives \
	clean_mocha

clean_escjava: clean_escjava_ast clean_escjava_backpred clean_vcGeneration clean_jniw
	rm -f ${VERSION_FILE} ${VERSION_TEMP}

clean_vcGeneration:
	(cd java/escjava/vcGeneration; ./clean.sh)

clean_tests:
	${MAKE} -C ${TESTSOURCEDIRECTORY} clean

clean_classfiles:
	rm -rf ${CLASSDIRECTORY}
	-rm -rf ${ESCJAVA_CLASSFILES4RAC}

clean_docs: 
	rm -f doc/escjava.html doc/escjava-public.html doc/man1/*.1

clean_misc:
	rm -f ${STAMPFILE}
	rm -f ${TESTSOURCEDIRECTORY}/escjava/test/*/log ${TESTSOURCEDIRECTORY}/matrix/log
	find . -name mon.out -o -name "core.*" -exec rm -f {} \;

clean_archives:
	rm -f ${CLASSDIRECTORY}/escjava.zip
	rm -f ${CLASSDIRECTORY}/compiled.zip

clean_jniw:
ifdef JNIW
	${MAKE} -C java/escjava/prover/jniw clean
endif

###########################################################################
# Deliverable generation targets.
## FIXME - check this out - move it one level up???

.PHONY: archives

archives: archive_esctools
	cd ${ESCTOOLS_ROOT}; \
	tar cjf ${PROJECT}-${VERSION}.tbz ${PROJECT}

###########################################################################
# Testing targets.

TOP_ESCJAVA_TEST_DIR = ${TESTSOURCEDIRECTORY}/escjava/test

.PHONY: test_escjava tests junittests jdktests \
	checkspecs javaspecs test_escjava_parser \
	test_matrix self_test test-release self_typecheck

test_escjava: build
	$(MAKE) tests

tests:
	$(MAKE) junittests
	$(MAKE) test_escjava_parser
	$(MAKE) -C test/nijmegen/OCF runtests
	$(MAKE) -C test/nijmegen/JCF runtests
	$(MAKE) jdktests
	##$(MAKE) self_typecheck
	##$(MAKE) test_matrix   ## FIXME

junittests:
	$(MAKE) -C test/junittests runtests

jdktests:
	$(MAKE) -C test/jdktests runtests

checkspecs:
	@echo "Type checking the ESC/Java2 specs...."
#ifdef JML_ROOT
	jml -Q ${ESCJAVA2_SPECS}
#endif
	escjava -Q */*/*.* */*/*/*.*

javaspecs:
	$(MAKE) -C test/javaspecs runtests

test_escjava_parser:
	$(MAKE) -C ${TOP_ESCJAVA_TEST_DIR} test

self_typecheck:
	@echo "Using SPECS=${SPECS}"
	@JAVA_VM_FLAGS=-Xmx256m \
	${ESCJ} -classpath "$(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}:${CLASSPATH}:${JAVAFE_LIB})" \
		-sourcepath "$(call canonicalize,$(SOURCEPATH):$(SPECS):$(JUNIT_SOURCEPATH))" \
		-specs "$(call canonicalize,${ESCTOOLS_ROOT}/specs})" \
		-source 1.4 -typecheck -quiet \
		${PACKAGE_LIST}


%-self:
	JAVA_VM_FLAGS=-Xmx256m \
	${ESCJ} -classpath "$(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}:${CLASSPATH}:${JAVAFE_LIB})" \
		-sourcepath "$(call canonicalize,$(SOURCEPATH):$(SPECS):$(JUNIT_SOURCEPATH))" \
		-specs "${call canonicalize,${ESCTOOLS_ROOT}/specs}" \
		-source 1.4 -typecheck -quiet \
		$*

test_matrix:
	JAVA_VM_FLAGS=-Xmx256m \
	${ESCJ} -source 1.4 -sourcepath ${TESTSOURCEDIRECTORY} \
		-classpath "${call canonicalize,${TESTSOURCEDIRECTORY}:${ESCTOOLS_ROOT}/Escjava/java:${ESCTOOLS_ROOT}/Utils}" \
		-specs "${call canonicalize,${ESCTOOLS_ROOT}/specs" \
		${TESTSOURCEDIRECTORY}/matrix/*.java

self_test_nocheck: build
	JAVA_VM_FLAGS=-Xmx256m \
		${ESCJ} -classpath "$(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}:${CLASSPATH}:${JAVAFE_LIB})" \
		-sourcepath "$(call canonicalize,${SOURCEPATH})" \
		-vclimit 1000000 \
		-nocheck -source 1.4 -quiet -nosemicolonwarnings \
		$(PACKAGE_LIST)

self_test: build
	@echo PROVER_KILL_TIME=${PROVER_KILL_TIME}
	@echo JAVA_VM_FLAGS=${JAVA_VM_FLAGS}
	export JAVA_VM_FLAGS="$(JAVA_VM_FLAGS)"; \
	${ESCJ} -classpath "$(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}:${CLASSPATH}:${JAVAFE_LIB})" \
		-sourcepath "$(call canonicalize,${SOURCEPATH})" \
		-vclimit 1000000 \
		-source 1.4 -nosemicolonwarnings \
		$(ESCJ_EXTRA_FLAGS) \
		$(PACKAGE_LIST)

# Chalin: temporary custom targets based on self_test.
self_test0:
	unset CLASSPATH; \
	export JAVA_VM_FLAGS=-Xmx1024m; \
	${MAKE} ESCJ_EXTRA_FLAGS+=" -nowarn Modifies" PROVER_KILL_TIME=7 self_test

## FIXME - is this used? should it be?
## This tests a binary release that is present in the RELTEMP directory,
## with RELCLASSPATH pointing to the relevant esctools2.jar file.

test-release:
	@echo TESTING RELEASE - Escjava
	$(MAKE) ESC_CLASSPATH=$(call canonicalize,${RELCLASSPATH}:${RELTEMP}) \
		CLASSDIRECTORY=${RELTEMP} \
		ALLTESTS=1 \
		ESCJ=${RELTEMP}/escj \
		ESCJ_SIMPLIFY_DIR=${RELTEMP} \
		tests

###########################################################################

.PHONY: string check-core-jdk-specs

string:
	${ESCJ} \
		-classpath "$(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}:${CLASSPATH})" \
		-specs "${call canonicalize,${ESCTOOLS_ROOT}/specs-kiniry:${JML2_SPECS_JAR}" \
		-nowarn Deadlock -nowarn Post -nowarn Invariant \
		-source 1.4 -quiet -noSemicolonWarnings \
		../specs-kiniry/java/lang/String.java \
		-routine String

check-core-jdk-specs: string
