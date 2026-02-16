# Copyright 2000, 2001, Compaq Computer Corporation
# Altered extensively by David R. Cok as part of the 2003 upgrade of
# the ESC tools.

# author: David Cok
# author: Joe Kiniry

JAVAFE_VERSION=1.0

export TOP = ..
include ${TOP}/Makefile.defs
# The above defines JAVAFE_ROOT, JAVAC, JAVAC_FLAGS

# Various conventional UNIX tools are used in the following, including cc in 
# preprocess only mode.

##############################################################################
## Variables needed in the Makefile.
export CLASSDIRECTORY = ${JAVAFE_ROOT}/Javafe/classfiles
export SOURCEDIRECTORY = ${JAVAFE_ROOT}/Javafe/java
export TESTSOURCEDIRECTORY = ${JAVAFE_ROOT}/Javafe/test
export CAN_CLASSDIRECTORY = ${call canonicalize,${CLASSDIRECTORY}}

## This one is needed in submakes that don't define their own classpath
RAW_CLASSPATH = ${CLASSDIRECTORY}:${UTILS_CLASSFILES}:${BCEL_LIB}
export CLASSPATH = $(call canonicalize,${RAW_CLASSPATH})

##############################################################################

.PHONY: default all build buildall test 

default: build

all: clean build test docs

build: fixperms compile

buildall: fixperms compileall

jar: build test
	jar cf dist/lib/javafe-$(JAVAFE_VERSION).jar -C classfiles .

ifdef ALLTESTS
test: extensive_tests self_typecheck typecheckJDK
else
test: runtest self_typecheck
endif

###########################################################################
# build targets

SOURCEFILES := java/javafe/*.java java/javafe/*/*.java java/tohtml/*.java

CLASSESSTAMP = classes.stamp

SHELLSCRIPTS =	java/javafe/ast/make_tags \
		java/javafe/ast/insert_ensures

.PHONY: fixperms compile compileall javafeall source jtools

fixperms:
	@chmod 755 $(SHELLSCRIPTS)

## Compiles everything that is missing or out of date
compile: ${CLASSDIRECTORY} source ${CLASSESSTAMP}

## Compiles all out of date java files
${CLASSESSTAMP}: ${SOURCEFILES}
	${JAVAC} ${JAVAC_FLAGS} -classpath $(call canonicalize,${CLASSDIRECTORY}:${SOURCEDIRECTORY}:${UTILS_CLASSFILES}:${BCEL_LIB}) \
	  -d ${CAN_CLASSDIRECTORY} $?
	touch ${CLASSESSTAMP}

## Full compile, including all java files, whether out of date or not
compileall: ${CLASSDIRECTORY} source javafeall

## Compiles all java files, whether out of date or not
javafeall:
	( cd java; \
	${JAVAC} ${JAVAC_FLAGS} -classpath $(call canonicalize,${CLASSDIRECTORY}:${SOURCEDIRECTORY}:${UTILS_CLASSFILES}:${BCEL_LIB}) \
	  -d ${CAN_CLASSDIRECTORY} \
	  `find javafe -name "*.java"` \
	  `find tohtml -name "*.java"` )
	touch ${CLASSESSTAMP}

## Runs all targets that create java files using other tools
source: 
	${MAKE} -C astgen
	${MAKE} -C java/javafe/ast source
	${MAKE} -C java/javafe/tc source

## Creates the output directory (where class files go), if necessary
${CLASSDIRECTORY}:
	mkdir -p ${CLASSDIRECTORY}

## Builds other tools (FIXME - do we need this? - not part of the regular build)
jtools:	
	${MAKE} -C java/jtools

###########################################################################
# clean targets

.PHONY: clean cleanclasses

clean: cleanclasses
	rm -f doc/javadoc/*.html ${CLASSESSTAMP}
	${MAKE} -C java/javafe/ast clean
	${MAKE} -C test/javafe/parser/test clean
	${MAKE} -C java/javafe/tc clean
	${MAKE} -C test/javafe/test clean
	${MAKE} -C java/jtools clean
	${MAKE} -C astgen clean     ##  Has to be last - used by others

# Erase ${CLASSDIRECTORY}
cleanclasses:
	rm -rf ${CLASSDIRECTORY}
	-rm -rf ${JAVAFE_CLASSFILES4RAC}

###########################################################################
## docs targets

.PHONY: docs

docs:	
	@true

## javadocs are generated from the parent directory now

###########################################################################
## test targets

JAVAFETESTTARGETS = \
		javafe/test/Print.java			\
		javafe/test/SuperlinksTest.java		\
		javafe/test/SupertypeTest.java		\
		javafe/test/CountFilesLoaded.java	\
		javafe/test/LocTool.java		\
		javafe/test/lex/TestLex.java		\
		javafe/parser/test/TestLex.java		\
		javafe/parser/test/TestExpr.java	\
		javafe/parser/test/TestParse.java

## Runs basic tests on Javafe

.PHONY: runtest extensive_tests test_jdk_sources typecheckJDK \
	self_typecheck test_release javafe_test

runtest: javafe_test
	${MAKE} -C test/javafe/parser/test runtest
	${MAKE} -C java/javafe/util runtest
	${MAKE} -C test/javafe/test runtest

## Runs basic tests, plus some (but not all) that are long
extensive_tests: javafe_test runtest
	${MAKE} -C test/javafe/parser/test bigtests

# This is included in extensive_tests above
test_jdk_sources:
	${MAKE} -C test/javafe/parser/test testparsejdkidempotence

## Runs the Javafe tool on bodies of code
typecheckJDK: javafe_test
	${MAKE} -C test/javafe/test typecheckJDK

self_typecheck:
	-${MAKE} -C test/javafe/test typecheckJavafe

self_test: build
	@echo PROVER_KILL_TIME=${PROVER_KILL_TIME}
	@echo JAVA_VM_FLAGS=${JAVA_VM_FLAGS}
	${ESCJ} -classpath "$(call canonicalize,${JUNIT_LIB}:${RAW_CLASSPATH}:${JAVAFE_LIB})" \
		-sourcepath "$(call canonicalize,${SOURCEPATH})" \
		-specs "$(SPECS)" \
		-vclimit 1000000 \
		-source 1.4 -nosemicolonwarnings \
		$(ESCJ_EXTRA_FLAGS) \
		$(PACKAGE_LIST)

# Example of using self_test0 to test a single file.
# make SPECS=/home/chalin/dev/JMLspecs/java4 PACKAGE_LIST=javafe.TestTool self_test0

self_test0:
	unset CLASSPATH; \
	export JAVA_VM_FLAGS=-Xmx1024m; \
	export PROVER_KILL_TIME=4; \
	export PROVER_CC_LIMIT=1; \
	${MAKE} ESCJ_EXTRA_FLAGS+=" -nonNullByDefault -nowarn Modifies -nowarn Exception" self_test

## This tests a binary release that is untarred in the RELTEMP directory,
## with RELCLASSPATH pointing to the esctools1.jar file that is being tested.
## The current directory should be the usual JAVAFE_ROOT directory so that
## the tests are available.
test-release: 
	@echo TESTING RELEASE - Javafe
	${MAKE} JAVAFE_CLASSPATH=$(call canonicalize,${RELCLASSPATH}) CLASSDIRECTORY=$(call canonicalize,${RELTEMP}) javafe_test
	${MAKE} ALLTESTS=1 CLASSPATH=$(call canonicalize,${RELCLASSPATH}) -C java/javafe/util runtest
	${MAKE} ALLTESTS=1 JAVAFE_CLASSPATH_CAN=$(call canonicalize,${RELCLASSPATH}) CLASSPATH=$(call canonicalize,${RELCLASSPATH}:${RELTEMP}) -C test/javafe/test runtest
	${MAKE} ALLTESTS=1 CLASSPATH=$(call canonicalize,${RELCLASSPATH}:${RELTEMP}) -C test/javafe/parser/test runtest
	rm -rf ${RELTEMP}/*

## This builds tests; presumes that the build has been made
javafe_test:
	@echo "building: ${JAVAFETESTTARGETS}"
	@( cd test; \
	  ${JAVAC} -classpath $(call canonicalize,${JAVAFE_CLASSPATH}:${TESTSOURCEDIRECTORY}) \
	  -d $(call canonicalize,${CLASSDIRECTORY}) \
	  ${JAVAFETESTTARGETS} )

###########################
## Build AST vector classes
###########################

AST_DIR = ${SOURCEDIRECTORY}/javafe/ast

ASTGEN_DIR = ${JAVAFE_ROOT}/astgen
ASTGEN = ${ASTGEN_DIR}/astgen
ASTFILELIST = ${ASTGEN_DIR}/astfilelist

${ASTFILELIST} ${ASTGEN}:
	${MAKE} -C ${ASTGEN_DIR}

VECTEMPLATE = ${SOURCEDIRECTORY}/javafe/util/_TYPE_Vec.j

# The component type for these is in Javafe.
JavafeTypeVec = \
	${AST_DIR}/LocalVarDeclVec.java \
	${AST_DIR}/GenericVarDeclVec.java

.PHONY: vector

vector: source ${AST_DIR}/GeneratedTags.java ${JavafeTypeVec}

# FIXME - ought to make these depend on the component type.
${AST_DIR}/%Vec.java: ${VECTEMPLATE}
	@if [ -e ${JAVAFE_SOURCE_DIR}/javafe/ast/$*.java ]; then \
		sed s/_TYPE_/$*/g ${VECTEMPLATE} | \
		sed s/_PKG_/javafe.ast/g              | \
		sed s/_COMPONENTPKG_/javafe.ast/g > $@ ;\
	else \
		echo "Could not find component for $@" ;\
	fi

