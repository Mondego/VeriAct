## $Id$
##
## This file is part of ESC/Java2 from 2003 onward.
##
## author: David Cok
## author: Joe Kiniry
## author: Patrice Chalin
## author: Dermot Cochran
##
## copyright (c) 2007-2008 Dermot Cochran
##

export TOP = .
include Makefile.defs

################################################################################
## These are the standard targets.  They are executed in all the
## subdirectories that are listed in the SUBDIRS variable:
##  clean - removes all files created by a build
##  build - compiles out of date generated source and class files (default)
##  buildall - creates all generated source and class files 
##  test  - runs the tests (build must have been run)
##  alltests - test + rarely used or long tests
##  docs  - creates all docs; to get javadocs you must execute the
##          make from this directory
##  all+   - does all of the above
##
## You can run each of these in a subdirectory and only apply it to
## the code in that subdirectory, if you like.
##
## These are additional top-level-only targets
##  source-release - makes the tar file constituting a full source release
##  binary-release - makes the tar file constituting a binary release
##  releases-notests - makes both releases without doing any tests
##  releases - makes both releases with tests after
##  jmlc - run jmlc on source files in Javafe, Escjava and Utils
##  self_typecheck - run escjava2's typechecker on all source files
##
## You can run each of these in a subdirectory and only apply it to
## the code in that subdirectory, if you like.

export ESCJAVA ?= ${ESCTOOLS_ROOT}/release-files/escj

# List subprojects in the order in which they should be built.
SUBDIRS = Utils Escjava 

.PHONY: default all all+ nodocs nodocs+ jars

default: build

all: clean build jars test quiet-docs

all+: clean build jars quiet-docs alltests

nodocs: clean build jars test

nodocs+: clean build jars alltests

jars: $(ESCJAVA2_SPECS_JAR) $(ESCJAVA2_SPECS_JAVA1_4_JAR) $(ESCJAVA2_SPECS_JAVACARD_JAR) jar srcjar

################################################################################
## Top-level build and test rules

.PHONY: build buildall test escjavatests alltests \
	escjavaalltests findbugs self_typecheck test1

build buildall: source fix-perms
	for d in $(SUBDIRS) ; do \
	    $(MAKE) -C $$d $@ || exit 1; \
	done

source: fix-perms
	$(MAKE) -C Escjava source generate_files

test: build test0

test0:
	for d in $(SUBDIRS) ; do \
	    $(MAKE) -C $$d test || exit 1; \
	done

# Use the following target if you have *some* files that were compiled with jmlc (e.g. Utils)
test1: build
	${MAKE} EXTRA_CP=":$(JML_CLASSROOT)/bin/jmlruntime.jar" test

escjavatests: build
	$(MAKE) -C Escjava test || exit 1;

alltests:	build
	${MAKE} test ALLTESTS=1
	##${MAKE} specs_test

escjavaalltests: build
	$(MAKE) -C Escjava test ALLTESTS=1 || exit 1;

ifdef FINDBUGS
findbugs: jar
	${FINDBUGS} -textui esctools2.jar
endif

# Choose which specs you wish to use for documentation generation and
# typechecking.

# The JML specs,
#SPECS = ${JML_ROOT}/specs:${JML_ROOT}
# The ESC/Java2 specs.
export SPECS ?= $(ESCJAVA2_SPECS)

self_typecheck: build
	$(MAKE) -C Escjava self_typecheck

self_esc: build
	$(MAKE) -C Escjava self_test

################################################################################
## Test using jmlc compiled sources:

# Eventually use $(SUBDIRS).  For now only test Escjava
jmlc_test:
	$(MAKE) -C Utils build
	@echo CLASSPATH=${CLASSPATH}
	@echo
	for d in Escjava ; do \
	    $(MAKE) -C $$d USE_RAC=1 test || exit 1; \
	done

################################################################################
## General rules to clean up build tree

.PHONY: fix-perms fix-perms-helper fix-644-perms fix-755-pers

PERMS_FILE=.perms-fixed

## This rule was needed because the repository did not properly keep permissions
## (expecially x bits), so sometimes we need a reset.  Delete the
## datestamp file ".perms-fixed" to force re-execution.
fix-perms: fix-perms-helper $(PERMS_FILE)

fix-perms-helper:
	[ -e "$(PERMS_FILE)" ] || $(MAKE) fix-644-perms fix-755-perms

$(PERMS_FILE):
	touch $(PERMS_FILE)

clean-perms-stamp:
	-rm -f $(PERMS_FILE)

fix-644-perms:
	find $(ESCTOOLS_ROOT) -type f \
		\( -name "*.[0-9]" -or \
		   -name "*.[chjlo]" -or \
		   -name "*ans" -or \
		   -name "*diff" -or \
		   -name "*.el" -or \
		   -name "*.gif" -or \
		   -name "*.html" -or \
		   -name "*.spec" -or \
		   -name "*.j*" -or \
		   -name "*Make*" -or \
		   -name "*out" -or \
		   -name "README*" -or \
		   -name "*.txt" -or \
		   -name "*.zip" \) \
		-exec chmod 644 {} \; ;

fix-755-perms:
		find $(ESCTOOLS_ROOT) -type f \
		\( -name "*.sh" -or \
		   -name "*.pl" -or \
		   -name "rtest*" -or \
		   -name "escj" -or \
		   -name "escj.bat" -or \
		   -name "escjava2.exe" -or \
		   -name "escjava2" -or \
		   -name "escjava.bat.win" -or \
		   -name "escjava.linux" -or \
		   -name "escjava.nt.win" -or \
		   -name "escjava.solaris" -or \
		   -name "escjava.tru64" -or \
		   -name "run" -or \
		   -name "*_tags" -or \
		   -name "Simplify-1.5*" -or \
		   -name "insert_ensures" \) \
		-exec chmod 755 {} \;

################################################################################
## Clean rules

.PHONY: clean clean-release clean-norel clean-release-test clean-norel-noreltemp

# To really clean, we need to clean Javafe last since some of its
# tools are used in cleaning.
clean: clean-norel clean-release clean-release-test clean-norel-noreltemp cleanup

# Cleans out the release directory
clean-release:
	rm -rf ${RELDIR}

# Cleans everything except the release directory
clean-norel: clean-norel-noreltemp
	rm -rf ${RELTEMP}

# Cleans release test directory
clean-release-test:
	rm -rf ${RELTEST}

# Cleans everything except the various release directories (test,
# temp, release)
clean-norel-noreltemp:
	rm -rf ${PATCH_DIR}
	for d in $(SUBDIRS) ; do \
	    ( $(MAKE) -C $$d clean || exit 1) ; \
	done
	-rm -rf Javafe
	-rm -rf Simplify
	-rm -rf classfiles/*
	$(MAKE) -C docs clean
	$(MAKE) -C papers clean
	$(MAKE) -C slides clean
	rm -f $(JAR_FILES)/*.jar
	rm -f tags TAGS SpecTest.java .specs

# Cleans all backups files, log files, merge files, etc.
cleanup:
	-find ${ESCTOOLS_ROOT} \
		\( -name "*~" -or -name "*.#*" -or -name "semantic.cache*" \) \
		-exec rm -f {} \; > /dev/null 2>&1
	-rm -f docs_log
ifdef COMSPEC
	 -find Escjava -name "*.j" -o -name "*.sh" -exec d2u {} \;
endif

################################################################################
## Specifications

$(ESCJAVA2_SPECS_JAR):
	( cd specs; jar cf $(ESCJAVA2_SPECS_JAR) . )

$(ESCJAVA2_SPECS_JAVA1_4_JAR):
	( cd specs-java1.4; jar cf $(ESCJAVA2_SPECS_JAVA1_4_JAR) . )

$(ESCJAVA2_SPECS_JAVA1_5_JAR):
	( cd specs-java1.4; jar cf $(ESCJAVA2_SPECS_JAVA1_5_JAR) . )
$(ESCJAVA2_SPECS_JAVACARD_JAR):
	( cd specs-javacard; jar cf $(ESCJAVA2_SPECS_JAVACARD_JAR) . )

# RDG: We are not shipping jmlspecs.jar right now because the large VC
# conditions generated for the full JML specs can make our theorem prover choke

#$(JML2_SPECS_JAR):
#ifdef JML_ROOT
#	${MAKE} -C ${JML_ROOT} jmlspecs-jar
#	cp ${JML_ROOT}/bin/jmlspecs.jar ${JML2_SPECS_JAR}
#else
#	@echo NOT REMAKING JARS FROM JML - since JML_ROOT not defined
#	@true
#endif

################################################################################
## Rules to build documentation

.PHONY: alldocs textdocs quiet-docs docs slides papers javadoc jmldoc jml

alldocs: textdocs javadoc

textdocs: quiet-docs slides papers

quiet-docs:
	-${MAKE} docs > docs_log 2>&1

docs:
	-${MAKE} -C docs

slides:
	${MAKE} -C slides build

papers:
	${MAKE} -C papers build

## Javadoc depends on the build target since some of the .java files
## are generated in the course of the build (e.g. the AST classes).
## TODO: Should have -overview , header, footer, bottom, group,
## stylesheetfile link/linkoffline, subpackages -FIXME Kiniry
javadoc: build
	rm -rf $(JAVADOC_GEN_DIR)
	mkdir -p $(JAVADOC_GEN_DIR)
	mkdir -p $(JAVADOC_GEN_DIR)/images
	cp $(DOCS_STYLESHEET) $(JAVADOC_GEN_DIR)
	$(JAVADOC) -sourcepath \
	  $(call canonicalize,$(SOURCEPATH):$(JUNIT_SOURCEPATH)) \
	  -classpath $(call canonicalize,${CLASSPATH}:${JUNIT_LIB}:${ANT_LIB}:${XMLRPC_LIB}:${BCEL_LIB}:${JAVAFE_LIB}) \
	  -overview docs/overview.html \
	  -header $(COPYRIGHT) \
	  -footer $(COPYRIGHT) \
	  -bottom '<a href="http://secure.ucd.ie/products/opensource/ESCJava2/">The ESC/Java2 Project Homepage</a>' \
	  -stylesheetfile $(DOCS_STYLESHEET) \
	  -private \
	  -doctitle "ESCTools API" \
	  -windowtitle "ESCTools" \
	  -author -version -use \
	  -linksource \
	  -group EscJava "escjava.*" \
	  -group Utils   "junitutils.*" \
	  -tag warning -tag todo -tag note -tag design -tag usage \
	  -tag requires \
	  -d $(JAVADOC_GEN_DIR) \
          $(PACKAGE_LIST)

jmldoc: build
	rm -rf $(JMLDOC_GEN_DIR)
	mkdir -p $(JMLDOC_GEN_DIR)
	mkdir -p $(JMLDOC_GEN_DIR)/images
	$(JMLDOC) -sourcepath \
	  $(call canonicalize,$(SOURCEPATH):$(SPECS):$(JUNIT_SOURCEPATH)) \
	  -classpath "$(call canonicalize,${ESC_CLASSPATH}:${JUNIT_LIB}:${ANT_LIB}:${BCEL_LIB}:${JAVAFE_LIB}:${XMLRPC_LIB}:${CLASSPATH})" \
	  -overview docs/overview.html \
	  -header $(COPYRIGHT) \
	  -footer $(COPYRIGHT) \
	  -bottom '<a href="http://kind.ucd.ie/products/opensource/ESCJava2/">The ESC/Java2 Project Homepage</a>' \
	  -stylesheetfile $(DOCS_STYLESHEET) \
	  -private --source 1.4 \
	  -doctitle "ESCTools API" \
	  -windowtitle "ESCTools" \
	  -author -version -use \
	  -linksource \
	  -group EscJava "escjava.*" \
	  -group Utils   "junitutils.*" \
	  -tag warning -tag todo -tag note -tag design -tag usage \
	  -tag requires \
	  -d $(JMLDOC_GEN_DIR) \
	  -A \
          $(PACKAGE_LIST)

################################################################################
## JML checker targets:

JML_PACKAGES=junitutils escjava

jml: build
	-$(JML) \
	  --sourcepath "$(call canonicalize,$(SOURCEPATH):$(ESCTOOLS_ROOT)/specs)" \
	  --classpath "$(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}:${CLASSPATH})" \
	  --assignable --Assignable --purity --defaultNonNull \
	  --keepGoing --source 1.4 --Quiet --recursive \
	  $(JML_PACKAGES)

#===============================================================================
## JMLc targets:
#
.PHONY: jmlc jmlc_utils jmlc_fe jmlc_ej jmlc_with_new_semantics

# FIXME: Escjava Makefiles should be adapted like the Utils Makefile
# so that the build target's behavior is altered when USE_JMLC is defined.

jmlc: jmlc_utils jmlc_ej jmlc_with_new_semantics

jmlc_utils:
	$(MAKE) USE_JMLC=1 -C Utils buildall

jmlc_ej: ${ESCJAVA_CLASSFILES4RAC}
	export JMLC_XMX=512M; \
	$(MAKE) JMLC_CLASSFILES_DIR=${ESCJAVA_CLASSFILES4RAC} \
		TARGET=escjava \
		JMLC_FLAGS+=' --excludefiles "^(SPVC)\.java"' \
		jmlc1

# Apply jmlc to one TARGET at a time; class files are to be saved in JMLC_CLASSFILES_DIR.
jmlc1:
	$(JMLC) \
	  --sourcepath "$(call canonicalize,$(SOURCEPATH):$(SPECS))" \
	  --classpath "$(call canonicalize,${ESCJAVA_BUILD_CLASSPATH}:)${CLASSPATH}" \
	  -d $(JMLC_CLASSFILES_DIR) \
	  -a -A --purity \
	  -n \
	  --keepGoing --recursive \
	  $(JMLC_FLAGS) \
	  $(TARGET) \
	  && touch .jmlc

# Some files cannot be compiled under the current/old semantics because the generated code
# has a try/catch block that is beyond the limits of the VM.  Hence, we compile these files
# separately using the new semantics (-T).
jmlc_with_new_semantics:
	$(MAKE) JMLC_CLASSFILES_DIR=${ESCJAVA_CLASSFILES4RAC} \
		TARGET='Escjava/java/escjava/sp/SPVC.java' \
		JMLC_FLAGS+=' --efficientRAC' jmlc1

${ESCJAVA_CLASSFILES4RAC}:
	mkdir ${ESCJAVA_CLASSFILES4RAC}

JMLC_CLASSFILES=jmlc_classfiles

${JMLC_CLASSFILES}:
	mkdir -p ${JMLC_CLASSFILES}

#..............................................................................
# Run jmlc on each file one by one (this target is superceded by target 'jmlc').

jmlc1b1: build jmlc_escjava_1b1

 

jmlc_escjava_1b1:
	@echo "ESCJAVA============================================================"
	for f in `find Escjava/java -name *.java`; do \
		${MAKE} JMLC_CLASSFILES_DIR=${ESCJAVA_CLASSFILES4RAC} TARGET=$$f jmlc1; \
	done
	${MAKE} jmlc_escjava_cleanup

################################################################################
## Rules to build various release types

.PHONY: releases test-releases releases-notests source-release \
	jar srcjar binary-release test-binary-release \
	generate-patches generate-archives generate-specs-archive \
	generate-toplevel-archive zero-length-files version

## FIXME: get mocha right

releases: releases-notests 
	${MAKE} test-releases

test-releases:
	@echo "Testing the releases..."
	${MAKE} test-source-release
	${MAKE} test-binary-release

releases-notests:
	@echo "Making release versions ${ESC_PROJECT}-${ESC_VERSION} ........"
	${MAKE} binary-release
	${MAKE} source-release

source-release: cleanup quiet-docs
	@echo "Creating source release..."
	cp -rf ${ESCJAVA2_SPECS} ${RELTEMP} 
	cp -rf ${ESCJAVA2_SPECS_JAVA1_4} ${RELTEMP} 
	cp -rf ${ESCJAVA2_SPECS_JAVACARD} ${RELTEMP} 
	rm -rf ${RELTEMP}/original-specs 
	cp -r ${ORIG_SRC_SPECS} ${RELTEMP} 
#   Copy all source code, tests, libs, Makefiles, ChangeLog, etc. 
	cp -rf ${UTILS_SOURCE_DIR} ${RELTEMP}
	cp -rf ${ESCJAVA_ROOT} ${RELTEMP}
	-cp -rf ${MOCHA_SOURCE_DIR} ${RELTEMP}/Escjava/mochalib
	cp ${XMLRPC_LIB} ${RELTEMP}/Escjava/xmlrpc.jar 
	cp Makefile ${RELTEMP}
	cp Makefile.defs ${RELTEMP}
	cp Makefile.local.* ${RELTEMP} 
	cp build.xml ${RELTEMP} 
	cp prj.el ${RELTEMP} 
	cp ChangeLog ${RELTEMP} 
	cp -rf ${ORIG_SRC_SPECS} ${RELTEMP} 
	cp -rf docs ${RELTEMP}
	cp -rf papers ${RELTEMP}
	-cp -rf slides ${RELTEMP}
	cp -rf specs* ${RELTEMP}
	cp license.txt ${RELTEMP}
	mkdir ${RELTEMP}/jars
	-cp -r ${ESCTOOLS_ROOT}/jars/mochalib.jar ${RELTEMP}/jars
	mkdir -p ${RELTEMP}/Utils
	cp -rf ${ESCTOOLS_ROOT}/Utils/BCEL ${RELTEMP}/Utils
	cp -rf ${ESCTOOLS_ROOT}/Utils/Javafe ${RELTEMP}/Utils
#   Copy Javadoc API docs 
	cp -rf $(JAVADOC_GEN_DIR) ${RELTEMP}/docs 
#   Copy other miscellaneous files 
	mkdir -p ${RELTEMP}/release-files
	mkdir -p ${RELTEMP}/Escjava/release/master/bin
	cp ${ESCJ_ROOT}/escj ${RELTEMP}/release-files 
	cp ${ESCJ_ROOT}/escj.bat ${RELTEMP}/release-files 
	cp ${ESCJ_ROOT}/escjava2.exe ${RELTEMP}/release-files 
	cp ${ESCJ_ROOT}/escjava2.sample ${RELTEMP}/release-files 
	cp ${ESCJ_ROOT}/escjava2 ${RELTEMP}/release-files 
	cp ${ESCJ_SIMPLIFY_DIR}/Simplify-1.5.4.* ${RELTEMP}/Escjava/release/master/bin
	cp ${ESCJ_SIMPLIFY_DIR}/Simplify-1.5.5.* ${RELTEMP}/Escjava/release/master/bin
	cp README.first README.txt ${RELTEMP} 
	cd release-files; cp `ls | grep -v .svn` ${RELTEMP} 
	-find ${RELTEMP} -name "*.svn*" -exec rm -rf {} \; > /dev/null 2>&1
	-find ${RELTEMP} -name "*~" -or -name ".#*" -exec rm -f {} \; > /dev/null 2>&1 
	cd ${RELTEMP}; tar cjvf ${RELDIR}/${RELSRCTAR}.tbz *
	cd ${RELTEMP}; tar czvf ${RELDIR}/${RELSRCTAR}.tgz *
	cd ${RELTEMP}; zip -Ar ${RELDIR}/${RELSRCTAR}.zip *
	rm -rf ${RELTEMP}

version:
	$(MAKE) -C Escjava version

jar:	build
	mkdir -p ${RELTEMP}/sub
	cp ${ESCTOOLS_ROOT}/Escjava/java/escjava/vcGeneration/xml/*.xslt ${ESCJAVA_CLASSFILES}/escjava/vcGeneration/xml
	cp ${ESCTOOLS_ROOT}/Escjava/java/escjava/vcGeneration/xml/xmlprover.dtd ${ESCJAVA_CLASSFILES}/escjava/vcGeneration/xml
	cp -Rf ${ESCJAVA_CLASSFILES}/* ${RELTEMP}/sub
	mkdir -p ${RELTEMP}/sub/junitutils
	-( cd ${ESCTOOLS_ROOT}/Utils/junitutils; \
	  cp `find . -name '*.class'` ${RELTEMP}/sub/junitutils )
	( cd ${RELTEMP}/sub; jar xf $(ESCJAVA2_SPECS_JAR) )
	( cd ${RELTEMP}/sub; jar xf $(ESCJAVA2_SPECS_JAVA1_4_JAR) )
	( cd ${RELTEMP}/sub; jar xf $(ESCJAVA2_SPECS_JAVACARD_JAR) )
	-find ${RELTEMP} -name "*.svn*" -exec rm -rf {} \; > /dev/null 2>&1
	( cd ${RELTEMP}/sub; jar cmf ../../manifest.txt \
	  ${RELJAR} *; jar i ${RELJAR}; )
	cp ${RELJAR} ${ESCTOOLS_ROOT}
	rm -rf ${RELTEMP}/sub

srcjar:
	mkdir -p ${RELTEMP}/sub
	cp -Rf ${ESCTOOLS_ROOT}/Escjava/java/* ${RELTEMP}/sub
	cp -Rf ${ESCJAVA2_SPECS} ${RELTEMP}/sub/specs
	cp -Rf ${ESCJAVA2_SPECS_JAVA1_4} ${RELTEMP}/sub/specs-java1.4
	cp -Rf ${ESCJAVA2_SPECS_JAVACARD} ${RELTEMP}/sub/specs/javacard
	cp -Rf ${ORIG_SRC_SPECS} ${RELTEMP}/sub/original-specs
	mkdir -p ${RELTEMP}/sub/junitutils
	-( cd ${ESCTOOLS_ROOT}/Utils/junitutils; \
	  cp `find . -name '*.java'` ${RELTEMP}/sub/junitutils )
	-find ${RELTEMP} -name "*.svn*" -exec rm -rf {} \; > /dev/null 2>&1
	( cd ${RELTEMP}/sub; jar cf ${RELSRCJAR} *; )
	rm -rf ${RELTEMP}/sub

.PHONY: binary-release
binary-release: build jars quiet-docs cleanup
	@echo "Creating binary release .........."
#       RELDIR is the staging area for all files
#       First copy all .class files to sub and build a jar file
	mkdir -p ${RELDIR}
	rm -rf ${RELTEMP}
	mkdir -p ${RELTEMP}
	cp license.txt ${RELTEMP}
	${MAKE} jars
#       Now copy in all specs, new and old
	rm -rf ${RELTEMP}/specs*
	cp -rf ${ESCJAVA2_SPECS} ${RELTEMP}
	cp -rf ${ESCJAVA2_SPECS_JAVA1_4} ${RELTEMP}
	cp -rf ${ESCJAVA2_SPECS_JAVACARD} ${RELTEMP}
	rm -rf ${RELTEMP}/original-specs
	cp -rf ${ORIG_SRC_SPECS} ${RELTEMP}
#       Copy Javadoc API docs
	cp -rf $(JAVADOC_GEN_DIR) ${RELTEMP}
#       Copy papers and slides and other documentation
	mkdir ${RELTEMP}/papers
	cp papers/*.pdf ${RELTEMP}/papers
	mkdir ${RELTEMP}/slides
	-cp slides/ETAPSTutorial/*.pdf ${RELTEMP}/slides
	-cp slides/introjml.pdf ${RELTEMP}/slides
	-cp slides/escjava_demo.pdf ${RELTEMP}/slides
	mkdir ${RELTEMP}/docs
	cp docs/Escjava2-ImplementationNotes/*.pdf ${RELTEMP}/docs
	cp docs/Escjava2-Extending/Escjava2-Extending.pdf ${RELTEMP}/docs
	cp docs/Escjava2-Logics/Escjava2-Logics.pdf ${RELTEMP}/docs
	cp docs/Escjava2-Provers/Escjava2-Provers.pdf ${RELTEMP}/docs
	cp docs/ESCJAVA-UsersManual.html ${RELTEMP}/docs
	cp -rf Escjava/release/master/examples ${RELTEMP}
#       Copy actual jar files with all ESC/Java2's classfiles and the two jar files
#       with specs in them.
	cp ${RELJAR} ${RELTEMP}
	cp $(ESCJAVA2_SPECS_JAR) ${RELTEMP}
	cp $(ESCJAVA2_SPECS_JAVA1_4_JAR) ${RELTEMP}
	cp $(ESCJAVA2_SPECS_JAVACARD_JAR) ${RELTEMP}
#       cp $(JML2_SPECS_JAR) ${RELTEMP}
#       Copy other miscellaneous files
	${MAKE} clean-norel-noreltemp
	${MAKE} -C Utils build
	mkdir -p ${RELTEMP}/Utils
	cp -rf ${ESCTOOLS_ROOT}/Utils/BCEL ${RELTEMP}/Utils
	cp -rf ${ESCTOOLS_ROOT}/Utils/Javafe ${RELTEMP}/Utils
	cp ${ESCTOOLS_ROOT}/release-files/quicktest ${RELTEMP}
	cp ${ESCTOOLS_ROOT}/release-files/escj ${RELTEMP}
	cp ${ESCTOOLS_ROOT}/release-files/escj.bat ${RELTEMP}
	cp ${ESCTOOLS_ROOT}/release-files/escjava2 ${RELTEMP}
	cp ${ESCTOOLS_ROOT}/release-files/escjava2.exe ${RELTEMP}
	cp ${ESCTOOLS_ROOT}/release-files/escjava2.sample ${RELTEMP}
	cp ${ESCJ_SIMPLIFY_DIR}/Simplify-1.5.4.* ${RELTEMP}
	cp ${ESCJ_SIMPLIFY_DIR}/Simplify-1.5.5.* ${RELTEMP}
#   The two readme files below are irrelevant for binary release
#   README.first README.txt
	cd release-files; cp `ls | grep -v CVS` ${RELTEMP} 
	-find ${RELTEMP} -name "*~" -exec rm -f {} \; > /dev/null 2>&1
	-find ${RELTEMP} -name "*.svn*" -exec rm -rf {} \; > /dev/null 2>&1
	-find ${RELTEMP} -name "src-html" -exec rm -rf {} \; > /dev/null 2>&1
	-rm -rf ${RELTEMP}/sub
	cd ${RELTEMP}; tar cjvf ${RELDIR}/${RELTAR}.tbz *
	cd ${RELTEMP}; tar czvf ${RELDIR}/${RELTAR}.tgz *
	cd ${RELTEMP}; zip -Ar ${RELDIR}/${RELTAR}.zip *
	rm -rf ${RELTEMP}

test-binary-release:
	@echo "Testing binary release ............"
	rm -rf ${RELTEST}
	mkdir -p ${RELTEST}
	cp ${RELDIR}/${RELTAR}.tbz ${RELTEST}
	cd ${RELTEST} ;\
		unset ESC_CLASSPATH ; \
		unset ESCTOOLS_RELEASE ; \
		tar xjf ${RELTAR}.tbz ;\
		${RELTEST}/quicktest

test-source-release:
	@echo "Testing source release ............"
	rm -rf ${RELTEST}
	mkdir -p ${RELTEST}
	cp ${RELDIR}/${RELSRCTAR}.tbz ${RELTEST}
	cd ${RELTEST} ;\
		unset ESCJ_ROOT ; \
		unset ESCTOOLS_ROOT ; \
		unset ESC_SPECS ; \
		unset JUNIT_LIB ; \
		unset ANT_LIB ; \
		unset PATCH_DIR ; \
		unset RELDIR ; \
		unset RELTEMP ; \
		unset RELTEST ; \
		unset ESCJ_SIMPLIFY_DIR ; \
		unset ESC_CLASSPATH ;\
		unset JAR_FILES ; \
		tar xjf ${RELSRCTAR}.tbz ;\
		ESCTOOLS_ROOT=${RELTEST} $(MAKE) -s clean build test

generate-archives: generate-toplevel-archive generate-specs-archive

## Contains both specs and original-specs for the patch release
generate-specs-archive:
	mkdir -p ${RELTEMP}
	rm -rf ${RELTEMP}/specs/*
	rm -rf ${RELTEMP}/original-specs
	cp -rf ${ESCJAVA2_SPECS} ${RELTEMP}
	cp -rf ${ESCJAVA2_SPECS_JAVA1_4} ${RELTEMP}
	cp -rf ${ESCJAVA2_SPECS_JAVACARD} ${RELTEMP}
	cp -rf ${ORIG_SRC_SPECS} ${RELTEMP}
	( cd ${RELTEMP}; tar cjf ${PATCH_DIR}/${SPECS_TAR} specs original-specs; )
	rm -rf ${RELTEMP}/specs ${RELTEMP}/original-specs

generate-toplevel-archive: clean
	rm -rf ${RELTEMP}/ESCTools
	mkdir -p ${RELTEMP}/ESCTools
	cp ${ESCTOOLS_ROOT}/ChangeLog ${ESCTOOLS_ROOT}/Makefile \
		${ESCTOOLS_ROOT}/Makefile.defs \
		${ESCTOOLS_ROOT}/Makefile.local.sample \
		${ESCTOOLS_ROOT}/README.first ${ESCTOOLS_ROOT}/README.txt \
		${ESCTOOLS_ROOT}/TODO.kotl ${ESCTOOLS_ROOT}/build.xml \
		${ESCTOOLS_ROOT}/clean_env.sh ${RELTEMP}/ESCTools
	cp -rf ${ESCTOOLS_ROOT}/docs ${RELTEMP}/ESCTools
	cp -rf ${ESCTOOLS_ROOT}/papers ${RELTEMP}/ESCTools
	-cp -rf ${ESCTOOLS_ROOT}/slides ${RELTEMP}/ESCTools
	cp -rf ${ESCTOOLS_ROOT}/Utils ${RELTEMP}/ESCTools
	cp -rf ${JAR_FILES} ${RELTEMP}/ESCTools
# 	cp $(JML2_SPECS_JAR) ${RELTEMP}/ESCTools
# 	cp $(ESCJAVA2_SPECS_JAR) ${RELTEMP}/ESCTools
	mkdir -p ${RELTEMP}/ESCTools/Escjava/release/master/bin
	cp ${ESCTOOLS_ROOT}/Escjava/release/master/bin/Simplify-1.5.* \
		${RELTEMP}/ESCTools/Escjava/release/master/bin
	( cd ${RELTEMP}/ESCTools; tar cjf ${PATCH_DIR}/${TOPLEVEL_TAR} . ; )
	rm -rf ${RELTEMP}/ESCTools

## Until we can figure out how to have diff/patch detect and generate
## 0 length files, we have to archive them up all independently.
zero-length-files:
	mkdir -p ${PATCH_DIR}
	cd ${ESCTOOLS_ROOT}; \
	find . -empty ! -name .M3SHIP -print0 | \
		xargs -0 tar cjf ${PATCH_PREFIX}-ZeroLengthFiles.tbz
	( cd ${ESCTOOLS_ROOT}; \
	  for f in `cd ${ESCTOOLS_ORIGINAL}; find . -empty`; do \
	      if [ ! -e $$f ]; then echo $$f; fi; \
	  done ) > ${PATCH_PREFIX}-emptyFilesThatDisappeared

############################################################################
## Testing our specifications

.PHONY: specs_test check_specs

# SPECS_TEST_TARGETS ?= java.io java.lang \
# 	java.lang.reflect java.math java.net java.util java.util.regex
SPECS_TEST_TARGETS:=`find ${ESCJAVA2_SPECS} -type d | grep -v \.svn`
SPECS_TEST_CP ?= ${JUNIT_LIB}

## Run escjava on the specs
specs_test: specs_test_java15

specs_test_java14:
	$(MAKE) ESCJAVA2_SPECS=${ESCJAVA2_SPECS_JAVA1_4} _specs_test
specs_test_java15:
	$(MAKE) ESCJAVA2_SPECS=${ESCJAVA2_SPECS_JAVA1_5} _specs_test

_specs_test: build
	@echo "TESTING specs: "
	@echo "  CLASSPATH: ${SPECS_TEST_CP}"
	@echo "  SPECS path: ${ESCJAVA2_SPECS}"
	@echo "  Packages to be tested: ${SPECS_TEST_TARGETS}"
	${ESCJAVA} -quiet \
	  -classpath ${SPECS_TEST_CP} \
	  -specs ${ESCJAVA2_SPECS} \
	  ${SPECS_TEST_TARGETS}

## Experimental rule to generate a file that references every type
## that currently has a spec and run escjava2 on it to try to detect
## problems with package- and private-level types.
#check_specs: build $(JML2_SPECS_JAR)
check_specs: build
	find specs -type f | grep -v Entries | grep -v Repository | grep -v Root | \
	  grep -v "*~" | sed -e s+specs/++ | sed -e s+\\.spec++ | \
	  sed -e s+\\.refines-spec++ | sed -e s+\\.refines++ | \
	  sed -e s+\\.jml++ | sed -e s+\\.java++ | sed -e s+-java++ | \
	  sed -e s+\\.refines-java++ | sed -e s+/+.+g | \
	  grep -v VERSION | grep -v META | grep -v '\.\.' | \
	  grep -v "\.$$" | sort > .specs
	cat .specs | sed -e s+.*+import\ \&\;+ > SpecTest.java
	@echo "class SpecTest {" >> SpecTest.java
	i=0; for f in `cat .specs`; do \
	  echo "$$f x$$i ;" >> SpecTest.java ; \
	  ((i=i+1)); \
	done
	@echo "}" >> SpecTest.java
	CLASSPATH=$(call canonicalize,${CLASSPATH}:${JUNIT_LIB}:${ANT_LIB}:${BCEL_LIB}:${JAVAFE_LIB}); $(ESCJAVA) \
	  SpecTest.java
	rm -f .specs SpecTest.java

################################################################################
## Show system variables

.PHONY: show show-vars

show: show-vars

show-vars:
	@echo "CLASSPATH = $(CLASSPATH)"

################################################################################
## Simple smoke tests

.PHONY: make-test

## This is simply a smoke test of the Makefile, specific to Windows,
## thus the guard using the environmental variable COMSPEC defined by
## cygwin.
make-test:
ifdef COMSPEC
	@echo `cygpath -m $(ESCJ_SIMPLIFY_DIR)`
	@echo ${foo}
endif

################################################################################
## Maven pre-configuration

.PHONY: pre-maven
export MAVEN_FILES_ROOT=${ESCTOOLS_ROOT}
export MVN ?= /usr/bin/mvn

## Copy all Java files to default locations expected by Maven
pre-maven: source
	mkdir -p ${MAVEN_FILES_ROOT}/src/main/java
	find ${ESCTOOLS_ROOT}/Escjava/java/escjava -name "*.java" -exec cp -f {} ${MAVEN_FILES_ROOT}/src/main/java \;
	find ${ESCTOOLS_ROOT}/Escjava/mochalib/java -name "*.java" -exec cp -f {} ${MAVEN_FILES_ROOT}/src/main/java \;
	find ${ESCTOOLS_ROOT}/Utils/junitutils -name "*.java" -exec cp -f {} ${MAVEN_FILES_ROOT}/src/main/java \;

maven: pre-maven
	${MVN} clean test install

# End of Makefile
