## $Id$
##
## This file is part of Javafe from 2007 onward.
##
## author: David Cok
## author: Joe Kiniry
## author: Patrice Chalin
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
##  all+   - does all of the above
##
## You can run each of these in a subdirectory and only apply it to
## the code in that subdirectory, if you like.  Note though, that the
## subdirectories other than Javafe presume that Javafe has been built.
##
## These are additional top-level-only targets
##  source-release - makes the tar file constituting a full source release
##  binary-release - makes the tar file constituting a binary release
##  releases-notests - makes both releases without doing any tests
##  releases - makes both releases with tests after
##  jmlc - run jmlc on source files in Javafe and Utils
##
## You can run each of these in a subdirectory and only apply it to
## the code in that subdirectory, if you like.  Note though, that the
## subdirectories other than Javafe presume that Javafe has been
## built.

# List subprojects in the order in which they should be built.  Javafe
# must be before any other Javafe directories.
SUBDIRS = Utils Javafe

.PHONY: default all all+ nodocs nodocs+ jars

default: build

all: clean build jars test quiet-docs

all+: clean build jars quiet-docs alltests

nodocs: clean build jars test

nodocs+: clean build jars alltests

jars: jar srcjar

################################################################################
## Top-level build and test rules

.PHONY: build buildall test javafetests alltests \
	javafealltests findbugs self_typecheck test1

build buildall: source fix-perms
	for d in $(SUBDIRS) ; do \
	    $(MAKE) -C $$d $@ || exit 1; \
	done

source: fix-perms
	$(MAKE) -C Javafe source vector

test: build test0 jml

test0:
	for d in $(SUBDIRS) ; do \
	    $(MAKE) -C $$d test || exit 1; \
	done

# Use the following target if you have *some* files that were compiled with jmlc (e.g. Utils)
test1: build
	${MAKE} EXTRA_CP=":$(JML_CLASSROOT)/bin/jmlruntime.jar" test

javafetests: build
	$(MAKE) -C Javafe test || exit 1;

alltests:	build
	${MAKE} test ALLTESTS=1
#	${MAKE} specs_test

javafealltests: build
	$(MAKE) -C Javafe test ALLTESTS=1 || exit 1;

self_typecheck: build
	$(MAKE) -C Javafe self_typecheck 

self_test self_test0: build
	export PROVER_CC_LIMIT=1; \
	export PROVER_IN_ITS_OWN_THREAD=1; \
	export PROVER_KILL_TIME=4; \
	export SPECS="$(HOME)/dev/JMLspecs/java4"; \
	$(MAKE) ESCJ_EXTRA_FLAGS+=" -nonNullByDefault" \
		SPECS="$(HOME)/dev/JMLspecs/java4" \
		ESCJ_VERBOSE=1 -C Javafe self_test0

# Note: self_test also tests Utils. Use the following if you only want to test Utils.
self_test_utils:
	$(MAKE) -C Utils self_test0

################################################################################
## General rules to clean up build tree

.PHONY: fix-perms .perms-fixed

## This rule was needed because the repository did not properly keep permissions
## (expecially x bits), so sometimes we need a reset.  Delete the
## datestamp file ".perms-fixed" to force re-execution.
# PC: this is no longer needed now that we are using SVN rather than CVS.
#fix-perms:	.perms-fixed
fix-perms:


.perms-fixed:
	@if [ ! -e ".perms-fixed" ]; \
	then \
		find $(JAVAFE_ROOT) -type f \
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
		-exec chmod 644 {} \; ;\
		find $(JAVAFE_ROOT)/Javafe -type f \
		\( -name "*.sh" -or \
		   -name "*.pl" -or \
		   -name "rtest*" -or \
		   -name "run" -or \
		   -name "*_tags" -or \
		   -name "insert_ensures" \) \
		-exec chmod 755 {} \; ;\
		touch .perms-fixed; \
	fi

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
	for d in $(SUBDIRS) ; do \
	  if [ !"$$d"=="Javafe" ]; then \
	    ( $(MAKE) -C $$d clean || exit 1) ; fi ; \
	done
	$(MAKE) -C Javafe clean
	rm -f $(JAR_FILES)/*.jar
	rm -f tags TAGS SpecTest.java .specs

# Cleans all backups files, log files, merge files, etc.
cleanup:
	-find ${JAVAFE_ROOT}/Javafe \
		\( -name "*~" -or -name "*.#*" -or -name "semantic.cache*" \) \
		-exec rm -f {} \; > /dev/null 2>&1
	-rm -f docs_log

################################################################################
## Rules to build documentation

.PHONY: alldocs textdocs quiet-docs docs slides papers javadoc jmldoc jml

alldocs: textdocs javadoc

textdocs: quiet-docs slides papers

# Choose which specs you wish to use for documentation generation and
# typechecking.

# The JML specs,
#SPECS = ${JML_ROOT}/specs:${JML_ROOT}
# The ESC/Java2 specs.
export SPECS ?= $(JAVAFE_ROOT)/specs

## Javadoc depends on the build target since some of the .java files
## are generated in the course of the build (e.g. the AST classes).
## TODO: Should have -overview , header, footer, bottom, group,
## stylesheetfile link/linkoffline, subpackages -FIXME Kiniry
javadoc: build
	rm -rf $(JAVADOC_GEN_DIR)
	mkdir -p $(JAVADOC_GEN_DIR)
	mkdir -p $(JAVADOC_GEN_DIR)/images
	cp $(DOCS_STYLESHEET) $(JAVADOC_GEN_DIR)
	cp $(JAVAFE_ROOT)/Javafe/doc/javadoc/images/*.gif $(JAVADOC_GEN_DIR)/images
	$(JAVADOC) -sourcepath \
	  $(call canonicalize,$(SOURCEPATH):$(JUNIT_SOURCEPATH)) \
	  -classpath $(call canonicalize,${CLASSPATH}:${JUNIT_LIB}:${ANT_LIB}:${XMLRPC_LIB}:${BCEL_LIB}) \
	  -overview docs/overview.html \
	  -header $(COPYRIGHT) \
	  -footer $(COPYRIGHT) \
	  -bottom '<a href="http://kind.ucd.ie/products/opensource/E/">The ESC/Java2 Project Homepage</a>' \
	  -stylesheetfile $(DOCS_STYLESHEET) \
	  -private \
	  -doctitle "Javafe API" \
	  -windowtitle "Javafe" \
	  -author -version -use \
	  -linksource \
	  -group Javafe  "javafe.*" \
	  -group Utils   "junitutils.*" \
	  -tag warning -tag todo -tag note -tag design -tag usage \
	  -tag requires \
	  -d $(JAVADOC_GEN_DIR) \
          $(PACKAGE_LIST)

jmldoc: build
	rm -rf $(JMLDOC_GEN_DIR)
	mkdir -p $(JMLDOC_GEN_DIR)
	mkdir -p $(JMLDOC_GEN_DIR)/images
	cp $(JAVAFE_ROOT)/doc/javadoc/images/*.gif $(JMLDOC_GEN_DIR)/images
	$(JMLDOC) -sourcepath \
	  $(call canonicalize,$(SOURCEPATH):$(SPECS):$(JUNIT_SOURCEPATH)) \
	  -classpath "$(call canonicalize,${JUNIT_LIB}:${ANT_LIB}:${BCEL_LIB}:${XMLRPC_LIB}:)${CLASSPATH}" \
	  -overview docs/overview.html \
	  -header $(COPYRIGHT) \
	  -footer $(COPYRIGHT) \
	  -bottom '<a href="http://kind.ucd.ie/products/opensource/Javafe/">The Javafe Project Homepage</a>' \
	  -stylesheetfile $(DOCS_STYLESHEET) \
	  -private --source 1.4 \
	  -doctitle "Javafe API" \
	  -windowtitle "Javafe" \
	  -author -version -use \
	  -linksource \
	  -group Javafe  "javafe.*" \
	  -group Utils   "junitutils.*" \
	  -tag warning -tag todo -tag note -tag design -tag usage \
	  -tag requires \
	  -d $(JMLDOC_GEN_DIR) \
	  -A \
          $(PACKAGE_LIST)

################################################################################
## JML checker targets:

JML_PACKAGES=javafe

jml: build
	-$(JML) \
	  --sourcepath "$(call canonicalize,$(SOURCEPATH):$(SPECS))" \
	  --classpath $(call canonicalize,${JAVAFE_CLASSPATH}:${JUNIT_LIB}:${ANT_LIB}:${XMLRPC_LIB}:${BCEL_LIB}) \
	  --assignable --Assignable --purity --defaultNonNull -E parse \
	  --keepGoing --source 1.4 --Quiet --recursive \
	  $(JML_PACKAGES)

#===============================================================================
## JMLc targets:
#
.PHONY: jmlc jmlc_utils jmlc_fe

# FIXME: Javafe Makefiles should be adapted like the Utils Makefile
# so that the build target's behavior is altered when USE_JMLC is defined.

jmlc: jmlc_utils jmlc_fe

jmlc_utils:
	$(MAKE) USE_JMLC=1 -C Utils buildall

jmlc_fe: ${JAVAFE_CLASSFILES4RAC}
	export JMLC_XMX=512M; \
	$(MAKE) JMLC_CLASSFILES_DIR=${JAVAFE_CLASSFILES4RAC} \
		TARGET=javafe \
		JMLC_FLAGS+=' --excludefiles "^(ASTNode|TypeDeclElem)\.java"' \
		jmlc1

# Apply jmlc to one TARGET at a time; class files are to be saved in
# JMLC_CLASSFILES_DIR.
jmlc1:
	$(JMLC) \
	  --sourcepath "$(call canonicalize,$(SOURCEPATH):$(SPECS))" \
	  --classpath "$(call canonicalize,${CLASSPATH})" \
	  -d $(JMLC_CLASSFILES_DIR) \
	  -a -A --purity \
	  -n \
	  --keepGoing --recursive \
	  $(JMLC_FLAGS) \
	  $(TARGET) \
	  && touch .jmlc

# Some files cannot be compiled under the old semantics (-O) because
# the generated code has a try/catch block that is beyond the limits
# of the VM.  Hence, if you wish to use the old semantics, you must
# compile these files separately using the new semantics (no -O
# switch).  The files in question are javafe/ast/ASTNode.java
# and javafe/ast/TypeDeclElem.java.

${JAVAFE_CLASSFILES4RAC}:
	mkdir ${JAVAFE_CLASSFILES4RAC}

JMLC_CLASSFILES=jmlc_classfiles

${JMLC_CLASSFILES}:
	mkdir -p ${JMLC_CLASSFILES}

#..............................................................................
# Run jmlc on each file one by one (this target is superceded by target 'jmlc').

jmlc1b1: build jmlc_javafe_1b1

jmlc_javafe_1b1:
	@echo "JAVAFE============================================================"
	for f in `find Javafe/java -name *.java ! -name "*/jcheck/*"`; do \
		${MAKE} JMLC_CLASSFILES_DIR=${JAVAFE_CLASSFILES4RAC} TARGET=$$f jmlc1; \
	done;
	${MAKE} jmlc_javafe_cleanup

################################################################################
## Rules to build various release types

.PHONY: releases test-releases releases-notests source-release patch-release \
	jar srcjar binary-release test-binary-release test-patch-release \
	generate-patches generate-archives generate-specs-archive \
	generate-toplevel-archive zero-length-files version

releases: releases-notests 
	${MAKE} test-releases

test-releases:
	@echo "Testing the releases..."
	${MAKE} test-source-release
	${MAKE} test-binary-release

releases-notests:
	@echo "Making release versions ${JAVAFE_PROJECT}-${JAVAFE_VERSION} ........"
	mkdir -p $(JAVAFE_ROOT)/releases
	${MAKE} binary-release
	${MAKE} source-release

source-release: cleanup alldocs
	@echo "Creating source release..."
	cp -rf ${JAVAFE_SPECS} ${RELTEMP} 
#   Copy all source code, tests, libs, Makefiles, etc. 
	cp -rf ${UTILS_SOURCE_DIR} ${RELTEMP}
	cp -rf ${JAVAFE_ROOT}/Javafe ${RELTEMP}
	cp Makefile ${RELTEMP}
	cp Makefile.defs ${RELTEMP}
	cp -rf docs ${RELTEMP}
	cp -rf specs* ${RELTEMP}
	cp license.txt ${RELTEMP}
	mkdir -p ${RELTEMP}/bin
	cp -rf release-files/bin/ast* ${RELTEMP}/bin
	mkdir ${RELTEMP}/jars
#   Copy Javadoc API docs 
	cp -rf $(JAVADOC_GEN_DIR) ${RELTEMP}/docs 
#   Copy other miscellaneous files 
	cp README.first README.txt ${RELTEMP} 
	cd release-files; cp README.release RELEASE_NOTES.txt ${RELTEMP} 
	-find ${RELTEMP} -name "*.svn*" -exec rm -rf {} \; > /dev/null 2>&1
	-find ${RELTEMP} -name "*~" -or -name ".#*" -exec rm -f {} \; > /dev/null 2>&1 
	cd ${RELTEMP}; tar cjvf ${RELDIR}/${RELSRCTAR}.tbz *
	cd ${RELTEMP}; tar czvf ${RELDIR}/${RELSRCTAR}.tgz *
	cd ${RELTEMP}; zip -Ar ${RELDIR}/${RELSRCTAR}.zip *
	rm -rf ${RELTEMP}

jar:	build
	mkdir -p ${RELTEMP}/sub
	cp -Rf ${JAVAFE_CLASSFILES}/* ${RELTEMP}/sub
	mkdir -p ${RELTEMP}/sub/junitutils
	( cd ${JAVAFE_ROOT}/Utils/junitutils; \
	  cp `find . -name '*.class'` ${RELTEMP}/sub/junitutils )
	( cd ${RELTEMP}/sub; jar cf \
	  ${RELJAR} *; jar i ${RELJAR}; )
	cp ${RELJAR} ${JAVAFE_ROOT}
	rm -rf ${RELTEMP}/sub

srcjar:
	mkdir -p ${RELTEMP}/sub
	cp -Rf ${JAVAFE_ROOT}/Javafe/java/* ${RELTEMP}/sub
	cp -Rf ${JAVAFE_SPECS} ${RELTEMP}/sub/specs
	mkdir -p ${RELTEMP}/sub/junitutils
	( cd ${JAVAFE_ROOT}/Utils/junitutils; \
	  cp `find . -name '*.java'` ${RELTEMP}/sub/junitutils )
	( cd ${RELTEMP}/sub; jar cf ${RELSRCJAR} *; )
	cp ${RELSRCJAR} ${JAVAFE_ROOT}
	rm -rf ${RELTEMP}/sub

.PHONY: binary-release
# Binary release needs to include the test classfiles used by clients of this library
binary-release: build test jars alldocs cleanup
	@echo "Creating binary release .........."
#       RELDIR is the staging area for all files
#       First copy all .class files to sub and build a jar file
	mkdir -p ${RELDIR}
	rm -rf ${RELTEMP}
	mkdir -p ${RELTEMP}
	cp license.txt ${RELTEMP}
	${MAKE} jars
#       Now copy in all specs, new and old
	rm -rf ${RELTEMP}/specs
	cp -rf ${JAVAFE_SPECS} ${RELTEMP}
#       Copy Javadoc API docs
	cp -rf $(JAVADOC_GEN_DIR) ${RELTEMP}
	mkdir ${RELTEMP}/docs
	cp ${RELJAR} ${RELTEMP}
	mkdir -p ${RELTEMP}/bin
	cp -rf release-files/bin/ast* ${RELTEMP}/bin
#       Copy other miscellaneous files
	${MAKE} clean-norel-noreltemp
	${MAKE} -C Utils build
	mkdir -p ${RELTEMP}/Utils
	cp -rf ${JAVAFE_ROOT}/Utils/BCEL ${RELTEMP}/Utils
#   The two readme files below are irrelevant for binary release
#   cp README.first README.txt ${RELTEMP}
	cd release-files; cp README.release RELEASE_NOTES.txt ${RELTEMP} 
	-find ${RELTEMP} -name "*~" -exec rm -f {} \; > /dev/null 2>&1
	-find ${RELTEMP} -name "*.svn*" -exec rm -rf {} \; > /dev/null 2>&1
	-find ${RELTEMP} -name "src-html" -exec rm -rf {} \; > /dev/null 2>&1
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
	    unset JAVAFE_CLASSPATH ; \
	    unset JAVAFE_RELEASE ; \
	    tar xjf ${RELTAR}.tbz ;\

test-source-release:
	@echo "Testing source release ............"
	rm -rf ${RELTEST}
	mkdir -p ${RELTEST}
	cp ${RELDIR}/${RELSRCTAR}.tbz ${RELTEST}
	cd ${RELTEST} ;\
		unset JAVAFE_ROOT ; \
		unset JAVAFE_SPECS ; \
		unset JUNIT_LIB ; \
		unset ANT_LIB ; \
		unset PATCH_DIR ; \
		unset RELDIR ; \
		unset RELTEMP ; \
		unset RELTEST ; \
		unset JAR_FILES ; \
		tar xjf ${RELSRCTAR}.tbz ;\
		JAVAFE_ROOT=${RELTEST} $(MAKE) -s clean build test                         

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
	@echo `cygpath -m`
	@echo ${foo}
endif

# End of Makefile
