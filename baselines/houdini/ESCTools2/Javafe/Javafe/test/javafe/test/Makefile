# $Id$
# Copyright 2000, 2001, Compaq Computer Corporation
# Modified 2007, Systems Research Group, University College Dublin

export TOP=../../../..
include ${TOP}/Makefile.defs

runtest:
	@echo "TESTING Javafe: test/javafe/test"
	@cd lex; ${MAKE} runtest
	@cd fe; ${MAKE} runtest

clean: 
	cd lex; ${MAKE} clean
	cd fe; ${MAKE} clean

# $(foreach jar,$(wildcard $${d}/jars/*.jar),$(jar):)

# @note kiniry 25 Dec 2005 - Only the source of the JDK that we are
# currently running in (1.4 at this time) should completely typecheck.
# @see bug #142
.PHONY: typecheckJDK typecheckOldJDK typecheckJDK14
typecheckJDK: typecheckOldJDK typecheckJDK14

typecheckOldJDK:
	@for d in ${JDK_PRE14_SOURCES}; do \
	    echo "===== type checking $${d}" ;\
	    if [ -e $${d} ]; then \
	    (cd $${d} && \
		echo `find . -name '*.java'` | xargs \
		${JAVA} \
		-classpath ${CAN_CLASSDIRECTORY} \
		javafe.test.Print \
		-classpath $${d}$(call canonicalize,:$${d}/jars/classes.jar:$${d}/jars/rt.jar:$${d}/jars/tools.jar:$${d}/jars/ui.jar) \
		-typecheck -noprint -preferSource) ;\
	    else \
		echo "      $${d} does not exist";\
	    fi ;\
	done

typecheckJDK14:
	@for d in ${JDK_14_SOURCES}; do \
	    echo "===== type checking $${d}" ;\
	    if [ -e $${d} ]; then \
	    (cd $${d} && \
		echo `find . -name '*.java'` | xargs -n 300 \
		${JAVA} \
		-classpath ${CAN_CLASSDIRECTORY}$(call canonicalize,:classes.jar) \
		 javafe.test.Print \
		-classpath $${d}$(call canonicalize,:$${d}/jars/classes.jar:$${d}/jars/rt.jar:$${d}/jars/tools.jar:$${d}/jars/ui.jar) \
		-source 1.4 \
		-typecheck -noprint -preferSource ) ;\
	    else \
		echo "      $${d} does not exist";\
	    fi ;\
	done

.PHONY: typecheckJavafe
typecheckJavafe:
	@echo "===== type checking the Javafe package ====="
	@cd ${JAVAFE_ROOT}/Javafe/java && \
		for d in ast parser tc util filespace reader genericfile ; do \
		/bin/echo -n "$${d}	" ;\
		echo `find javafe/$${d} -name '*.java'` | xargs \
			${JAVA} -classpath $(call canonicalize,../classfiles${EXTRA_CP}:${BCEL_LIB}) javafe.test.Print \
			-classpath "${call canonicalize,.:../../Utils:${EXTRA_CP}:${BCEL_LIB}}" \
			-typecheck -noprint  ;\
		done 
	@/bin/echo -n "javafe/.	"
	@cd ${JAVAFE_ROOT}/Javafe/java && \
		${JAVA} -classpath $(call canonicalize,../classfiles${EXTRA_CP}:${BCEL_LIB}) javafe.test.Print \
		-classpath "${call canonicalize,.:../../Utils:${EXTRA_CP}:${BCEL_LIB}}" \
		-typecheck -noprint  javafe/*.java ;\


# End of Makefile
