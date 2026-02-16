# $Id$

export TOP = ..
include ${TOP}/Makefile.defs

###########################################################################
# build targets

SOURCEFILES := junitutils/*.java
CLASSESSTAMP = classes.stamp
CLASSDIRECTORY=.

CP = ${call canonicalize,${UTILS_CLASSFILES}:${JUNIT_LIB}:$(ESCJAVA2_SPECS)}

PROVER_KILL_TIME?=7

#------------------------------------------------------------------------------

.PHONY: clean test build esctest compile buildall compileall

build:    checkjunitjar compile
buildall: checkjunitjar compileall

compile: ${CLASSESSTAMP}
compileall: clean compile

${CLASSESSTAMP}: 
# ${SOURCEFILES}
# ifdef USE_JMLC
# 	$(JMLC) $(JMLC_FLAGS) $?
# 	@# For the command on the previous line to succeed, you need to 
# 	@# have junit and the JML run-time jar & specs in your class path.
# 	@#$(JMLC) $(JMLC_FLAGS) --classpath $(CP) $?
# else
# 	@#${JAVAC} -classpath ${CP} $?
# 	@echo "junitutils: no longer build here since they are a part of the Javafe."
# endif
# 	touch ${CLASSESSTAMP}

.PHONY: checkjunitjar
checkjunitjar:
	@if [ ! -e ${JUNIT_LIB} ]; then \
		echo Error: ${JUNIT_LIB} should be a version of junit.jar, but it does not exist ;\
		exit 1; \
	fi

test: build

self_test:
	@echo PROVER_KILL_TIME=$(PROVER_KILL_TIME)
	${ESCJ} -classpath ${CP} \
		${ESCJ_FLAGS} \
		${SOURCEFILES}

self_test0:
	${MAKE} PROVER_KILL_TIME=7 self_test

clean:
	@-rm -f junitutils/*.class ${CLASSESSTAMP}

.PHONY: docs
docs:
	@true

.PHONY: info
info:
	@echo "This package contains: " `cat $(SOURCEFILES) | wc -l` " LOC."

# End of file
