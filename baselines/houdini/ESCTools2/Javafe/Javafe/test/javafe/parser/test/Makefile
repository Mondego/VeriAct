# $Id$
#
# Copyright 2000, 2001, Compaq Computer Corporation; 2006 dsrg.org.

JAVA ?= java

parseidempotence_java_files = $(wildcard *.java) \
	$(wildcard ${JAVAFE_ROOT}/Javafe/java/javafe/parser/*.java) \
	$(wildcard ${JAVAFE_ROOT}/Javafe/java/javafe/ast/*.java)

## FIXME _ what about testparse? (It is very verbose)

runtest: testlex testexpr testparseidempotence testpragmalexer testpragmaparser

bigtests: testexprbig testparsejdkidempotence

clean:
	rm -f *~ *.out

testlex:
	@echo "TESTING Javafe: testlex"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestLex \
	  < simpletoks > simpletoks.out \
	  && ${DIFF} simpletoks.Lex simpletoks.out
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestLex \
	  < literaltoks > literaltoks.out \
	  && ${DIFF} literaltoks.Lex literaltoks.out
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestLex lookahead javakeywords \
	  < simpletoks > simpletoks.out \
	  && ${DIFF} simpletoks.Lookahead simpletoks.out
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestLex lookahead javakeywords \
	  < literaltoks > literaltoks.out \
	  && ${DIFF} literaltoks.Lookahead literaltoks.out

testpragmalexer:
	@echo "TESTING Javafe: testpragmalexer"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestLex \
	  parsepragmas javakeywords \
	  < pragmatoks >pragmatoks.out \
	&& ${DIFF} pragmatoks.out pragmatoks.ans

testpragmaparser:
	@echo "TESTING Javafe: testpragmaparser"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestParse idempotence check HasPragmas.j

testexpr:
	@echo "TESTING Javafe: testexpr"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestExpr check < test-exprs1.j
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestExpr check compare < compare-exprs-small.j

testparseidempotence:
	@echo "TESTING Javafe: testparseidempotence"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestParse diff simpletoks simpletoks
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestParse progress check idempotence \
		compilation-unit.j $(call canonicalize,${parseidempotence_java_files})
# The first line above tests the diff algorithm...


testexprbig:
	@echo "TESTING Javafe: testexprbig"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestExpr check compare < compare-exprs-big.j

.PHONY: jdk_source_listings
jdk_source_listings:
	find ${JDK10_SOURCES} -follow -name "*.java" -print > jdk-1.0-files 
	find ${JDK11_SOURCES} -follow -name "*.java" -print > jdk-1.1-files 
	find ${JDK12_SOURCES} -follow -name "*.java" -print > jdk-1.2-files 
	find ${JDK13_SOURCES} -follow -name "*.java" -print > jdk-1.3-files 
	find ${JDK14_SOURCES} -follow -name "*.java" -print > jdk-1.4-files 

.PHONY: testparsejdkidempotence
testparsejdkidempotence:
	@echo "TESTING Javafe: testparsejdkidempotence"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestParse diff simpletoks simpletoks assert
	@for d in ${JDK10_SOURCES} ${JDK11_SOURCES} ${JDK12_SOURCES} ${JDK13_SOURCES} ${JDK14_SOURCES} ; do \
	   echo "Executing TestParse on " $$d  ; \
	    CLASSPATH=$(call canonicalize,${CLASSPATH}:.) ; \
	   ( cd $$d; find . -follow -name "*.java" -print | \
		xargs \
			${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestParse assert progress silent idempotence );\
	done

testparse:
	@echo "TESTING Javafe: testparse"
	@${JAVA} -classpath ${CLASSPATH} javafe.parser.test.TestParse progress check print verboseprogress \
		compilation-unit.j *.java \
		${JAVAFE_ROOT}/Javafe/java/javafe/parser/*.java \
		${JAVAFE_ROOT}/Javafe/java/javafe/ast/*.java

