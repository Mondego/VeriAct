# $Id$
#
# Copyright 2003-2006 dsrg.org
#

export TOP = ../../../..
include ${TOP}/Makefile.defs

TESTSOURCEDIRECTORY ?= ${ESCJAVA_ROOT}/test
PARSER_TEST_PACKAGE = escjava.parser.test
PARSER_TEST_DIR = ${TESTSOURCEDIRECTORY}/escjava/parser/test

## For these tests inherited from the original ESC release we use the
## specification files that came with the original release.
ESC_SPECS = ${ESCTOOLS_ROOT}/original-specs

CLASSDIRECTORY ?=${ESCJAVA_CLASSFILES}

###########################################################################

compile_escjava_parser_test_programs:
	@${JAVAC} ${JAVAC_FLAGS} \
		-classpath "$(call canonicalize,${ESC_CLASSPATH})" \
		-d $(call canonicalize,${CLASSDIRECTORY}) \
		$(call canonicalize,${PARSER_TEST_DIR}/TestLex.java) \
		$(call canonicalize,${PARSER_TEST_DIR}/TestEscPragmaParser.java)

###########################################################################
## parser test targets


test:	compile_escjava_parser_test_programs \
	test_lexer test_parser test_parser_idempotence test_escjava

test_and_clean: test clean

test_escjava:
	@echo "TESTING Escjava: test_escjava"
	-env CLASSPATH=. ./rtestall

%-test:
	env CLASSPATH=. ./rtest $* || ${DIFF} $*/out $*/ans

%-run:
	env CLASSPATH=. ./rtestrun $* 

CD_AND_JAVA = cd $(PARSER_TEST_DIR); $(JAVA) -classpath "$(call canonicalize,${ESC_CLASSPATH})" $(PARSER_TEST_PACKAGE)
CD_AND_JAVA_DEBUG = cd $(PARSER_TEST_DIR); $(JAVA) $(DEBUGGER_FLAGS) -classpath "$(call canonicalize,${ESC_CLASSPATH})" $(PARSER_TEST_PACKAGE)

test_lexer: compile_escjava_parser_test_programs
	@echo "TESTING Escjava: test_lexer"
	@$(CD_AND_JAVA).TestLex GoodPragmas.j | tee GoodPragmas.out | ${DIFF} - GoodPragmas.ans
	@$(CD_AND_JAVA).TestLex GoodSpecExpr.j | tee GoodSpecExpr.out | ${DIFF} - GoodSpecExpr.ans
	@$(CD_AND_JAVA).TestLex GoodJmlPragmas.j | tee GoodJmlPragmas.out | ${DIFF} - GoodJmlPragmas.ans
	@$(CD_AND_JAVA).TestLex lookahead GoodPragmas.j | tee GoodPragmas-lookahead.out | ${DIFF} - GoodPragmas.ans
	@$(CD_AND_JAVA).TestLex lookahead GoodSpecExpr.j | tee GoodSpecExpr-lookahead.out  | ${DIFF} - GoodSpecExpr.ans
	@$(CD_AND_JAVA).TestLex lookahead GoodJmlPragmas.j | tee GoodJmlPragmas-lookahead.out | ${DIFF} - GoodJmlPragmas.ans

test_parser: compile_escjava_parser_test_programs
	@echo "TESTING Escjava: test_parser"
	@$(CD_AND_JAVA).TestEscPragmaParser print TypeDeclElemPragmas1.java | \
		tee TypeDeclElemPragmas1.out | \
		${DIFF} - TypeDeclElemPragmas1.ans

test_parser_idempotence: compile_escjava_parser_test_programs
	@echo "TESTING Escjava: test_parser_idempotence"
	$(CD_AND_JAVA).TestEscPragmaParser progress check idempotence \
		*.java ../../../../java/escjava/ast/*.java

clean:	clean_out_and_diff clean_escjava_parser_test

clean_escjava_parser_test:
	rm -f ${PARSER_TEST_DIR}/*.out
	rm -f ${CLASSDIRECTORY}/javafe/parser/test/*.class

clean_out_and_diff:
	rm -f test*/out test*/diffs out diffs
	for f in test* ; do \
		( cd $$f; rm -f out diffs ;);\
	done

# End of file
