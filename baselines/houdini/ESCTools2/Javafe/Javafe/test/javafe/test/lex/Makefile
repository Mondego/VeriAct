# $Id$
# Copyright 2000, 2001, Compaq Computer Corporation

JAVA ?= java

clean:
	rm -rf *.class test*.out

#
# Testing using TestLex:
#

TESTOR=javafe.test.lex.TestLex

test1.out test2.out: test1.in test2.in
	@${JAVA} -classpath "${JAVAFE_CLASSPATH_CAN}" ${TESTOR} < test1.in > test1.out
	@${JAVA} -classpath "${JAVAFE_CLASSPATH_CAN}" ${TESTOR} < test2.in > test2.out

runtest: test1.out test2.out
	@echo "TESTING Javafe: test/javafe/test/lex"
	@${DIFF} test1.out test1.ans
	@${DIFF} test2.out test2.ans
