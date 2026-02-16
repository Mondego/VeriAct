# Copyright Hewlett-Packard, 2002

P = escjava.parser.test

runtest: testlex testparse testparseidempotence

full: java
	javac -d ${CLASSDIRECTORY} TestLex.java TestEscPragmaParser.java

java:
	@echo "Nothing to build."

clean:
	rm -f *~ *.out
	rm -rf *.class ${CLASSDIRECTORY}/javafe/parser/test/*.class

testlex:
	${JAVA} ${P}.TestLex GoodPragmas.j | diff - GoodPragmas.ans
	${JAVA} ${P}.TestLex GoodSpecExpr.j | diff - GoodSpecExpr.ans
	${JAVA} ${P}.TestLex lookahead GoodPragmas.j | diff - GoodPragmas.ans
	${JAVA} ${P}.TestLex lookahead GoodSpecExpr.j | diff - GoodSpecExpr.ans

testparse:
	${JAVA} ${P}.TestEscPragmaParser print TypeDeclElemPragmas1.java | diff - TypeDeclElemPragmas1.ans

testparseidempotence:
	${JAVA} ${P}.TestEscPragmaParser progress check idempotence \
		*.java ../*.java ../../ast/*.java
