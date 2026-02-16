TOP = ../../../..
#include ${TOP}/Escjava/Makefile

sourcepath = ../..:${TOP}/Javafe/java:${TOP}/Utils/:${TOP}/Utils/junit.jar
classpath = ../../../classfiles:${TOP}/Javafe/classfiles:${TOP}/Utils/:${TOP}/Utils/junit.jar

all:	typecheck testers

typecheck:
	escjava2 -sourcepath ${sourcepath} -typecheck *.java

verify:
	escjava2 -sourcepath ${sourcepath} *.java

testers:
	java -classpath ${classpath} escjava.prover.Atom
	java -classpath ${classpath} escjava.prover.SExp
	java -classpath ${classpath} escjava.prover.SList
	java -classpath ${classpath} escjava.prover.Simplify
	java -classpath ${classpath} escjava.prover.Cvc


