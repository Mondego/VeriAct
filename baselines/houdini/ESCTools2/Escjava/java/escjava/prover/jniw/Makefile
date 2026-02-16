##########################################################################
## Makefile for CVC JNI wrapper library libcvc3jniw.so
## "make all" makes the dynamic library
## "make test" makes a C++-executable test file
## "make java" makes a Java class test executable

## Note: if the jniw package moves, then it will be necessary to re-build
## the Cvc3Wrapper.h file (must be built from package root) and the
## JNI function calls in cvc3jniw.cpp must be modified to reflect the new
## names.  If not, then there will be library loading problems!

## TODO: move a copy of libcvc3jniw.so onto the libso path.

export TOP = ../../../../..
include ${TOP}/Makefile.defs

all: cvc3jniw.o
	$(CC) -shared -Wl,-soname,libcvc3jniw.so -L$(CVC_LIB_DIR) \
	-o libcvc3jniw.so cvc3jniw.o -lc -lcvc3
	cp libcvc3jniw.so ${TOP}/libs

Cvc3Wrapper.class: Cvc3WrapperException.java Cvc3Wrapper.java
	${JAVAC} Cvc3WrapperException.java
	${JAVAC} -classpath ../../.. Cvc3Wrapper.java

#Cvc3Wrapper.h: Cvc3Wrapper.class
#	${JAVAH} -jni -o Cvc3Wrapper.h escjava.prover.jniw.Cvc3Wrapper

cvc3jniw.o: cvc3jniw.cpp cvc3jniw.h Cvc3Wrapper.h
	$(CC) -O0 -g -Wall -fPIC -c \
	-I$(JDK_INCLUDE_DIR) -I$(JDK_INCLUDE_OS) \
	-I$(CVC_INCLUDE_DIR) -I. \
	-DCVCL_THEORY $(CVC_FLAGS) \
	-DHAVE_CONFIG_H  cvc3jniw.cpp

test: test.cpp all
	$(CC) -Wall -I. test.cpp -o test -L. -lcvc3jniw

java: all Cvc3Wrapper.class Test.java
	${JAVAC} -classpath ../../.. Test.java

clean: 
	-rm *.o *.so test *.class
	-rm ${TOP}/libs/*.so
