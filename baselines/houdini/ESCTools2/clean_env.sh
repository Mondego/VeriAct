#!/bin/sh

# This script *unsets* all environmental variables but for PATH set by
# Javafe/setup.sh, Escjava/setup.sh, and all related Makefiles.  Use
# this if you want/need to clean up your environment for "clean" build
# testing.

# Javafe/setup.sh union Escjava/setup.sh:
unset ANT_LIB
unset BOOTCLASSPATH
unset CLASSDIRECTORY
unset CLASSES
unset CLASSFILES
unset CLASSPATH
unset DECSRCCLASSDIRECTORY
unset DECSRCDIRECTORY
unset DECSRCLIBRARY
unset ESCJ
unset ESCJAVA_ROOT
unset ESCJ_SIMPLIFY
unset ESCSPEC
unset ESCTOOLS_JARS
unset ESCTOOLS_ROOT
unset ESC_CLASSPATH
unset JARS
unset JAVA
unset JAVAC
unset JAVADOC
unset JAVADOC_GEN_DIR
unset JAVAFE_CLASSFILES
unset JAVAFE_ROOT
unset JAVA_HOME
unset JDKBINARIES
unset JDK_BINARIES
unset JDK_SOURCES
unset JML_HOME
unset JML_SPECS
unset JUNIT_HOME
unset JUNIT_LIB
unset MOCHA_CLASSES
unset MOCHA_ROOT
unset SOURCEDIRECTORY
unset SOURCEPATH
unset TESTSOURCEDIRECTORY

# Top-level Makefile
unset TOP

# Makefile.defs
unset BOOTCLASSPATH
unset ESCJAVA_BUILD_CLASSPATH
unset ESCJAVA_CLASSFILES
unset ESC_CLASSPATH
unset ESCJAVA_ROOT
unset ESCJ_SIMPLIFY
unset ESCJ_SIMPLIFY_DIR
unset ESCSPEC
unset ESCTOOLS_JARS
unset ESC_PROJECT
unset ESC_VERSION
unset JAVA
unset JAVAC
unset JAVAC_FLAGS
unset JAVADOC
unset JAVADOC_GEN_DIR
unset JAVAFE_BUILD_CLASSPATH
unset JAVAFE_CLASSFILES
unset JAVAFE_CLASSPATH
unset JAVAFE_ROOT
unset JAVAFE_SOURCE_DIR
unset LD_LIBRARY_PATH
unset MOCHA_CLASSES
unset MOCHA_LIB
unset MOCHA_PROOT
unset MOCHA_ROOT
unset SIMPLIFY
unset SOURCEPATH
unset TOP

# Javafe/Makefile

# Escjava/Makefile

