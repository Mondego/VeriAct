#!/bin/bash
../release-files/escj -quiet  -classpath ../Utils/junit.jar    `find /home/jcharles/workspaces/Mobius/ESCJava2/specs-java1.5 -type d | grep -v \.svn` | sed "/.*Caution: Using given file as the .java file, even though it is not the java file for.*/ \
{D}" 1> log.std 2>log.err

