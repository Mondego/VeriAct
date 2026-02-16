#!/bin/bash

# Set the separator between directories in a file path
PSEP=/

# TARDIR is set to the directory containing the tar files of the original
# release

if [ $# != 0 ]; then
    TARDIR=$1
fi
if [ "$TARDIR" == "" ]; then
    echo Please supply a command-line argument to the install script that gives the location of the tar files for the original ESCTools
    exit
fi
echo Using original tar files from ${TARDIR}

# Presume that we are in the directory where the ESCTools are to be built
# And that the patches are in this directory
PATCH_DIR=.

# Create a fresh directory structure of the original tools
for t in ${TARDIR}${PSEP}*.tar; do
    if [ "$t" != "${TARDIR}${PSEP}simplify.tar" ]; then
	echo Untarring original $t
	tar xf $t
    fi
done
mkdir Simplify
cd Simplify
echo Untarring original $TARDIR${PSEP}simplify.tar
tar xf $TARDIR${PSEP}simplify.tar
cd ..

# Apply each patch.
for p in ${PATCH_DIR}${PSEP}*-patch.bz ; do
    echo Patching with $p
    bzcat $p | patch -sNp1
done

# Unpack the toplevel 
echo Applying archives
for q in specs TopLevel ZeroLengthFiles ; do
    echo .... Untarring ${PATCH_DIR}${PSEP}*-${q}.tbz
    tar xjf ${PATCH_DIR}${PSEP}*-${q}.tbz
done


echo Removing empty files
rm -rf `cat ${PATCH_DIR}${PSEP}*emptyFilesThatDisappeared`

# Fix permissions on test files
#echo Fixing permissions
#chmod +x `find . -name run`
#chmod +x `find . -name rtestall`
#chmod +x `find . -name rtest`
#chmod +x `find . -name make_tags`
#chmod +x `find . -name insert_ensures`
#chmod +x `find . -name clean.sh`
#chmod +x `find . -name "*.sh"`
#chmod +x `find . -name "*.pl"`

(cd Escjava/release/master/bin; mv Simplify-1.5.4.exe.win Simplify-1.5.4.exe)

###### Have the user create their own Makefile.local.
###### Wait for them to finish.

#export ESCTOOLS_ROOT=`pwd`
#echo Building and testing the patched release in ${ESCTOOLS_ROOT}

# Clean, build, and test the release.
#make -s clean build test

echo Completed installation and test script
