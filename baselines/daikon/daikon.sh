#!/bin/bash
# Exit codes:
# 1  - wrong arguments
# 2  - compile error
# 3  - DynComp timeout
# 4  - Chicory (trace generation) failure
# 5  - Daikon inference failure
# 6  - JML annotation failure
# 7  - ESC annotation failure

if [ $# -ne 3 ]; then
  echo "Usage: $0 <class_name> <test_class_name> <number_of_test>"
  exit 1  # exit code 1: wrong arguments
fi


class_name=$1 
test_class_name=$2
numberOfTest=$3

echo "Class Name: $class_name"
echo "Test Class Name: $test_class_name"
echo "Number of Test: $numberOfTest"

echo $numberOfTest

echo "Step-1: Compile the program with the -g switch to enable debugging symbols."
javac -g *.java
if [ $? -ne 0 ]; then
    echo "Step-1: Compile failed"
    exit 2  # exit code 2: compile error
fi
echo "Step-1: Done"


echo "Step-3: Run the program under the control of the Chicory front"

echo "Step-3a: Generate comparability information and trace file for test: 0"
# [FIXED TIME 5 min to generate decls-DynComp once]
timeout 300 java -cp .:$DAIKONDIR/daikon.jar daikon.DynComp --decl-file=Test_0.decls-DynComp $test_class_name < 0.txt
if [ $? -eq 124 ]; then
    echo "Step-3-a: Exit DynComp timed out"
    exit 3  # exit code 3: DynComp timeout
fi
echo "Step-3-a: Done"

echo "Step-3-b: Generating trace file for all tests"
counter=0
while [ $counter -lt $numberOfTest ]; do 
  
  java -cp .:$DAIKONDIR/daikon.jar daikon.Chicory --dtrace-file=Test_$counter.dtrace.gz --comparability-file=Test_0.decls-DynComp $test_class_name < $counter.txt
  if [ $? -ne 0 ]; then
      echo "Step-3-b: Chicory failed for test $counter"
      exit 4  # exit code 4: trace generation failure
  fi
  ((counter++))
done
echo "Step-3-b: Done"


echo "Step-3-c: Run Daikon on the trace files."
#[With] Customized Config
#java -cp .:$DAIKONDIR/daikon.jar daikon.Daikon --config inv-config.config --suppress_redundant Test*.dtrace.gz
#[With] Default Config
java -cp .:$DAIKONDIR/daikon.jar daikon.Daikon --suppress_redundant Test*.dtrace.gz
if [ $? -ne 0 ]; then
    echo "Step-3-c: Daikon inference failed"
    exit 5  # exit code 5: Daikon inference failure
fi
echo "Step-3-c: Done"


echo "Step-4: Annotate the source with JML"
java -cp .:$DAIKONDIR/daikon.jar daikon.tools.jtb.Annotate --no_reflection --format jml Test*.inv.gz $class_name.java
if [ $? -ne 0 ]; then
    echo "Step-4: JML annotation failed"
    exit 6  # exit code 6: JML annotation failure
fi
echo "Step-4: Done"

echo "Step-5: Annotate the source with ESC/Java2 with the same invariants."
java -cp .:$DAIKONDIR/daikon.jar daikon.tools.jtb.Annotate --no_reflection --format esc Test*.inv.gz $class_name.java
if [ $? -ne 0 ]; then
    echo "Step-5: ESC annotation failed"
    exit 7  # exit code 7: ESC annotation failure
fi
echo "Step-5: Done"

echo "Step-6: Clean up input (*.txt), compiled (*.class) and trace (*.dtrace.gz) files."
rm -f *.txt *.class *.dtrace.gz
echo "Step-6: Done"