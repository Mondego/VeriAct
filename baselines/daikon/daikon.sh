#!/bin/bash

if [ $# -ne 2 ]; then
  echo "Usage: $0 <class_name> <integer> "
  exit 1
fi


class_name=$1 
test_class_name="${class_name}Test"
numberOfTest=$2

echo $numberOfTest

echo "Step-1: Compile the program with the -g switch to enable debugging symbols."
javac -g *.java
echo "Step-1: Done"


echo "Step-3-a: Run the program under the control of the Chicory front"
counter=0
while [ $counter -lt $numberOfTest ]; do
  echo "Generate comparability information and trace file for test: $counter"
  java -cp .:$DAIKONDIR/daikon.jar daikon.DynComp --decl-file=Test_$counter.decls-DynComp $test_class_name < $counter.txt
  echo $counter
  java -cp .:$DAIKONDIR/daikon.jar daikon.Chicory --dtrace-file=Test_$counter.dtrace.gz --comparability-file=Test_$counter.decls-DynComp $test_class_name <$counter.txt
  ((counter++))
done
echo "Step-3-a: Done"


echo "Step-3-b: Run Daikon on the trace files."
#java -cp .:$DAIKONDIR/daikon.jar daikon.Daikon --config inv-config.config --suppress_redundant Test*.dtrace.gz
java -cp .:$DAIKONDIR/daikon.jar daikon.Daikon --suppress_redundant Test*.dtrace.gz
echo "Step-3-b: Done"


echo "Step-4: Annotate the source with JML"
java -cp .:$DAIKONDIR/daikon.jar daikon.tools.jtb.Annotate --no_reflection --format jml Test*.inv.gz $class_name.java
echo "Step-4: Done"

echo "Step-5: Annotate the source with ESC/Java2 with the same invariants."
java -cp .:$DAIKONDIR/daikon.jar daikon.tools.jtb.Annotate --no_reflection --format esc Test*.inv.gz $class_name.java
echo "Step-5: Done"

echo "Step-6: Clean up the intermediate files."
rm -f *.txt *.class *.decls-DynComp *.dtrace.gz
echo "Step-6: Done"