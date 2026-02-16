#!/bin/bash
# author : Clement

# this script should be removed and the job should be done in another/nicer way

cd java/escjava/vcGeneration

./division-j-file.pl TFunction.j 
./division-j-file.pl TLiteral.j

cd coq

./PreludeGenerator.sh
