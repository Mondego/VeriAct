#!/bin/sh
f=../../../../../docs/Escjava2-Logics/coq/defs.v

cat Prelude.head > Prelude.java
cat $f | sed 's/\\/\\\\/g' | \
sed 's/"/\\"/g' | \
sed 's/^/	   out.println("/' |\
 sed 's/$/");/' >> Prelude.java
cat Prelude.tail >> Prelude.java

