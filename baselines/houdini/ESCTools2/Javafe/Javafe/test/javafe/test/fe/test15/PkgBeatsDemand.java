package p;

import q.*;

class PkgBeatsDemand2 extends Test {	// no error (javac is wrong here)
   // Make sure we reference p.Test, not q.Test
   int y = Test.x;	// no error (javac is wrong here)
}
