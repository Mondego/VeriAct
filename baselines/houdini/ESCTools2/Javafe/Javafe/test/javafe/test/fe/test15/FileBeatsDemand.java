import q.*;

class FileBeatsDemand2 extends Test {
   // Make sure we reference <unnamed>.Test, not q.Test
   int y = Test.x;	// no error (javac is mistaken here)
}

// This overrides q.Test:
class Test { public static int x; }
