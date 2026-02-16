/**
 ** This file tests the various type checks needed specifically for
 ** user inserted C.this expressions.
 **
 ** Other files test things like accessibility and scoping.
 **/

class ThisTest extends ThisTest2 {

    int j = 0;

    // Test refering to same class or its superclass:
    int t1 = ThisTest.this.j + ThisTest.this.i;
    int t2 = ThisTest2.this.i;                 // error (current javac bug)
    static int t3 = ThisTest.this.j;           // error
    int t4 = Inner.this.t1;                    // error

    // Test from 1 level down:
    class Inner {
	int t1 = ThisTest.this.j + ThisTest.this.i;
	int t2 = ThisTest2.this.i;                 // error

    // Test from 2 levels down:
	class Inner2 {
	    int t1 = ThisTest.this.j + ThisTest.this.i;
	    int t2 = ThisTest2.this.i;                 // error
	}
    }
}

class ThisTest2 {
    int i = 0;
}
