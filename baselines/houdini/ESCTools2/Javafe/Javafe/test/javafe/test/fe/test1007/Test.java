
public class Test {
    public static void test(int j) {
	switch(j) {
	case A.i:
	    // case A.B.C.i:
	    throw new RuntimeException();
	}
	A.B.C.getI();
    }
}
