//#FLAGS: -Q

import java.io.*;

public class TestSystem4 extends LocalTestCase {


  public void testOut() {
    PrintStream out_orig = System.out;
    try {
	assertTrue( out_orig != null);
        OutputStream f = new FileOutputStream("testfile");
	PrintStream n = new PrintStream(f);
	System.setOut(n);
	assertTrue( System.out == n);
        System.out.print("");
	assertTrue( System.out == n);
    } catch (Exception e) {
        // reachable if the constructors fail
        System.out.println("EXCEPTION " + e);
    } finally {
        System.setOut(out_orig);
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }

}
