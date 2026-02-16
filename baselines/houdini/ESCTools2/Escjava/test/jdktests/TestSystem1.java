//#FLAGS: -Q

import java.io.*;

public class TestSystem1 extends LocalTestCase {

  public void testIn() {
    InputStream in_orig = System.in;
    try {
	assertTrue( in_orig != null);
        InputStream i = new FileInputStream("testfile");
        System.setIn(i);
        assertTrue( System.in == i);
        int c = System.in.read();
        assertTrue( System.in == i);
    } catch (Exception e) {
        // reachable if FileInputStream fails
        System.out.println("EXCEPTION " + e);
    } finally {
	System.setIn(in_orig);
    }
    //@ assert false; // TEST FOR CONSISTENCY

  }

  public void testInNull() {
    InputStream in_orig = System.in;
    try {
        //@ set System.allowNullStreams = true;
	assertTrue( in_orig != null);
	System.setIn(null);
	assertTrue( System.in == null);
        try {
	    int c = System.in.read();  // ERROR
            assertTrue(false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
        }
    } catch (Exception e) {
        assertTrue( false);
        System.out.println("EXCEPTION " + e);
    } finally {
	System.setIn(in_orig);
        //@ set System.allowNullStreams = false;
    }

  }

  public void testInNull2() {
    InputStream in_orig = System.in;
    System.setIn(null);  // ERROR
  }

  public void testOut(/*@ non_null */ PrintStream n) {
    PrintStream out_orig = System.out;
    try {
	assertTrue( out_orig != null);
	//PrintStream n = new PrintStream(new FileOutputStream("testfile"));
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

  public void testOutNull() {
    PrintStream out_orig = System.out;
    try {
        //@ set System.allowNullStreams = true;
	assertTrue( out_orig != null);
	System.setOut(null);
	assertTrue( System.out == null);
        try {
	    System.out.print("");  // ERROR
            assertTrue(false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
        }
    } catch (Exception e) {
        System.out.println("EXCEPTION " + e);
        assertTrue( false);

    } finally {
        System.setOut(out_orig);
        //@ set System.allowNullStreams = false;
    }
  }

  public void testOutNull2() {
    PrintStream out_orig = System.out;
    try {
	System.setOut(null);           // ERROR
    } finally {
        System.setOut(out_orig);
    }
  }


  public void testErr(/*@ non_null */ PrintStream n) {
    PrintStream err_orig = System.err;
    try {
	assertTrue( err_orig != null);
	//PrintStream n = new PrintStream(new FileOutputStream("testfile"));
	System.setErr(n);
	assertTrue( System.err == n);
        System.err.print("");
	assertTrue( System.err == n);
    } catch (Exception e) {
        // reachable if the constructors fail
        System.out.println("EXCEPTION " + e);
    } finally {
        System.setErr(err_orig);
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testErrNull() {
    PrintStream err_orig = System.err;
    try {
        //@ set System.allowNullStreams = true;
	assertTrue( err_orig != null);
	System.setErr(null);
	assertTrue( System.err == null);
        try {
	    System.err.print("");  // ERROR
            assertTrue(false);
        } catch (Exception e) {
            assertTrue( e instanceof NullPointerException);
        }
    } catch (Exception e) {
        System.out.println("EXCEPTION " + e);
        assertTrue( false);

    } finally {
        System.setErr(err_orig);
        //@ set System.allowNullStreams = false;
    }
  }

  public void testErrNull2() {
    PrintStream err_orig = System.err;
    try {
	System.setErr(null);           // ERROR
    } finally {
        System.setErr(err_orig);
    }
  }

}
