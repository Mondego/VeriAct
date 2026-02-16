import java.io.File;

public class TestFile extends LocalTestCase {

  public void testConstructor() {
    File f = new File("s");

    String n = null;
    File fn = null;

    try {
      new File(n);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NullPointerException);
    }

    try {
      new File(f,n);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NullPointerException);
    }

    try {
      new File(n,n);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NullPointerException);
    }

    java.net.URI uri = null;
    try {
      new File(uri);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NullPointerException);
    }
//@ assert false; // TEST FOR CONSISTENCY

  }

  public void testCompareTo() {
    File f = new File("s");

    String n = null;
    File fn = null;

    int j = f.compareTo(f);
    assertTrue( j == 0);
/*
    try {
      int i = f.compareTo(fn);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NullPointerException);
    }
*/
/*
    try {
      Object on = null;
      f.compareTo(on);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NullPointerException);
    }
*/
    try {
      f.compareTo("a");
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof ClassCastException);
    }
//@ assert false; // TEST FOR CONSISTENCY

  }



}
