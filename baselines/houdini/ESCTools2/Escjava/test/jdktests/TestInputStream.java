import java.io.*;

public class TestInputStream extends LocalTestCase {

  public InputStream s;

  public void setUp() {
    try {
      s = new FileInputStream("TestInputStream.java");
    } catch (Exception e) {
      throw new RuntimeException(); //@ nowarn Exception;
    }
  } //@ nowarn Exception;



  //@ requires s != null && s.isOpen;
  public void testOpen() {
    //@ assume s.availableBytes > 10;
    try {
	s.read();
	s.close();
    } catch (Exception e) {
        // might happen;
    }
    try {
        s.close();
        assertTrue( true) ;
    } catch (Exception e) {
    }
    //@ assert !s.isOpen;
    try {
      s.read();
      assertTrue( false) ;
    } catch (Exception e) {
      assertTrue( e instanceof IOException);
    }
//@ assert false; // TEST FOR CONSISTENCY
  }

// FIXME - no tests of reset, mark, markSupported, 
}
