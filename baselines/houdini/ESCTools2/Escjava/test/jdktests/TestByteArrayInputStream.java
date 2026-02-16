import java.io.*;

public class TestByteArrayInputStream extends LocalTestCase {

  byte[] a;

  public void setUp() {
    a = new byte[20];
    for (int i=0; i<a.length; ++i) a[i] = (byte)i;
  }

  public void test1() {
    //@ assume a != null && a.length == 20;
    ByteArrayInputStream b = new ByteArrayInputStream(a);
    assertTrue ( b.available() == 20);
    // This is the earliest possible location of an assert false that
    // does *not* get caught.  It is not an invariant consistency
    // problem because checking just after a call to a method foo()
    // with no spec does not cause the missed warning.  
    int c = b.read();
    //@ assert false;
    assertTrue( c == a[0] );
    byte[] q = new byte[10];
    b.skip(4);
    //@ assert b.pos == 5;
    try { 
      int n = b.read(q); 
      //@ assert b.pos == 15;
      //@ assert n == 10;
      assertTrue( q[0] == a[5] );
      assertTrue( q[9] == a[14] );
    } catch (Exception e){}

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void test2() {
    //@ assume a != null && a.length == 20;
    ByteArrayInputStream b = new ByteArrayInputStream(a,2,12);
    assertTrue ( b.available() == 12);
    int c = b.read();
    assertTrue( c == a[2] );
    byte[] q = new byte[10];
    q[6] = (byte)-1;
    q[9] = (byte)-1;
    b.skip(4);
    //@ assert b.pos == 7;
    try { 
      int n = b.read(q); 
      //@ assert b.pos == 14;
      //@ assert n == 7;
      assertTrue( q[0] == a[7] );
      assertTrue( q[6] == a[13] );
      assertTrue( q[9] == -1 );
    } catch (Exception e){}

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void test2a() {
    //@ assume a != null && a.length == 20;
    ByteArrayInputStream b = new ByteArrayInputStream(a,2,12);
    assertTrue ( b.available() == 12);
    int c = b.read();
    assertTrue( c == a[2] );
    byte[] q = new byte[10];
    q[0] = (byte)-1;
    q[6] = (byte)-1;
    q[9] = (byte)-1;
    b.skip(4);
    //@ assert b.pos == 7;
    try { 
      int n = b.read(q,2,5); 
      //@ assert b.pos == 12;
      //@ assert n == 5;
      assertTrue( q[0] == (byte)-1 );
      assertTrue( q[2] == a[7] );
      assertTrue( q[6] == a[11] );
      assertTrue( q[9] == (byte)-1 );
    } catch (Exception e){}

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testMark() {
    //@ assume a != null && a.length == 20;
    ByteArrayInputStream b = new ByteArrayInputStream(a);
    b.skip(5);
    b.reset();
    //@ assert b.pos == 0;
    int c = b.read();
    assertTrue ( c == a[0]);
    b.reset();
    b.skip(5);
    b.mark(0);
    b.skip(6);
    b.reset();
    //@ assert b.pos == 5;
    c = b.read();
    assertTrue ( c == a[5]);
//@ assert false; // TEST FOR CONSISTENCY

  }

  public void testNoCopy() {
    //@ assume a != null && a.length == 20;
    a[0] = 10;
    ByteArrayInputStream b = new ByteArrayInputStream(a);
    a[0] = 17;
    int c = b.read();
    assertTrue ( c == 17);

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testBase() {
    //@ assume a != null && a.length == 20;
    InputStream b = new ByteArrayInputStream(a);
    //@ assume b.markSupported();
    try {
    b.mark(0);
    int c = b.read();
    b.reset();
    int cc = b.read();
    assertTrue( c == cc);
    } catch (Exception e) {}

//@ assert false; // TEST FOR CONSISTENCY
  }

}
