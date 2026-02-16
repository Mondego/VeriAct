public class TestStringBuffer extends LocalTestCase {

  public void testConstructor() {
    StringBuffer sb = new StringBuffer();
    //@ assert sb.accumulatedString.equals("");
    assertTrue( sb.toString().equals("") );

    sb =  new StringBuffer(10);
    //@ assert sb.accumulatedString.equals("");
    assertTrue( sb.toString().equals("") );

    String s = "ASDF";
    sb = new StringBuffer(s);
    //@ assert sb.accumulatedString.equals(s);
    assertTrue( sb.toString().equals(s) );

    try {
       sb = new StringBuffer(null);
       assertTrue( false );
    } catch (Exception e) {
       assertTrue( e instanceof NullPointerException );
    }
    
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testLength() {
    String s = "ASDF";
    assertTrue( s.length() == 4);
    StringBuffer sb = new StringBuffer(s);
    assertTrue( sb.length() == 4);
    assertTrue( sb.toString().length() == 4);
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testCharAt() {
    String s = "ASDF";
    assertTrue( s.length() == 4);
    StringBuffer sb = new StringBuffer(s);
    char c = sb.charAt(3);
    assertTrue( sb.charAt(2) == s.charAt(2));

    try {
        sb.charAt(-1);
        assertTrue (false);
    } catch (Exception e) {
        assertTrue( e instanceof StringIndexOutOfBoundsException );
    }

    try {
        sb.charAt(4);
        assertTrue (false);
    } catch (Exception e) {
        assertTrue( e instanceof StringIndexOutOfBoundsException );
    }

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testReverse() {
    String s = "ASDF";
    assertTrue( s.length() == 4);
    StringBuffer sb = new StringBuffer(s);
    int k = sb.length();
    char c = sb.charAt(0);
    StringBuffer sbb = sb.reverse();
    assertTrue( sbb == sb);
    assertTrue( sb.length() == k);
    assertTrue( c == sb.charAt(3));
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testToString() {
    String s = "ASDF";
    assertTrue( s.length() == 4);
    StringBuffer sb = new StringBuffer(s);

    assertTrue( sb.toString().equals( s.toString() ) );

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testSetLength() {
    String s = "ASDF";
    assertTrue( s.length() == 4);
    StringBuffer sb = new StringBuffer(s);
    assertTrue( sb.length() == 4);

     sb.setLength(2);
    assertTrue( sb.length() == 2);
    sb.setLength(6);
    assertTrue( sb.length() == 6);
    sb.setLength(0);
     assertTrue( sb.length() == 0);

   try {
      sb.setLength(-1);
      assertTrue( false );
   } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException );
   }
  
//@ assert false; // TEST FOR CONSISTENCY
  }

// getChars
// indexOf*2
// lastIndexOf*2
// replace
// subSequence
// capacity
// ensureCapacity
// deleteCharAt
// append, insert
// setCharAt
// delete
// substring*2



  // test the various appens

}
