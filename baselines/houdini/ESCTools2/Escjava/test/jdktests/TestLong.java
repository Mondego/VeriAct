public class TestLong extends LocalTestCase {

  public void testConstructor() {
    Long i = new Long(7);
    //@ assert \fresh(i);
    assertTrue( i.longValue() == 7 );
    Long j = new Long(-1);
    assertTrue ( i !=j ) ;
    assertTrue ( j.longValue() == -1);
    j = new Long(7L);
    assertTrue ( i !=j ) ;
    assertTrue ( j.longValue() == 7);
//@ assert false; // TEST FOR CONSISTENCY -1
  }

  String s;

  public void testConstructor2() {
    //@ assume Long.parseable("8");
    //@ assume Long.parseable("8L");
    //@ assume Long.parseLong("8") == 8;
    Long i = new Long("8");
    //@ assert \fresh(i);

    i = Long.valueOf("8");

    try { 
      i = new Long(null);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = new Long("");
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = Long.valueOf(null);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = Long.valueOf("");
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    //@ assume s.length() == 0;

    try { 
      i = new Long(s);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = Long.valueOf(s);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }


//@ assert false; // TEST FOR CONSISTENCY - 2
  }

  public void testValue() {
    Long i = new Long(10);
    assertTrue( i.longValue() == 10);
    assertTrue( i.intValue() == 10);
    assertTrue( i.shortValue() == 10);
    assertTrue( i.byteValue() == 10);

//@ assert false; // TEST FOR CONSISTENCY - 3
  }

  public void testCompareTo() {
    Long i = new Long(9);
    Long iii = new Long(9);
    Long ii = new Long(10);
    Object o = new Object();
    Object n = null;
    Long nn = null;

    assertTrue( i.equals(i) );
    assertTrue( !i.equals(ii) );
    assertTrue( i.equals(iii) );
    assertTrue( !i.equals(o) );
    assertTrue( !i.equals(n) );
    assertTrue( !i.equals(nn) );

    assertTrue( i.compareTo(ii) == -1);
    assertTrue( ii.compareTo(i) == 1);
    assertTrue( i.compareTo(i) == 0);
    assertTrue( i.compareTo(iii) == 0);

    try {
      i.compareTo(o);
      assertTrue( false);
    } catch (Exception e) {
      assertTrue( e instanceof ClassCastException );
    }

    try {
      i.compareTo(n);
      assertTrue( false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException );
    }

    try {
      i.compareTo(nn);
      assertTrue( false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException );
    }

//@ assert false; // TEST FOR CONSISTENCY - 4
  }

  public void testValueOf() {
    testValueOf("1"); 

    try {
        Long.valueOf(null);
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        Long.valueOf("");
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        Long.valueOf(null,2);
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        Long.valueOf("",8);
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        //@ assume Long.parseable("1");
	Long.valueOf("1",0);
        assertTrue( false);
    } catch (Exception e) {
        assertTrue ( e instanceof NumberFormatException );
    }
//@ assert false; // TEST FOR CONSISTENCY - 5
  }

  public void testValueOf(String s) {
  //@ assume Long.parseable(s);
    assertTrue( Long.valueOf(s).longValue() == Long.parseLong(s) );
//@ assert false; // TEST FOR CONSISTENCY -6
  }

  //@ requires Long.parseable(s,8);
  public void testValueOf2(String s) {
    assertTrue( Long.valueOf(s,8).longValue() == Long.parseLong(s,8) );
//@ assert false; // TEST FOR CONSISTENCY - 7
  }

  public void testToString() {
    Long i = new Long(-1);
    String s = i.toString();
    assertTrue( Long.parseLong(s) == -1);
    s = Long.toString(-2);
    assertTrue( Long.parseLong(s) == -2);
    s = Long.toString(-2,10);
    assertTrue( Long.parseLong(s,10) == -2);
    s = Long.toString(100,0);
    assertTrue( Long.parseLong(s,10) == 100);
    assertTrue( Long.parseLong(s) == 100);
    s = Long.toString(Long.MAX_VALUE,10);
    assertTrue( Long.parseLong(s) == Long.MAX_VALUE);
    s = Long.toString(Long.MIN_VALUE,10);
    assertTrue( Long.parseLong(s) == Long.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 8a
  }

  public void testToString2() {
    s = Long.toString(-1,16);
    assertTrue( Long.parseLong(s,16) == -1);
    s = Long.toString(Long.MAX_VALUE,16);
    assertTrue( Long.parseLong(s,16) == Long.MAX_VALUE);
    s = Long.toString(Long.MIN_VALUE,16);
    assertTrue( Long.parseLong(s,16) == Long.MIN_VALUE);

    s = Long.toString(-1,8);
    assertTrue( Long.parseLong(s,8) == -1);
    s = Long.toString(Long.MAX_VALUE,8);
    assertTrue( Long.parseLong(s,8) == Long.MAX_VALUE);
    s = Long.toString(Long.MIN_VALUE,8);
    assertTrue( Long.parseLong(s,8) == Long.MIN_VALUE);

//@ assert false; // TEST FOR CONSISTENCY - 8b
  }

// FIXME - how to handle negative strings

  public void testHexString() {
    String s = Long.toHexString(1);
    assertTrue( Long.parseLong(s,16) == 1);
    assertTrue( s.equals(Long.toString(1,16)) );
    s = Long.toHexString(-1);
    //assertTrue( Long.parseLong(s,16) == -1);
    //assertTrue( s.equals(Long.toString(-1,16)) );
    s = Long.toHexString(Long.MAX_VALUE);
    assertTrue( Long.parseLong(s,16) == Long.MAX_VALUE);
    s = Long.toHexString(Long.MIN_VALUE);
    // FIXME assertTrue( Long.parseLong(s,16) == Long.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 9
  }


  public void testOctalString() {
    String s = Long.toOctalString(1);
    assertTrue( Long.parseLong(s,8) == 1);
    assertTrue( s.equals(Long.toString(1,8)) );
    s = Long.toOctalString(-1);
    //assertTrue( (int)Long.parseLong(s,8) == -1);
    //assertTrue( s.equals(Long.toString(-1 + (1L<<32),8)) );
    s = Long.toOctalString(Long.MAX_VALUE);
    assertTrue( Long.parseLong(s,8) == Long.MAX_VALUE);
    s = Long.toOctalString(Long.MIN_VALUE);
    //assertTrue( (int)Long.parseLong(s,8) == Long.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 9b
  }

  public void testBinaryString() {
    String s = Long.toBinaryString(1);
    assertTrue( Long.parseLong(s,2) == 1);
    assertTrue( s.equals(Long.toString(1,2)) );
    s = Long.toBinaryString(-1);
    //assertTrue( (int)Long.parseLong(s,2) == -1);
    //assertTrue( s.equals(Long.toString(-1 + (1L<<32),2)) );
    s = Long.toBinaryString(Long.MAX_VALUE);
    assertTrue( Long.parseLong(s,2) == Long.MAX_VALUE);
    s = Long.toBinaryString(Long.MIN_VALUE);
    //assertTrue( (int)Long.parseLong(s,2) == Long.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 9c
  }
  public void testTYPE() {
// FIXME    //@ assert Long.TYPE == \type(long);
    // FIXME assertTrue( Long.TYPE == long.class );
    assertTrue( Long.MIN_VALUE == 0x8000000000000000L );
    assertTrue( Long.MAX_VALUE == 0x7FFFFFFFFFFFFFFFL );
    assertTrueNP( Long.MAX_VALUE + 1 == Long.MIN_VALUE);

//@ assert false; // TEST FOR CONSISTENCY - 10
  }

/* Needs tests for  - FIXME

flaotValue
doubleValue
getLong
decode
also converting from other radix strings to Long

*/
}
