public class TestInteger extends LocalTestCase {

  public void testConstructor() {
    Integer i = new Integer(7);
    //@ assert \fresh(i);
    assertTrue( i.intValue() == 7 );
    Integer j = new Integer(-1);
    assertTrue ( i !=j ) ;
    assertTrue ( j.intValue() == -1);
    j = new Integer(7);
    assertTrue ( i !=j ) ;
    assertTrue ( j.intValue() == 7);
//@ assert false; // TEST FOR CONSISTENCY -1
  }

  String s;

  public void testConstructor2() {
    //@ assume Integer.parseable("8");
    //@ assume Integer.parseInt("8") == 8;
    Integer i = new Integer("8");
    //@ assert \fresh(i);

    i = Integer.valueOf("8");

    try { 
      i = new Integer(null);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = new Integer("");
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = Integer.valueOf(null);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = Integer.valueOf("");
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    //@ assume s.length() == 0;

    try { 
      i = new Integer(s);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }

    try { 
      i = Integer.valueOf(s);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof NumberFormatException);
    }


//@ assert false; // TEST FOR CONSISTENCY - 2
  }

  public void testValue() {
    Integer i = new Integer(10);
    assertTrue( i.intValue() == 10);
    assertTrue( i.longValue() == 10);
    assertTrue( i.shortValue() == 10);
    assertTrue( i.byteValue() == 10);

//@ assert false; // TEST FOR CONSISTENCY - 3
  }

  public void testCompareTo() {
    Integer i = new Integer(9);
    Integer iii = new Integer(9);
    Integer ii = new Integer(10);
    Object o = new Object();
    Object n = null;
    Integer nn = null;

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
        Integer.valueOf(null);
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        Integer.valueOf("");
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        Integer.valueOf(null,2);
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        Integer.valueOf("",8);
        assertTrue(false);
    } catch (Exception e) {
        assertTrue( e instanceof NumberFormatException );
    }

    try {
        //@ assume Integer.parseable("1");
	Integer.valueOf("1",0);
        assertTrue( false);
    } catch (Exception e) {
        assertTrue ( e instanceof NumberFormatException );
    }
//@ assert false; // TEST FOR CONSISTENCY - 5
  }

  public void testValueOf(String s) {
  //@ assume Integer.parseable(s);
    assertTrue( Integer.valueOf(s).intValue() == Integer.parseInt(s) );
//@ assert false; // TEST FOR CONSISTENCY -6
  }

  //@ requires Integer.parseable(s,8);
  public void testValueOf2(String s) {
    assertTrue( Integer.valueOf(s,8).intValue() == Integer.parseInt(s,8) );
//@ assert false; // TEST FOR CONSISTENCY - 7
  }

  public void testToString() {
    Integer i = new Integer(-1);
    String s = i.toString();
    assertTrue( Integer.parseInt(s) == -1);
    s = Integer.toString(-2);
    assertTrue( Integer.parseInt(s) == -2);
    s = Integer.toString(-2,10);
    assertTrue( Integer.parseInt(s,10) == -2);
    s = Integer.toString(100,0);
    assertTrue( Integer.parseInt(s,10) == 100);
    assertTrue( Integer.parseInt(s) == 100);
    s = Integer.toString(Integer.MAX_VALUE,10);
    assertTrue( Integer.parseInt(s) == Integer.MAX_VALUE);
    s = Integer.toString(Integer.MIN_VALUE,10);
    assertTrue( Integer.parseInt(s) == Integer.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 8a
  }

  public void testToString2() {
    String s = Integer.toString(-1,16);
    assertTrue( Integer.parseInt(s,16) == -1);
    s = Integer.toString(Integer.MAX_VALUE,16);
    assertTrue( Integer.parseInt(s,16) == Integer.MAX_VALUE);
    s = Integer.toString(Integer.MIN_VALUE,16);
    assertTrue( Integer.parseInt(s,16) == Integer.MIN_VALUE);

    s = Integer.toString(-1,8);
    assertTrue( Integer.parseInt(s,8) == -1);
    s = Integer.toString(Integer.MAX_VALUE,8);
    assertTrue( Integer.parseInt(s,8) == Integer.MAX_VALUE);
    s = Integer.toString(Integer.MIN_VALUE,8);
    assertTrue( Integer.parseInt(s,8) == Integer.MIN_VALUE);

//@ assert false; // TEST FOR CONSISTENCY - 8b
  }

  public void testHexString() {
    String s = Integer.toHexString(1);
    assertTrue( Integer.parseInt(s,16) == 1);
    assertTrue( s.equals(Integer.toString(1,16)) );
    s = Integer.toHexString(-1);
    assertTrue( (int)Long.parseLong(s,16) == -1);
    assertTrue( s.equals(Long.toString(-1 + (1L<<32),16)) );
    s = Integer.toHexString(Integer.MAX_VALUE);
    assertTrue( Integer.parseInt(s,16) == Integer.MAX_VALUE);
    s = Integer.toHexString(Integer.MIN_VALUE);
    assertTrue( (int)Long.parseLong(s,16) == Integer.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 9
  }

  public void testOctalString() {
    String s = Integer.toOctalString(1);
    assertTrue( Integer.parseInt(s,8) == 1);
    assertTrue( s.equals(Integer.toString(1,8)) );
    s = Integer.toOctalString(-1);
    assertTrue( (int)Long.parseLong(s,8) == -1);
    assertTrue( s.equals(Long.toString(-1 + (1L<<32),8)) );
    s = Integer.toOctalString(Integer.MAX_VALUE);
    assertTrue( Integer.parseInt(s,8) == Integer.MAX_VALUE);
    s = Integer.toOctalString(Integer.MIN_VALUE);
    assertTrue( (int)Long.parseLong(s,8) == Integer.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 9b
  }

  public void testBinaryString() {
    String s = Integer.toBinaryString(1);
    assertTrue( Integer.parseInt(s,2) == 1);
    assertTrue( s.equals(Integer.toString(1,2)) );
    s = Integer.toBinaryString(-1);
    assertTrue( (int)Long.parseLong(s,2) == -1);
    assertTrue( s.equals(Long.toString(-1 + (1L<<32),2)) );
    s = Integer.toBinaryString(Integer.MAX_VALUE);
    assertTrue( Integer.parseInt(s,2) == Integer.MAX_VALUE);
    s = Integer.toBinaryString(Integer.MIN_VALUE);
    assertTrue( (int)Long.parseLong(s,2) == Integer.MIN_VALUE);
//@ assert false; // TEST FOR CONSISTENCY - 9c
  }

  public void testTYPE() {
// FIXME    //@ assert Integer.TYPE == \type(int);
    // FIXME assertTrue( Integer.TYPE == int.class );
    assertTrue( Integer.MIN_VALUE == 0x80000000 );
    assertTrue( Integer.MAX_VALUE == 0x7FFFFFFF );
    assertTrueNP( Integer.MAX_VALUE + 1 == Integer.MIN_VALUE);

//@ assert false; // TEST FOR CONSISTENCY - 10
  }

  public void testGetInteger() {
    String s = "123";
/*  FIXME - need to be able to reason about getProperty better
    try { System.setProperty("A",s); } catch (SecurityException e) {}
    // @ assert System.getProperty("A") == s;
    //@ assert System.getProperty("A") != null;
    //@ assert "A" != null;
    //@ assert ! "A".equals("");
    //@ assume Integer.decodable(s);
    Integer i = Integer.getInteger("A");
    assertTrue( i.toString().equals(s) );

    try { System.setProperty("A","#123"); } catch (SecurityException e) {}
    System.setProperty("A","#123");
    i = Integer.getInteger("A");
    assertTrueNP( Integer.toString(i.intValue(),16).equals("123") );
*/
    
//@ assert false; // TEST FOR CONSISTENCY - 11
  }

/* Needs tests for  - FIXME

floatValue
doubleValue
getInteger
decode
also conversions from other radix strings

*/
}
