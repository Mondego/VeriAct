public class TestBoolean extends LocalTestCase {

  public void testConstructors() {
    assertTrue( new Boolean(true).booleanValue() );
    assertTrue( !new Boolean(false).booleanValue() );
    assertTrue( new Boolean("true").booleanValue() );
    assertTrue( new Boolean("True").booleanValue() );
    assertTrue( new Boolean("TRUE").booleanValue() );

    assertTrue( !new Boolean("T").booleanValue() );
    assertTrue( !new Boolean("1").booleanValue() );
    assertTrue( !new Boolean("t").booleanValue() );
    assertTrue( !new Boolean(null).booleanValue() );
    assertTrue( !new Boolean("").booleanValue() );
//@ assert false; // TEST FOR CONSISTENCY
  }
  public void testConstructors2() {
    assertTrue( !new Boolean("false").booleanValue() );
    assertTrue( !new Boolean("False").booleanValue() );
    assertTrue( !new Boolean("FALSE").booleanValue() );
    assertTrue( !new Boolean("F").booleanValue() );
    assertTrue( !new Boolean("f").booleanValue() );
    assertTrue( !new Boolean("0").booleanValue() );
    assertTrue( !new Boolean("XXXX").booleanValue() ); // FIXME - NOT PROVABLE

    assertTrue ( new Boolean(true) != Boolean.TRUE);
    assertTrue ( new Boolean(false) != Boolean.FALSE);
//@ assert false; // TEST FOR CONSISTENCY
  }
 
  public void testConstants() {
    assertTrue (Boolean.TRUE.booleanValue());
    assertTrue (!Boolean.FALSE.booleanValue());
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testValueOf() {
    Boolean b;
    b = Boolean.valueOf(true);
    assertTrue (b.booleanValue()); 
    assertTrue(b == Boolean.TRUE);
    b = Boolean.valueOf("true");
    assertTrue (b.booleanValue()); 
    assertTrue(b == Boolean.TRUE);
    b = Boolean.valueOf("True");
    assertTrue (b.booleanValue()); 
    assertTrue(b == Boolean.TRUE);
    b = Boolean.valueOf("TRUE");
    assertTrue (b.booleanValue()); 
    assertTrue(b == Boolean.TRUE);

    b = Boolean.valueOf(false);
    assertTrue (!b.booleanValue()); 
    assertTrue(b == Boolean.FALSE);
    b = Boolean.valueOf("T");
    assertTrue (!b.booleanValue());   
    assertTrue (b == Boolean.FALSE);
    b = Boolean.valueOf("false");
    assertTrue (!b.booleanValue()); 
    assertTrue(b == Boolean.FALSE);
    b = Boolean.valueOf("False");
    assertTrue (!b.booleanValue()); 
    assertTrue(b == Boolean.FALSE);
    b = Boolean.valueOf("FALSE");
    assertTrue (!b.booleanValue()); 
    assertTrue(b == Boolean.FALSE);
    b = Boolean.valueOf("");
    assertTrue (!b.booleanValue()); 
    assertTrue(b == Boolean.FALSE);
    b = Boolean.valueOf(null);
    assertTrue (!b.booleanValue()); 
    assertTrue(b == Boolean.FALSE);
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testString() {
    String stt = "true";
    String sff = "false";
    String st = Boolean.TRUE.toString();
    String sf = Boolean.FALSE.toString();
    assertTrue (st.equals(stt));
    assertTrue (sf.equals(sff));
    assertTrue (st == stt);
    assertTrue (sf == sff);
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testHashCode() {
    int ht = Boolean.TRUE.hashCode();
    int hf = Boolean.FALSE.hashCode();
    assertTrue (ht != hf);
    int htt = new Boolean(true).hashCode();
    int hff = new Boolean(false).hashCode();
    assertTrue (ht == htt);
    assertTrue (hf == hff);
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testEquals() {
    Object o = new Object();
    assertTrue ( !Boolean.TRUE.equals(Boolean.FALSE) );
    assertTrue (  Boolean.TRUE.equals(Boolean.TRUE) );
    assertTrue (  Boolean.TRUE.equals(new Boolean(true)) );
    assertTrue ( !Boolean.TRUE.equals(new Boolean(false)) );
    assertTrue ( !Boolean.TRUE.equals(null) );
    assertTrue ( !Boolean.TRUE.equals(o) );
//@ assert false; // TEST FOR CONSISTENCY
  }
  public void testEquals2() {
    Object o = new Object();
    assertTrue (  Boolean.FALSE.equals(Boolean.FALSE) );
    assertTrue ( !Boolean.FALSE.equals(Boolean.TRUE) );
    assertTrue ( !Boolean.FALSE.equals(new Boolean(true)) );
    assertTrue (  Boolean.FALSE.equals(new Boolean(false)) );
    assertTrue ( !Boolean.FALSE.equals(null) );
    assertTrue ( !Boolean.FALSE.equals(o) );
//@ assert false; // TEST FOR CONSISTENCY
  }

  // FIXME - these need better handling of java.util.Properties and of
  // interned Strings to be provable
  public void testGetProperty() throws RuntimeException {
    assertTrue (!Boolean.getBoolean(null));
    assertTrueNP (!Boolean.getBoolean(""));
    System.setProperty("a","TRUE");
    assertTrueNP (Boolean.getBoolean("a"));
    assertTrueNP (!Boolean.getBoolean("A")); // keys are case-sensitive
    assertTrueNP (!Boolean.getBoolean("java.home"));
    System.getProperties().remove("a"); 
    assertTrueNP (!Boolean.getBoolean("a"));
//@ assert false; // TEST FOR CONSISTENCY
  }

  // clone not implemented
}
