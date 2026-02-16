import java.util.Properties;

public class TestProperties extends LocalTestCase {

  public void testConstructor() {
    Properties p = new Properties();
    assertTrue (p.isEmpty() );
    //@ assert !p.containsKey("x");
    assertTrue ( p.getProperty("x") == null);

    String a = "a";
    String b = "b";
    //@ assume !a.equals(b); // FIXME - would like to prove
    //@ assert !p.content.hasMap(a);
    //@ assert !p.content.hasMap(b);
    p.setProperty(a,b);
    //@ assert !a.equals(b); // FIXME - would like to prove
    //@ assert (\forall String s; p.content.hasMap(s) ==> a.equals(s));
    //@ assert p.content.hasMap(a);
    //@ assert !p.content.hasMap(b);
    //@ assert p.content.maps(a) == b;
    try {
    assertTrue (p.size() == 1);
    assertTrue( p.containsKey(a) );
    assertTrue( !p.containsKey(b) );
    //@ assert a != null;
    assertTrue( p.getProperty(a) == b );
    assertTrue( p.getProperty(b) == null );

    Properties pp = new Properties(p);
    assertTrue( pp.size() == 0);
    assertTrue( !pp.containsKey(a) );
    assertTrue( !pp.containsKey(b) );
    assertTrue( pp.getProperty(a) == b );
    assertTrue( pp.getProperty(b) == null );
    //@ assert false; // TEST FOR CONSISTENCY
    } catch (Exception e) {}
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testNull() {
    Properties p = new Properties();
    try {
      Properties pp = new Properties(null);
      assertTrue( true );
    } catch (Exception e) {
      assertTrue( false );
    }
    try {
      p.getProperty(null);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException);
    }
    try {
      p.getProperty(null,null);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException);
    }
    try {
      p.getProperty("a",null);
      assertTrue(true);
    } catch (Exception e) {
      assertTrue( false );
    }
    try {
      p.setProperty(null,"b");
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException);
    }
    try {
      p.setProperty(null,null);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException);
    }
    try {
      p.setProperty("b",null);
      assertTrue(false);
    } catch (Exception e) {
      assertTrue( e instanceof NullPointerException);
    }
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testGet() {
    Properties p = new Properties();
    String s = "a";
    String ss = p.getProperty("x",s);
    assertTrue( s == ss);

    // Make sure that a null defaults is ok.
    p = new Properties(null);
    s = "a";
    ss = p.getProperty("x");
    assertTrue( ss == null);
    ss = p.getProperty("x",s);
    assertTrue( s == ss);
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testSet() {
    Properties p = new Properties();
    String s = "a";
    String ss = "b";
    p.setProperty(s,ss);
    assertTrue( p.getProperty(s) == ss);
//@ assert false; // TEST FOR CONSISTENCY
  }

  // This test shows that the defaults Properties of a Properties object is
  // backed by the supplied OBJECT not a copy of the supplied object.
  public void testDef() {
    String a = "a";
    String b = "b";
    String c = "c";
    String d = "d";
    Properties p = new Properties();
    //@ assert !c.equals(a);
    p.setProperty(a,b);
    assertTrue (p.getProperty(a) == b);
    assertTrue (p.getProperty(c) == null);
    Properties pp = new Properties(p);
    //@ assert !c.equals(a);
    assertTrue (p.getProperty(a) == b);
    assertTrue (p.getProperty(c) == null);
    assertTrue( pp.getProperty(c) == null); 
    p.setProperty(c,d);
    //@ assert p != pp;
    //@ assert !pp.content.hasMap(c);
    //@ assert pp.defaults == p;
    //@ assert pp.defaults.getProperty(c) == d;
    //@ assert p.defaults == null;
    assertTrue( pp.getProperty(c) == d);
//@ assert false; // TEST FOR CONSISTENCY
  }

  // The same test with fewer helpful steps
  public void testDef2() {
    String a = "a";
    String b = "b";
    String c = "c";
    String d = "d";
    Properties p = new Properties();
    p.setProperty(a,b);
    Properties pp = new Properties(p);
    //@ assert !c.equals(a);
    assertTrue( pp.getProperty(c) == null);
    p.setProperty(c,d);
    assertTrue( pp.getProperty(c) == d);
//@ assert false; // TEST FOR CONSISTENCY
  }
}
