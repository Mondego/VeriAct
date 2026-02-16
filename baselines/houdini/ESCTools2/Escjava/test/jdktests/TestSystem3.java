//#FLAGS: -Q

import java.io.*;
import java.util.Properties;
import java.util.Enumeration;

public class TestSystem3 extends LocalTestCase {

  String arbitrary = "java.home"; // an arbitrary non-null String

  public void testProperties3() {
    Properties prop = (Properties)System.getProperties().clone();
    try {
        try {
	    System.getProperty(null);
            assertTrue (false);
        } catch (Exception e) {
            assertTrue (e instanceof NullPointerException);
        }
/*
        try {
	    System.getProperty(null,"a");
            assertTrue (false);
        } catch (Exception e) {
            assertTrue (e instanceof NullPointerException);
        }
        try {
	    System.getProperty("a",null);
            assertTrue (true);
        } catch (Exception e) {
            assertTrue (false);
        }
        try {
	    System.setProperty(null,"a");
            assertTrue (false);
        } catch (Exception e) {
            assertTrue (e instanceof NullPointerException);
        }
        try {
	    System.setProperty("a",null);
            assertTrue (false);
        } catch (Exception e) {
            assertTrue (e instanceof NullPointerException);
        }
*/
    } finally {
        System.setProperties(prop);
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testProperties2() {
    Properties prop = System.getProperties();
    String s = System.getProperty("java.home");
    assertTrue (s != null);  // NOT PROVABLE
    String ss = System.getProperty("java.home","zzz");
    assertTrue (s == ss); // NOT PROVABLE
    s = System.getProperty("xxxxxx");
    assertTrue (s == null); // NOT PROVABLE
    s = System.getProperty("xxxxxx","yyy");
    assertTrue (s == "yyy"); // NOT PROVABLE
    String q = "java.home";
    assertTrue( System.getProperty(q) == System.getProperties().getProperty(q));    String qq = "xxx";
    try {
      assertTrue( prop.containsKey(q) || prop.getProperty(q,qq) == qq );
    } catch (Exception e) {}
    //@ assert false; // TEST FOR CONSISTENCY
  }

  public void testProperties() {
    //Properties prop = (Properties)System.getProperties().clone();
    Properties prop = System.getProperties();
    try {
	Properties p = new Properties();
	System.setProperties(p);
//@ assert System.systemProperties == p;
    //@ assert false; // TEST FOR CONSISTENCY
	Properties pp = System.getProperties();
/*
        //@ assert System.systemProperties == pp;
        assertTrue ( p == pp);

	System.setProperties(null);
	pp = System.getProperties();
        assertTrue(pp != null);
        System.setProperties(p);
        assertTrue ( System.getProperties() == p);
        System.setProperties(null);
        assertTrue (System.getProperties().equals(pp));
*/
    } finally {
//	System.setProperties(prop);
    }
    //@ assert false; // TEST FOR CONSISTENCY
  }
  // FIXME - need to test properties calls with SecurityExceptions

  // TODO: No tests for loadLibrary, load, mapLibraryName
  // TODO: No tests for exit, gc, runFinalization

// FIXME - test whether we are making copies on getting or setting system properties

   // FIXME test arraycopy for all primitive types

static class System {

    //@ public model static non_null Properties systemProperties;
    //@    initially systemProperties.containsKey("java.home");
    //@    initially systemProperties.getProperty("java.home") != null;
    //@    initially !systemProperties.containsKey("xxxxxx");
    //@ public model static non_null Properties nullSystemProperties;

    /*@ public normal_behavior
      @    ensures \result == systemProperties;
      @    //signals (SecurityException) (* if access is not permitted *);
      @*/
    //-@ function
    public /*@ pure @*/ static /*@ non_null @*/ Properties getProperties();

    /*@ public behavior
      @    requires props == null;
      @    assignable systemProperties;
      @    ensures systemProperties.equals(nullSystemProperties);
      @    signals_only SecurityException;
      @    signals (SecurityException) (* if access is not permitted *) &&
                                       \not_modified(systemProperties);
      @ also public behavior
      @    requires props != null;
      @    assignable systemProperties;
      @    ensures systemProperties == props;
      @    signals_only SecurityException;
      @    signals (SecurityException) (* if access is not permitted *) &&
                                       \not_modified(systemProperties);
      @*/
    public static void setProperties(Properties props);

// FIXME - conflict between exceptions
    //@ public behavior
    //@   requires key != null;
    //@   ensures \result == getProperties().getProperty(key,def);
    //@
    //@    signals (SecurityException) (* if access is not permitted *);
    //@ also public exceptional_behavior
    //@   requires key == null;
    //@   signals_only NullPointerException;
    public /*@ pure @*/ static String getProperty(String key, String def);


// FIXME - conflict between exceptions
    /*@ public behavior
      @    requires key != null && value != null;
      @    assignable systemProperties;
      @    signals (SecurityException) (* if access is not permitted *);
      @ also public exceptional_behavior
      @   requires key == null || value == null;
      @   signals_only NullPointerException;
      @*/
    public static String setProperty(String key, String value);

// FIXME - conflict between exceptions
    //@ public behavior
    //@   requires key != null;
    //@   ensures \result == getProperties().getProperty(key);
    //@   signals (SecurityException) (* if access is not permitted *);
    //@ also public exceptional_behavior
    //@   requires key == null;
    //@   signals_only NullPointerException;
    public /*@ pure @*/ static String getProperty(String key);

    private System();
}
}
