

/*
 * This file is part of the Esc/Java2 plugin project.
 * Copyright 2004 David R. Cok
 * 
 * Created on Feb 13, 2005
 */

/**
 * TODO Fill in class description
 * 
 * @author David R. Cok
 */
public class TestThrowable extends LocalTestCase {
  
  /** getMessage() returns what was supplied in the constructor
      getMessage() returns null if no string supplied in constructor
  */
  public void testMessage() {
    String s = "abcd";
    Throwable t = new Throwable("THR");
    assertTrue ((new Throwable()).getMessage() == null);
    assertTrue ((new Throwable((String)null)).getMessage() == null);
    assertTrue ((new Throwable(s)).getMessage() == s);
    assertTrue ((new Throwable(s,t)).getMessage() == s);
    assertTrue ((new Throwable(t)).getMessage() != null);
    assertTrueNP ((new Throwable(t)).getMessage().equals("java.lang.Throwable: THR"));
    assertTrue ((new Throwable(s,null)).getMessage() == s);
    assertTrue ((new Throwable(null,null)).getMessage() == null);
    assertTrue ((new Throwable((Throwable)null)).getMessage()==null);
  }

  public void testToString() {
    String s = "abcd";
    Throwable t = new Throwable("THR");
    assertTrue ((new Throwable()).toString() != null);
    assertTrueNP ((new Throwable()).toString().equals("java.lang.Throwable"));
    assertTrue ((new Throwable((String)null)).toString() != null);
    assertTrueNP ((new Throwable((String)null)).toString().equals("java.lang.Throwable"));
    assertTrue ((new Throwable(s)).toString() != null);
    assertTrueNP ((new Throwable(s)).toString().equals("java.lang.Throwable: " + s));
    assertTrue ((new Throwable(s,t)).toString() != null);
    assertTrueNP ((new Throwable(s,t)).toString().equals( "java.lang.Throwable: " + s));
    assertTrue ((new Throwable(t)).toString() != null);
    assertTrueNP ((new Throwable(t)).toString().equals("java.lang.Throwable: java.lang.Throwable: THR"));
    assertTrue ((new Throwable(s,null)).toString() != null);
    assertTrueNP ((new Throwable(s,null)).toString().equals("java.lang.Throwable: " + s));
    assertTrue ((new Throwable(null,null)).toString() != null);
    assertTrueNP ((new Throwable(null,null)).toString().equals("java.lang.Throwable"));
    assertTrue ((new Throwable((Throwable)null)).toString()!=null);
    assertTrueNP ((new Throwable((Throwable)null)).toString().equals("java.lang.Throwable"));
  }
  
  /** getCause() returns what was supplied in the constructor,
   *  or null
   *
   */
  public void testCause() {
    String s = "abcd";
    Throwable t = new Throwable("THR");
    assertTrue ((new Throwable()).getCause() == null);
    assertTrue ((new Throwable(s)).getCause() == null);
    assertTrue ((new Throwable(t)).getCause() == t);
    assertTrue ((new Throwable(s,t)).getCause() == t);
    assertTrue ((new Throwable((Throwable)null)).getCause() == null);
    assertTrue ((new Throwable(s,null)).getCause() == null);
    assertTrue ((new Throwable(null,null)).getCause() == null);
  }
  
  /** Tests that getCause() returns what was supplied in
   *  initCause()
   *
   */
  public void testInitCause() {
    String s = "abcd";
    Throwable t = new Throwable("THR");
    Throwable tt = new Throwable();
    Throwable ttt = tt.initCause(t);
    assertTrue (tt.getCause() == t);
    assertTrue (ttt == tt);
  }
  
  /** Once initCause is called, you can't call it again,
   *  even if the argument is null
   *
   */
  public void testDuplicateInitCause() {
    String s = "abcd";
    Throwable t = new Throwable("THR");
    Throwable tt = new Throwable();
    tt.initCause(t);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
    tt = new Throwable(s);
    tt.initCause(t);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
    tt = new Throwable(s);
    tt.initCause(null);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
    tt = new Throwable(s);
    tt.initCause(t);
    try { 
      tt.initCause(null);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
  }

  /** Can't call initCause on a Throwable that had a
   *  Throwable supplied in the constructor (even a 
   *  null one).
   *
   */
  public void testReInitCause() {
    String s = "abcd";
    Throwable t = new Throwable("THR");
    Throwable tt = new Throwable(t);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
    tt = new Throwable(s,t);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
    tt = new Throwable((Throwable)null);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
    tt = new Throwable(s,null);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
  }
  
  /** Can't be your own cause */
  public void testSelf() {
    String s = "abcd";
    Throwable t = new Throwable("THR");
    Throwable tt = new Throwable();
    try { 
      tt.initCause(tt);
      assertTrue(false);
    } catch (IllegalArgumentException e) {
      assertTrue(true);
    }    
    tt = new Throwable(s);
    try { 
      tt.initCause(tt);
      assertTrue(false);
    } catch (IllegalArgumentException e) {
      assertTrue(true);
    }  
    tt = new Throwable(t);
    try { 
      tt.initCause(tt);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }  
    tt = new Throwable(s,t);
    try { 
      tt.initCause(tt);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }  
  }

  public void testO() {
    Throwable t = new Throwable();
    //@ assert t.getClass() == \type(Throwable);
    assertTrue ( t.getClass() == Throwable.class);
  }
}
