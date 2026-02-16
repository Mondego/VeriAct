/*
 * This file is part of the Esc/Java2 project.
 * Copyright 2004 David R. Cok
 * 
 * Created on Feb 13, 2005
 */

/**
 * TODO Fill in class description
 * 
 * @author David R. Cok
 */
public class TestException extends LocalTestCase {
  
  /** getMessage() returns what was supplied in the constructor
      getMessage() returns null if no string supplied in constructor
  */
  public void testMessage() {
    String s = "abcd";
    Exception t = new Exception("THR");
    assertTrue ((new Exception()).getMessage() == null);
    assertTrue ((new Exception((String)null)).getMessage() == null);
    assertTrue ((new Exception(s)).getMessage() == s);
    assertTrue ((new Exception(s,t)).getMessage() == s);
    assertTrue ((new Exception(t)).getMessage() != null);
    assertTrueNP ((new Exception(t)).getMessage().equals("java.lang.Exception: THR"));
    assertTrue ((new Exception(s,null)).getMessage() == s);
    assertTrue ((new Exception(null,null)).getMessage() == null);
    assertTrue ((new Exception((Throwable)null)).getMessage()==null);
  }
  
  /** getCause() returns what was supplied in the constructor,
   *  or null
   *
   */
  public void testCause() {
    String s = "abcd";
    Exception t = new Exception("THR");
    assertTrue ((new Exception()).getCause() == null);
    assertTrue ((new Exception(s)).getCause() == null);
    assertTrue ((new Exception(t)).getCause() == t);
    assertTrue ((new Exception(s,t)).getCause() == t);
    assertTrue ((new Exception((Throwable)null)).getCause() == null);
    assertTrue ((new Exception(s,null)).getCause() == null);
    assertTrue ((new Exception(null,null)).getCause() == null);
  }
  
  /** Tests that getCause() returns what was supplied in
   *  initCause()
   *
   */
  public void testInitCause() {
    String s = "abcd";
    Exception t = new Exception("THR");
    Exception tt = new Exception();
    tt.initCause(t);
    assertTrue (tt.getCause() == t);
  }

  /** Can't be your own cause */
  public void testSelfCause() {
    Exception t = new Exception("THR");
    Exception tt = new Exception();
    try {
      tt.initCause(tt);
      assertTrue (false);
    } catch (Exception e) {
      assertTrue (e instanceof IllegalArgumentException );
    }
    assertTrue (tt.getCause() == null);
    tt.initCause(t);
    assertTrue (tt.getCause() == t);
  }
  
  /** Once initCause is called, you can't call it again,
   *  even if the argument is null
   *
   */
  public void testDuplicateInitCause() {
    String s = "abcd";
    Exception t = new Exception("THR");
    Exception tt = new Exception();
    tt.initCause(t);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
    tt = new Exception(s);
    tt.initCause(t);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
    
  }
  
  /** Once initCause is called, you can't call it again,
   *  even if the argument is null
   *
   */
  public void testDuplicateInitCause2() {
    String s = "abcd";
    Exception t = new Exception("THR");
    
    Exception tt = new Exception(s);
    tt.initCause(null);
    try { 
      tt.initCause(t);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }

    tt = new Exception(s);
    tt.initCause(t);
    try { 
      tt.initCause(null);
      assertTrue(false);
    } catch (IllegalStateException e) {
      assertTrue(true);
    }
  }

  public void testO() {
    Exception e = new Exception();
    assertTrue (e.getClass() == Exception.class);
    //@ assert \typeof(e) == \type(Exception);
    e = new NullPointerException();
    assertTrue (e.getClass() == NullPointerException.class);
    //@ assert \typeof(e) == \type(NullPointerException);
  }
}
