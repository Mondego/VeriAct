/*
 * This file is part of the Esc/Java2 project.
 * Copyright 2004-2005 David R. Cok
 * 
 * Created on Feb 13, 2005
 */

import junit.framework.TestCase;

/**
 * This class replicates material from junit just so
 * that assertions can be placed on the methods.
 * 
 * @author David R. Cok
 */
public class LocalTestCase extends TestCase {

  //@ public normal_behavior
  //@ requires b;
  //@ pure
  static public void assertTrue(boolean b) {
    junit.framework.TestCase.assertTrue(b);
  }

  // public normal_behavior
  // No specs - nothing to prove
  //@ pure
  static public void assertTrueNP(boolean b) {
    junit.framework.TestCase.assertTrue(b);
  }
}
