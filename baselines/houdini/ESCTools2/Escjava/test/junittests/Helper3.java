/*
 * Created on Aug 22, 2005
 * $Id$
 */

/**
 * @author kiniry
 *
 * Tests calling private helper from constuctor.
 * 
 * @see <a href="http://sort.ucd.ie/tracker/index.php?func=detail&aid=138&group_id=97&atid=441">Bug #138</a>
 */
public class Helper3 {
  //@ non_null
  Object o;

  // this method should not check as it calls a non-helper which violates the
  // object invariant in the precondition.
  public Helper3() {
    non_helper();  // ERROR - preconditions not satisfied
  }
  
  public Helper3(boolean b) {
    helper();
  }

  // trying the nonhelper in a method
  //@ modifies o;
  public void m() {
    helper();
    non_helper();
  }
  
  //@ modifies o;
  private void non_helper() {
    	o = new Object();
  }
  
  //@ modifies o;
  //@ ensures o != null;
  private /*@ helper @*/ void helper() {
    o = new Object();
  }


}
