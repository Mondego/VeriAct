/*
 * Created on Aug 21, 2005
 *
 * @design kiniry 21 Aug 2005 - Refactored out of InputEntry to avoid
 * existing recursion bug in typechecker in handling nested classes.
 */

package javafe;

import javafe.filespace.StringUtil;
import javafe.tc.OutsideEnv;

public class ClassInputEntry extends InputEntry {
  public ClassInputEntry(String n) { super(n); }
  public /*@non_null*/String type() { return "Class"; }
  public /*@non_null*/String typeOption() { return "class"; }
  public String verify() {
    return verify(name);
  }
  
  //+@ requires javafe.tc.OutsideEnv.initialized;
  static public String verify(/*@non_null*/String name) {
    int n = name.lastIndexOf('.');
    String[] p = StringUtil.parseList(name.substring(0,n==-1?0:n),'.');
    if (!OutsideEnv.reader().exists(p,name.substring(n+1))) {
      return "Class can not be found";
    }
    return null;
  }
  
}