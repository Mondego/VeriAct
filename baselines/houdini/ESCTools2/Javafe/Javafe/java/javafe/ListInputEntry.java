/*
 * Created on Aug 21, 2005
 *
 * @design kiniry 21 Aug 2005 - Refactored out of InputEntry to avoid
 * existing recursion bug in typechecker in handling nested classes.
 */

package javafe;


public class ListInputEntry extends InputEntry {
  public ListInputEntry(/*@non_null*/String n) { super(n); }
  public /*@non_null*/String type() { return "List"; }
  public /*@non_null*/String typeOption() { return "list"; }
  public /*@nullable*/String verify() {
    return verify(name);
  }
  static public /*@nullable*/String verify(/*@non_null*/String name) {
    java.io.File f= new java.io.File(name);
    if (f.exists() && f.isFile()) return null;
    return "List file does not exist";
  }
}