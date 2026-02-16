/*
 * Created on Aug 21, 2005
 *
 * @design kiniry 21 Aug 2005 - Refactored out of InputEntry to avoid
 * existing recursion bug in typechecker in handling nested classes.
 */

package javafe;

public class DirInputEntry extends InputEntry {
  public DirInputEntry(/*@non_null*/String n) { super(n); }
  public /*@non_null*/String type() { return "Directory"; }
  public /*@non_null*/String typeOption() { return "dir"; }
  public /*@nullable*/String verify() {
    return verify(name);
  }
  static public /*@nullable*/String verify(/*@non_null*/String name) {
    java.io.File f= new java.io.File(name);
    if (f.exists() && f.isDirectory()) return null;
    return "Directory does not exist";
  }
}