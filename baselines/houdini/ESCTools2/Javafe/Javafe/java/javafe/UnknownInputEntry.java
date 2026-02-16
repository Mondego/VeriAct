/*
 * Created on Aug 21, 2005
 * 
 * @design kiniry 21 Aug 2005 - Refactored out of InputEntry to avoid
 * existing recursion bug in typechecker in handling nested classes.
 */

package javafe;

public class UnknownInputEntry extends InputEntry {
  public UnknownInputEntry(String n) { super(n); auto=true; }
  public /*@non_null*/String type() { return "Unknown"; }
  public /*@non_null*/InputEntry resolve() {
    return InputEntry.make(name);
  }
}