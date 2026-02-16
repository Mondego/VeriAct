/* Copyright Hewlett-Packard, 2002 */

package escjava.ast;


import java.util.Hashtable;
import java.util.Vector;

import javafe.ast.*;
import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.Set;

// Convention: unless otherwise noted, integer fields named "loc"g refer
// to the locaction of the first character of the syntactic unit

//# TagBase javafe.tc.TagConstants.LAST_TAG + 1
//# VisitorRoot javafe.ast.Visitor




public abstract class ActionModifierPragma extends ModifierPragma {
  public int loc;



// Generated boilerplate constructors:

  /**
   ** Construct a raw ActionModifierPragma whose class invariant(s) have not
   ** yet been established.  It is the caller's job to
   ** initialize the returned node's fields so that any
   ** class invariants hold.
   **/
  //@ requires I_will_establish_invariants_afterwards
  protected ActionModifierPragma() {}    //@ nowarn Invariant,NonNullInit

  

// Generated boilerplate methods:

  /** Return the number of children a node has. */
  //@ ensures \result>=0
  public abstract int childCount();

  /** Return the first-but-ith child of a node. */
  //@ requires 0<=i
  public abstract Object childAt(int i);

  /** Return the tag of a node. */
  public abstract int getTag();

  /** Return a string representation of <code>this</code>.
  Meant for debugging use only, not for presentation. */
  public abstract String toString();

  /** Accept a visit from <code>v</code>.  This method simply
  calls the method of <code>v</code> corresponding to the
  allocated type of <code>this</code>, passing <code>this</code>
  as the argument.  See the design patterns book. */
  //@ requires v!=null
  public abstract void accept(javafe.ast.Visitor v);

  //@ requires v!=null
//@ ensures \result!=null
public abstract Object accept(javafe.ast.VisitorArgResult v, Object o);

  public void check() {
  }

}
