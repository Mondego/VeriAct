/* Copyright Hewlett-Packard, 2002 */


package escjava.sp;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.translate.GC;
import escjava.translate.Substitute;

import javafe.util.Location;
import javafe.util.Assert;

import java.util.Hashtable;
import java.util.Enumeration;

public class VarMap {

  /*@ spec_public */ private Hashtable table;
  /*@ invariant table != null ==>
                table.keyType == type(GenericVarDecl) &&
	        table.elementType == type(RefInt); */

  final private static VarMap botMap = new VarMap();
  //@ invariant botMap.table == null;
  //@ invariant this == botMap || this.table != null;

  final private static VarMap idMap = new VarMap();
  //@ invariant botMap != idMap;

  static {
    idMap.table = new Hashtable();
  }

  /** Returns the special "bottom" VarMap. */

  //@ ensures RES != null;
  //@ ensures RES.table == null;
  static VarMap bottom() {
    return botMap;
  }

  //@ ensures RES == (table == null);
  boolean isBottom() {
    return table == null;
  }

  /** Returns the identity VarMap. */

  //@ ensures RES != null;
  //@ ensures RES.table != null;
  static VarMap identity() {
    return idMap;
  }

  Expr get(/*@ non_null */ GenericVarDecl v) {
    return (Expr) table.get(v);
  }

  /** Returns a VarMap identical to "this", except mapping "v" to "e". */

  //@ ensures RES != null;
  //@ ensures table != null ==> RES.table != null;
  VarMap extend(/*@ non_null */ GenericVarDecl v, /*@ non_null */ Expr e) {
    if (this == botMap) return botMap;
    VarMap r = new VarMap();
    r.table = (Hashtable)table.clone();
    //@ assume r.table.keyType == table.keyType;
    //@ assume r.table.elementType == table.elementType;
    r.table.put(v,e);
    return r;
  }

  /** Returns a VarMap identical to "this", except mapping "v" to "e" for every
      pair <v,e> in the hashtable h. */

  //@ ensures RES != null;
  //@ ensures table != null ==> RES.table != null;
    VarMap extend(/*@ non_null */ Hashtable h) {
      if (this == botMap) return botMap;
      VarMap r = new VarMap();
      r.table = (Hashtable)table.clone();
      //@ assume r.table.keyType == table.keyType;
      //@ assume r.table.elementType == table.elementType;
      for( Enumeration e = h.keys(); e.hasMoreElements(); ) {
	  GenericVarDecl v = (GenericVarDecl)e.nextElement();
	  r.table.put(v, h.get(v));
      }
      return r;
  }

  /** Returns a VarMap identical to "this", except mapping each element of
  "vec" to itself. */
  //@ ensures RES != null;
  //@ ensures table != null ==> RES.table != null;
  VarMap unmap (/*@ non_null */ GenericVarDeclVec vec) {
    if (this == botMap) return botMap;
    VarMap r = new VarMap();
    r.table = (Hashtable)table.clone();
    //@ assume r.table.keyType == table.keyType;
    //@ assume r.table.elementType == table.elementType;
    for (int i = 0; i < vec.size(); i++)
      r.table.remove(vec.elementAt(i));
    return r;
  }

    Enumeration dom() {
	return table.keys();
    }

    Expr lookup(GenericVarDecl d) {
	return (Expr)table.get(d);
    }

  /** This is the two-input-map version of the more general <code>merge</code>
    * method below.<p>
    **/

  //@ requires rename.length == 2;
  //@ requires typeof(rename) == type(GuardedCmdVec[]);
  /*@ requires lastVarUse != null ==>
               lastVarUse.keyType == type(GenericVarDecl) &&
	       lastVarUse.elementType == type(RefInt); */
  //@ ensures RES != null;
  //@ ensures m.table != null || n.table != null ==> RES.table != null;
  static VarMap merge(/*@ non_null */ VarMap m, /*@ non_null */ VarMap n,
		      /*@ non_null */ GuardedCmdVec[] rename, int loc,
		      int p, Hashtable lastVarUse){
    VarMap[] mm = {m, n};
    return merge(mm, rename, loc, p, lastVarUse);
  }

  /** If all elements of "mm" are "bottom" then the result is "bottom".
    * Otherwise, the result is a map whose domain is the union of the
    * domains of the elements of "mm", restricted to those variables
    * whose "lastVarUse" value is at least "p" (if a variable is not in
    * the domain of "lastVarUse", its "lastVarUse" value is considered
    * to be negative infinity).  For each variable "v" in the domain of
    * the output map:
    * <ul>
    * <li>  if all non-"bottom" elements of "mm" map "v" to the same
    *       expression, then the output map also maps "v" to that
    *       expression;
    * <li>  otherwise, the output map maps "v" to a fresh variable "v'"
    *       (where the print name of "v'" is appropriately chosen
    *       given "v" and "loc"), and for each non-"bottom" "mm[i]",
    *       "rename[i]" includes "assume v' == mm[i][[v]]".
    * </ul>
    *
    * Parameter <code>lastVarUse</code> may be passed in as <code>null</code>,
    * in which case the method acts as if <code>lastVarUse</code> had been
    * a hashtable that maps every variable to max infinity.
    **/

  //@ requires elemsnonnull(mm);
  //@ requires mm.length == rename.length;
  //@ requires typeof(mm) == type(VarMap[]);
  //@ requires typeof(rename) == type(GuardedCmdVec[]);
  /*@ requires lastVarUse != null ==>
               lastVarUse.keyType == type(GenericVarDecl) &&
	       lastVarUse.elementType == type(RefInt); */
  //@ ensures RES != null;
  /*@ ensures (exists int i; 0 <= i && i < mm.length && mm[i].table != null)
              ==> RES.table != null; */
  //@ modifies rename[*];
  //@ ensures elemsnonnull(rename);
  static VarMap merge(VarMap[] mm, /*@ non_null */ GuardedCmdVec[] rename, int loc,
		      int p, Hashtable lastVarUse){
//
// The following should be ok, but code currently crashes because this
// precondition is not respected:
//
//  Assert.notFalse(loc!=Location.NULL);	!!!!

    Hashtable vars = new Hashtable();
    boolean nonBottom = false;
    for( int i=0; i<mm.length; i++) {
      rename[i] = GuardedCmdVec.make();
      if( mm[i].table != null ) {
	nonBottom = true;
	for( Enumeration e = mm[i].table.keys(); e.hasMoreElements(); ) {
	  GenericVarDecl var = (GenericVarDecl)e.nextElement();
	  boolean putIt;
	  if (lastVarUse != null) {
	    RefInt lastUse = (RefInt)lastVarUse.get(var);
	    putIt = lastUse != null && p <= lastUse.value;
	  } else {
	    putIt = true;
	  }
	  if (putIt) {
	    vars.put(var, mm[i].table.get(var));
	  }
	}
      }
    }

    if (!nonBottom) return botMap;

    VarMap r = new VarMap();
    r.table = new Hashtable();

    for( Enumeration e = vars.keys(); e.hasMoreElements(); ) {
      GenericVarDecl var = (GenericVarDecl)e.nextElement();
      Expr expr = (Expr)vars.get(var);
      boolean conflict = false;
      for( int i=0; i<mm.length; i++) {
	if( mm[i].table != null ) {
	  Expr mapto = (Expr)mm[i].table.get(var);
	  if (expr != mapto) {
	    conflict = true;
	    break;
	  }
	}
      }

      if (conflict) {
	int uloc = loc;
	if (uloc==Location.NULL)
	    uloc = var.locId;

	LocalVarDecl varPrime
	    = LocalVarDecl.make( Modifiers.NONE,
				 null,
				 var.id,
				 var.type,
				 uloc,
				 null, Location.NULL );
	VariableAccess varPrimeRef = VariableAccess.make( var.id, loc,
							  varPrime );
	VariableAccess varRef = VariableAccess.make( var.id, loc, var );
	
	for( int i=0; i<mm.length; i++) {
	  if( mm[i].table != null ) {
	    Expr mapto = (Expr)mm[i].table.get(var);
	    if (mapto == null)
	      mapto = varRef;
	    rename[i].addElement(GC.assume(GC.nary(TagConstants.ANYEQ,
						   varPrimeRef, mapto)));
	  }
	}
	r.table.put(var,varPrimeRef);
      } else {
	r.table.put(var, expr);
      }
    }

    return r;

  }

  /** Returns the result of applying "this", viewed as a substituiton,
  to "e". */

  //@ requires table != null;
  //@ ensures RES != null;
  public Expr apply(/*@ non_null */ Expr e){
    //    return Substitute.doSubst(table, e); -- to be deleted
    return GC.subst(Location.NULL, Location.NULL, table, e) ;
  }

}
