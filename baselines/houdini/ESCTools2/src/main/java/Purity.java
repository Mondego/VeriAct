/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;


import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.tc.Types;

import javafe.util.Assert;
import javafe.util.Info;
import javafe.util.Location;


public abstract class Purity {

  //// The decoration for purity, plus getters and setters for it

  /** Decorates <code>VarInit</code> nodes with purity information.
    Null means a <code>VarInit</code> is pure; non-null means it's
    impure.  Also used by <code>Translate</code> class to decorate
    <code>LocalVarDecl</code>s marked with "uninitialized" with the
    declaration of the corresponding boolean that keeps track of the
    variable's initialization state. */

  static ASTDecoration translateDecoration
    = new ASTDecoration("translateDecoration");

  /** Return true iff <code>expr</code> or any of its subexpressions
    mutates the heap or local variables.  Requires that
    <code>expr</code> or an expression containing it has been
    decorated by a call to <code>decorate</code>. */

  public static boolean impure(VarInit expr) {
    return translateDecoration.get(expr) != null;
  }

  /** Return true iff neither <code>expr</code> nor any of its
    subexpressions mutate the heap or local variables.  Requires that
    <code>expr</code> or an expression containing it has been
    decorated by a call to <code>decorate</code>. */

  public static boolean pure(VarInit expr) {
    return translateDecoration.get(expr) == null;
  }

  /** Set the decoration indicating the <code>expr</code> is
    impure. */

  private static void makeImpure(VarInit expr) {
    translateDecoration.set(expr, translateDecoration);
  }


  //// Traversal

  /** Decorate <code>expr</code> and its subexpressions with purity
    information. */

  public static void decorate(VarInit expr) {
      // Info.out("Decorate "+expr);
    int tag = expr.getTag();

    switch (tag) {
    case TagConstants.SETCOMPEXPR:
    case TagConstants.NEWINSTANCEEXPR:
    case TagConstants.ASSIGN:
    case TagConstants.ASGMUL:
    case TagConstants.ASGADD: case TagConstants.ASGSUB:
    case TagConstants.ASGLSHIFT: case TagConstants.ASGRSHIFT:
    case TagConstants.ASGURSHIFT: case TagConstants.ASGBITAND:
    case TagConstants.ASGBITOR: case TagConstants.ASGBITXOR:
    case TagConstants.ASGDIV: case TagConstants.ASGREM:
    case TagConstants.INC: case TagConstants.DEC:
    case TagConstants.POSTFIXINC: case TagConstants.POSTFIXDEC:
    case TagConstants.METHODINVOCATION:
      for (int i = 0; i < expr.childCount(); i++) {
	Object n = expr.childAt(i);
	if (n instanceof VarInit)
	  decorate((VarInit)n);
      }
      makeImpure(expr);
      return;

    case TagConstants.THISEXPR:
    case TagConstants.BOOLEANLIT: 
    case TagConstants.STRINGLIT:
    case TagConstants.CHARLIT:
    case TagConstants.DOUBLELIT: 
    case TagConstants.FLOATLIT:
    case TagConstants.INTLIT:
    case TagConstants.LONGLIT:
    case TagConstants.NULLLIT:
    case TagConstants.VARIABLEACCESS:
    case TagConstants.CLASSLITERAL:
    case TagConstants.TYPEEXPR:  // FIXME - TYPE-EQUIV
      return;

    case TagConstants.ARRAYINIT: 
    case TagConstants.ARRAYREFEXPR:
    case TagConstants.NEWARRAYEXPR:
    case TagConstants.CONDEXPR:
    case TagConstants.INSTANCEOFEXPR:
    case TagConstants.CASTEXPR:
    case TagConstants.OR: case TagConstants.AND:
    case TagConstants.BITOR: case TagConstants.BITXOR:
    case TagConstants.BITAND: case TagConstants.NE:
    case TagConstants.EQ: case TagConstants.GE:
    case TagConstants.GT: case TagConstants.LE:
    case TagConstants.LT: case TagConstants.LSHIFT:
    case TagConstants.RSHIFT: case TagConstants.URSHIFT:
    case TagConstants.ADD: case TagConstants.SUB:
    case TagConstants.STAR:
    case TagConstants.DIV: case TagConstants.MOD:
    case TagConstants.UNARYSUB: case TagConstants.UNARYADD:
    case TagConstants.NOT: case TagConstants.BITNOT:
    case TagConstants.PARENEXPR:
    case TagConstants.FIELDACCESS:
      {
	boolean impure = false;
	for (int i = 0; i < expr.childCount(); i++) {
	  Object n = expr.childAt(i);
	  if (n instanceof VarInit) {
	    VarInit child = (VarInit)n;
	    decorate(child);
	    impure = impure || impure(child);
	  }
	}
	if (impure) makeImpure(expr);
	return;
      }

    default:
      if (expr instanceof GeneralizedQuantifiedExpr) {
	GeneralizedQuantifiedExpr q = (GeneralizedQuantifiedExpr)expr;
        boolean impure = false;
	decorate(q.expr);
	impure = impure || impure(q.expr);
	if (q.rangeExpr != null) {
	    decorate(q.rangeExpr);
	    impure = impure || impure(q.rangeExpr);
	}
	if (impure) makeImpure(expr);
	if (impure) {
	    javafe.util.ErrorSet.error(q.getStartLoc(),
		"A quantified expression may not contain impure expressions");
	}
	return;
      } else {
	  //@ unreachable;
	  Assert.fail("Tag " + TagConstants.toString(tag) + " " +
		    Location.toString(expr.getStartLoc()) + " " + expr);
      }
    }
  }
}
