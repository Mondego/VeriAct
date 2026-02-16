/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import javafe.util.*;


/**
 ** Responsible for performing substitutions in expressions.
 **/


public class Substitute {

  /**
   ** Does substitution on GCExprs union (resolved) SpecExprs. <p>
   **
   ** No SubstExpr's may be contained in e.<p>
   **/

  public static Expr doSubst(Hashtable subst, Expr e) {
    if (e==null)
	return null;

    Expr result = null;
    switch( e.getTag() ) {
	
    /*************************************************************
     * Cases needed only for SpecExprs:
     */

    case TagConstants.ARRAYREFEXPR:
      {
	ArrayRefExpr ae = (ArrayRefExpr)e;
	result = ArrayRefExpr.make(doSubst(subst,ae.array),
				   doSubst(subst,ae.index),
				   ae.locOpenBracket,
				   ae.locCloseBracket);
	break;
      }

    // Code for BinaryExpr is in default case below.

    case TagConstants.CASTEXPR:
      {
	CastExpr ce = (CastExpr)e;
	result = CastExpr.make(doSubst(subst,ce.expr), ce.type,
			       ce.locOpenParen, ce.locCloseParen);
	break;
      }

    case TagConstants.CONDEXPR:
      {
	CondExpr ce = (CondExpr)e;
	result = CondExpr.make(doSubst(subst,ce.test), doSubst(subst,ce.thn),
			       doSubst(subst,ce.els), ce.locQuestion,
			       ce.locColon);
	break;
      }

    case TagConstants.FIELDACCESS:
      {
	FieldAccess fa = (FieldAccess)e;

	ObjectDesignator newOd = fa.od;
	if (newOd.getTag() == TagConstants.EXPROBJECTDESIGNATOR) {
	    ExprObjectDesignator eod = (ExprObjectDesignator)newOd;
	    newOd = ExprObjectDesignator.make(eod.locDot,
					      doSubst(subst, eod.expr));
	}

	FieldAccess newFa = FieldAccess.make(newOd, fa.id, fa.locId);
	newFa.decl = fa.decl;

	result = newFa;
	break;
      }

    case TagConstants.INSTANCEOFEXPR:
      {
	InstanceOfExpr ioe = (InstanceOfExpr)e;
	result = InstanceOfExpr.make(doSubst(subst,ioe.expr), ioe.type,
				     ioe.loc);
	break;
      }

    case TagConstants.PARENEXPR:
      {
	ParenExpr pe = (ParenExpr)e;
	result = ParenExpr.make( doSubst(subst,pe.expr), pe.locOpenParen,
			         pe.locCloseParen );
	break;
      }

    // UnaryExpr cases:
    case TagConstants.UNARYADD:
    case TagConstants.UNARYSUB:
    case TagConstants.NOT:
    case TagConstants.BITNOT:
    case TagConstants.INC:
    case TagConstants.DEC:
    case TagConstants.POSTFIXINC:
    case TagConstants.POSTFIXDEC:
      {
	UnaryExpr ue = (UnaryExpr)e;
	result = UnaryExpr.make( ue.op, doSubst(subst,ue.expr), ue.locOp);
	break;
      }



    /*************************************************************
     * Cases needed for GCExpr's:
     */

    case TagConstants.LABELEXPR:
      {
	LabelExpr le = (LabelExpr)e;
	result = LabelExpr.make( le.sloc, le.eloc, le.positive,
		  	         le.label, doSubst( subst, le.expr));
	break;
      }

    case TagConstants.GUARDEXPR:
      {
	GuardExpr ge = (GuardExpr)e;
	result = GuardExpr.make( doSubst( subst, ge.expr ), ge.locPragmaDecl );
	break;
      }

    case TagConstants.FORALL:
    case TagConstants.EXISTS:
      {
	QuantifiedExpr qe = (QuantifiedExpr)e;
	// FIXME not correctly considering captured variables here
	for(int i=0; i<qe.vars.size(); i++) {
	  Assert.notFalse( !subst.contains( qe.vars.elementAt(i) ));
	}
	ExprVec newNopats;
	if (qe.nopats == null) {
	  newNopats = null;
	} else {
	  newNopats = ExprVec.make(qe.nopats.size());
	  for (int i = 0; i < qe.nopats.size(); i++) {
	    newNopats.addElement(doSubst(subst, qe.nopats.elementAt(i)));
	  }
	}
	result = QuantifiedExpr.make( qe.sloc, qe.eloc, qe.quantifier,
		  	   	      qe.vars, doSubst(subst,qe.expr),
				      newNopats);
	break;
      }
	  
    case TagConstants.TYPEEXPR:
    case TagConstants.LOCKSETEXPR:
    case TagConstants.TIDEXPR:
    case TagConstants.MAINEXPR:
    case TagConstants.WITNESSEXPR:
    case TagConstants.RESEXPR:
    case TagConstants.WILDREFEXPR:
    case TagConstants.THISEXPR:
    case TagConstants.CLASSLITERAL:

    case TagConstants.BOOLEANLIT: 
    case TagConstants.CHARLIT:
    case TagConstants.DOUBLELIT: 
    case TagConstants.FLOATLIT:
    case TagConstants.INTLIT:
    case TagConstants.LONGLIT:
    case TagConstants.NULLLIT:
    case TagConstants.STRINGLIT:
    case TagConstants.SYMBOLLIT:
      {
	result = e;
	break;
      }
	  
    case TagConstants.VARIABLEACCESS:
      {
	VariableAccess va = (VariableAccess)e;
	Expr to = (Expr)subst.get( va.decl );

	if( to != null ) {
	  // System.out.println("Doing subst on "+va.decl.id);
	  result =  to;
	} else {
	  // System.out.println("Not doing subst on "+va.decl.id);
	  result = e;
	}
	break;
      }

    default:
      {
	if (e instanceof NaryExpr) {

	  NaryExpr ne = (NaryExpr)e;
	  ExprVec nu = ExprVec.make( ne.exprs.size() );
	  for( int i=0; i<ne.exprs.size(); i++ ) {
	    nu.addElement( doSubst( subst, ne.exprs.elementAt(i) ) );
	  }
	  result =  NaryExpr.make(ne.sloc, ne.eloc, ne.op, nu);
	} else if (e instanceof BinaryExpr) {

	  BinaryExpr be = (BinaryExpr)e;
	  result = BinaryExpr.make(be.op, doSubst(subst,be.left),
				   doSubst(subst,be.right), be.locOp);
	} else {

	    Assert.fail("Bad expr in Substitute.doSubst: "+e);
	    return null; // dummy return
        }
      }
    }

    escjava.tc.FlowInsensitiveChecks.copyType(e, result);
    return result;
  }



  /**
   ** Calculate the free variables of an expression or a GuardedCmd. <p>
   **
   ** Precondition; e must be resolved, e may not contain FieldAccess as
   **		    a subnode. <p>
   **
   ** [mdl: I have checked that this code works with invariants
   **  translated to Exprs.  I am unsure if it works correctly on
   **  anything else.]
   **
   ** Note: length is not a free variable of translated code.  (It turns
   ** into applications of the built-in function arrayLength.)
   **/
  public static Set freeVars(ASTNode e) {

    CalcFreeVars t = new CalcFreeVars();
    t.traverse(e);
    return t.getFreeVars();
  }


    public static boolean mentionsFresh(Expr e) {
	if( e.getTag() == TagConstants.FRESH )
	    return true;

	for( int i=0; i<e.childCount(); i++ ) {
	    Object o = e.childAt(i);
	    if( o instanceof Expr ) {
	        if( mentionsFresh( (Expr)o ) )
		    return true;
	    }
        }

	return false;
    }
}


class CalcFreeVars {
  
  private Set freeVars = new Set();
  private Vector quantifiedVars = new Vector();  // need a bag

  Set getFreeVars() {
    return freeVars;
  }

  void traverse(ASTNode e) {

    GenericVarDeclVec bindings = null;
    
    switch( e.getTag() ) {

      /*
       * Local/parameter/field variable references:
       */
      case TagConstants.AMBIGUOUSVARIABLEACCESS:
	{
	  Assert.precondition("Unresolved ASTNode passed to freeVars");
	  return;
        }
      case TagConstants.VARIABLEACCESS:
	{
	  VariableAccess va = (VariableAccess)e;

	  if( !quantifiedVars.contains(va.decl) ) {
	    freeVars.add( va.decl );
	  }
	  return;
	}
      case TagConstants.FIELDACCESS:
	{
	  Assert.precondition
	      ("ASTNode with FieldAccess subnode passed to freeVars");
	  break;
	}


      /*
       * Variable binding operators:
       */
      case TagConstants.SUBSTEXPR:
	{
	  GenericVarDecl var = ((SubstExpr)e).var;
	  bindings = GenericVarDeclVec.make();
	  bindings.addElement(var);
	  break;
	}
      case TagConstants.FORALL:
      case TagConstants.EXISTS:
	{
	  QuantifiedExpr qe = (QuantifiedExpr)e;
	  bindings = qe.vars;
	  break;
	}
      case TagConstants.VARINCMD:
	{
	  VarInCmd c = (VarInCmd)e;
	  bindings = c.v;
	  break;
	}

      /*
       * Need to handle Call's specially:
       */
      case TagConstants.CALL:
	{
	  Call call = (Call)e;
	  // want to include vars in call.desugared
	  // desugared is not considered a child,
	  // so do traversal explicitly
	  traverse( call.desugared );
	  return; // do not look at children
	}
    }

    // Record bound variables

    if( bindings != null ) {
      for(int i=0; i<bindings.size(); i++ ) {
	GenericVarDecl decl = bindings.elementAt(i);
	quantifiedVars.addElement(  decl );
      }
    }

    // Visit all children
    
    for( int i=0; i<e.childCount(); i++ ) {
      Object o = e.childAt(i);
      if( o instanceof ASTNode )
	traverse( (ASTNode)o );
    }

    // Remove bound variables from env
    if( bindings != null ) {
      for(int i=0; i<bindings.size(); i++ ) {
	// The following line appears not to be used [KRML, 12 Apr 1999]
	GenericVarDecl decl = bindings.elementAt(i);
	quantifiedVars.removeElementAt( quantifiedVars.size()-1 );
      }
    }
    
    return;
  }

}

