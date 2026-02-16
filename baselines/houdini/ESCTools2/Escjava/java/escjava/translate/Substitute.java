/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;


import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import javafe.ast.ASTNode;
import javafe.ast.AmbiguousVariableAccess;
import javafe.ast.ArrayInit;
import javafe.ast.ArrayRefExpr;
import javafe.ast.BinaryExpr;
import javafe.ast.CastExpr;
import javafe.ast.CondExpr;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.Identifier;
import javafe.ast.InstanceOfExpr;
import javafe.ast.MethodInvocation;
import javafe.ast.NewArrayExpr;
import javafe.ast.NewInstanceExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.ParenExpr;
import javafe.ast.ThisExpr;
import javafe.ast.UnaryExpr;
import javafe.ast.VariableAccess;
import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.Set;
import escjava.ast.ArrayRangeRefExpr;
import escjava.ast.Call;
import escjava.ast.GeneralizedQuantifiedExpr;
import escjava.ast.GuardExpr;
import escjava.ast.LabelExpr;
import escjava.ast.NaryExpr;
import escjava.ast.NotModifiedExpr;
import escjava.ast.NumericalQuantifiedExpr;
import escjava.ast.QuantifiedExpr;
import escjava.ast.ResExpr;
import escjava.ast.SetCompExpr;
import escjava.ast.SubstExpr;
import escjava.ast.TagConstants;
import escjava.ast.VarInCmd;
import escjava.ast.WildRefExpr;


/**
 ** Responsible for performing substitutions in expressions.
 **/


public class Substitute {

  private static class SetRef {  // used by method "doSubst"
    Set s;
  }

  /**
   ** Does substitution on GCExprs union (resolved) SpecExprs. <p>
   **
   ** No SubstExpr's may be contained in e.<p>
   **/

  //@ ensures e != null ==> \result != null;
  public static /*@ nullable */ Expr doSubst(/*@ non_null */ Hashtable subst, /*@ nullable */ Expr e) {
    if (e == null) {
      return null;
    }
    return doSubst(subst, e, new SetRef());
  }

  //@ ensures e != null ==> \result != null;
  public static /*@ nullable */ Expr doSimpleSubst(/*@ non_null */ Hashtable subst, /*@ nullable */ Expr e) {
    if (e == null) {
      return null;
    }
    return doSubst(subst, e, null);
  }



  /**
   ** Does substitution on GCExprs union (resolved) SpecExprs. <p>
   **
   ** No SubstExpr's may be contained in e.<p>
   **
   ** <code>rhsVars</code> refers to a pointer to a Set.  This pointer
   ** is either null or the set of variables (GenericVarDecl's) occurring
   ** in right-hand sides of <code>subst</code>.
   **/

  //@ modifies rhsVars.s;
  //@ ensures rhsVars != null ==> \old(rhsVars.s) != null ==> rhsVars.s == \old(rhsVars.s);
  //@ ensures \result != null;
  private static /*@ non_null */ Expr doSubst(/*@ non_null */ Hashtable subst, /*@ non_null */ Expr e,
                                              /*@ non_null */ SetRef rhsVars) {
    Expr result = null;
    boolean newInstance = true;
    switch( e.getTag() ) {

    /*************************************************************
     * Cases needed only for SpecExprs:
     */

    case TagConstants.WILDREFEXPR: {
      WildRefExpr w = (WildRefExpr)e;
      ObjectDesignator newOd = w.od;
      if (newOd != null &&
          newOd.getTag() == TagConstants.EXPROBJECTDESIGNATOR) {
        ExprObjectDesignator eod = (ExprObjectDesignator)newOd;
        newOd = ExprObjectDesignator.make(eod.locDot,
            doSubst(subst, eod.expr,rhsVars));
        ((ExprObjectDesignator)newOd).type = eod.type;
      }

      result = WildRefExpr.make(
          w.var == null ? null : doSubst(subst,w.var,rhsVars),
              newOd);
      break;
    }

    case TagConstants.ARRAYRANGEREFEXPR: {
      ArrayRangeRefExpr ar = (ArrayRangeRefExpr)e;
      result = ArrayRangeRefExpr.make(
          ar.locOpenBracket,
          doSubst(subst,ar.array,rhsVars),
          ar.lowIndex == null ? null : doSubst(subst,ar.lowIndex,rhsVars),
              ar.highIndex == null ? null : doSubst(subst,ar.highIndex,rhsVars));
      break;
    }

    case TagConstants.ARRAYREFEXPR:
    {
      ArrayRefExpr ae = (ArrayRefExpr)e;
      result = ArrayRefExpr.make(doSubst(subst,ae.array,rhsVars),
          doSubst(subst,ae.index,rhsVars),
          ae.locOpenBracket,
          ae.locCloseBracket);
      break;
    }

    // Code for BinaryExpr is in default case below.

    case TagConstants.CASTEXPR:
    {
      CastExpr ce = (CastExpr)e;
      result = CastExpr.make(doSubst(subst,ce.expr,rhsVars), ce.type,
          ce.locOpenParen, ce.locCloseParen);
      break;
    }

    case TagConstants.CONDEXPR:
    {
      CondExpr ce = (CondExpr)e;
      result = CondExpr.make(doSubst(subst,ce.test,rhsVars),
          doSubst(subst,ce.thn,rhsVars),
          doSubst(subst,ce.els,rhsVars), ce.locQuestion,
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
            doSubst(subst, eod.expr,rhsVars));
        ((ExprObjectDesignator)newOd).type = eod.type;
      }

      FieldAccess newFa = FieldAccess.make(newOd, fa.id, fa.locId);
      newFa.decl = fa.decl;

      result = newFa;
      break;
    }

    case TagConstants.INSTANCEOFEXPR:
    {
      InstanceOfExpr ioe = (InstanceOfExpr)e;
      result = InstanceOfExpr.make(doSubst(subst,ioe.expr,rhsVars),
          ioe.type, ioe.loc);
      break;
    }

    case TagConstants.PARENEXPR:
    {
      ParenExpr pe = (ParenExpr)e;
      result = ParenExpr.make(doSubst(subst,pe.expr,rhsVars),
          pe.locOpenParen, pe.locCloseParen);
      break;
    }

    // UnaryExpr cases:
    case TagConstants.UNARYADD:
    case TagConstants.UNARYSUB:
    case TagConstants.NOT:
    case TagConstants.BITNOT:
    case TagConstants.INC:
    case TagConstants.DEC:
    {
      UnaryExpr ue = (UnaryExpr)e;
      result = UnaryExpr.make( ue.op, doSubst(subst,ue.expr,rhsVars), ue.locOp);
      break;
    }



    /*************************************************************
     * Cases needed for GCExpr's:
     */

    case TagConstants.LABELEXPR:
    {
      LabelExpr le = (LabelExpr)e;
      result = LabelExpr.make( le.sloc, le.eloc, le.positive,
          le.label, doSubst(subst,le.expr,rhsVars));
      break;
    }

    case TagConstants.GUARDEXPR:
    {
      GuardExpr ge = (GuardExpr)e;
      result = GuardExpr.make( doSubst( subst, ge.expr, rhsVars ), ge.locPragmaDecl );
      break;
    }

    case TagConstants.NUM_OF:
    {
      NumericalQuantifiedExpr qe = (NumericalQuantifiedExpr)e;

      // this routine requires that the bound variables of the quantified
      // expression not occur as left-hand sides of the substitution,
      // so here's a run-time check of this condition
      for(int i=0; i<qe.vars.size(); i++) {
        Assert.notFalse( !subst.contains( qe.vars.elementAt(i) ));
      }

      // the routine also requires that the variables in the right-hand
      // sides of the substitution are not captured by the quantified
      // expression, so here's a check for that
      if (rhsVars != null) {
        if (rhsVars.s == null) {
          rhsVars.s = new Set();
          for (Enumeration variables = subst.elements(); variables.hasMoreElements(); ) {
            Expr ee = (Expr)variables.nextElement();
            rhsVars.s.union(freeVars(ee));
          }
        }
        for (int i = 0; i < qe.vars.size(); i++) {
          Assert.notFalse(!rhsVars.s.contains(qe.vars.elementAt(i)));
        }
      }

      ExprVec newNopats;
      if (qe.nopats == null) {
        newNopats = null;
      } else {
        newNopats = ExprVec.make(qe.nopats.size());
        for (int i = 0; i < qe.nopats.size(); i++) {
          newNopats.addElement(doSubst(subst, qe.nopats.elementAt(i), rhsVars));
        }
      }
      result = NumericalQuantifiedExpr.make( qe.sloc, qe.eloc, qe.quantifier,
          qe.vars, 
          qe.rangeExpr == null ? null :
            doSubst(subst,qe.rangeExpr,rhsVars),
            doSubst(subst,qe.expr,rhsVars),
            newNopats);
      break;
    }
    case TagConstants.MAXQUANT:
    case TagConstants.MIN:
    case TagConstants.SUM:
    case TagConstants.PRODUCT:
    {
      GeneralizedQuantifiedExpr qe = (GeneralizedQuantifiedExpr)e;

      // this routine requires that the bound variables of the quantified
      // expression not occur as left-hand sides of the substitution,
      // so here's a run-time check of this condition
      for(int i=0; i<qe.vars.size(); i++) {
        Assert.notFalse( !subst.contains( qe.vars.elementAt(i) ));
      }

      // the routine also requires that the variables in the right-hand
      // sides of the substitution are not captured by the quantified
      // expression, so here's a check for that
      if (rhsVars.s == null) {
        rhsVars.s = new Set();
        for (Enumeration variables = subst.elements(); variables.hasMoreElements(); ) {
          Expr ee = (Expr)variables.nextElement();
          rhsVars.s.union(freeVars(ee));
        }
      }
      for (int i = 0; i < qe.vars.size(); i++) {
        Assert.notFalse(!rhsVars.s.contains(qe.vars.elementAt(i)));
      }

      ExprVec newNopats;
      if (qe.nopats == null) {
        newNopats = null;
      } else {
        newNopats = ExprVec.make(qe.nopats.size());
        for (int i = 0; i < qe.nopats.size(); i++) {
          newNopats.addElement(doSubst(subst, qe.nopats.elementAt(i), rhsVars));
        }
      }
      result = GeneralizedQuantifiedExpr.make( qe.sloc, qe.eloc, qe.quantifier,
          qe.vars, doSubst(subst,qe.expr,rhsVars),
          doSubst(subst, qe.rangeExpr,rhsVars),
          newNopats);
      break;
    }
    case TagConstants.FORALL:
    case TagConstants.EXISTS:
    {
      QuantifiedExpr qe = (QuantifiedExpr)e;

      // this routine requires that the bound variables of the quantified
      // expression not occur as left-hand sides of the substitution,
      // so here's a run-time check of this condition
      for(int i=0; i<qe.vars.size(); i++) {
        Assert.notFalse( !subst.contains( qe.vars.elementAt(i) ));
      }

      // the routine also requires that the variables in the right-hand
      // sides of the substitution are not captured by the quantified
      // expression, so here's a check for that
      if (rhsVars != null) {
        if (rhsVars.s == null) {
          rhsVars.s = new Set();
          for (Enumeration variables = subst.elements(); variables.hasMoreElements(); ) {
            Expr ee = (Expr)variables.nextElement();
            rhsVars.s.union(freeVars(ee));
          }
        }
        for (int i = 0; i < qe.vars.size(); i++) {
          Assert.notFalse(!rhsVars.s.contains(qe.vars.elementAt(i)));
        }
      }

      ExprVec newNopats;
      if (qe.nopats == null) {
        newNopats = null;
      } else {
        newNopats = ExprVec.make(qe.nopats.size());
        for (int i = 0; i < qe.nopats.size(); i++) {
          newNopats.addElement(doSubst(subst, qe.nopats.elementAt(i), rhsVars));
        }
      }

      ExprVec newPats;
      if (qe.pats == null) {
        newPats = null;
      } else {
        newPats = ExprVec.make(qe.pats.size());
        for (int i = 0; i < qe.pats.size(); i++) {
          newPats.addElement(doSubst(subst, qe.pats.elementAt(i), rhsVars));
        }
      }

      result = QuantifiedExpr.make( qe.sloc, qe.eloc, qe.quantifier,
          qe.vars, 
          qe.rangeExpr == null ? null :
            doSubst(subst,qe.rangeExpr,rhsVars),
            doSubst(subst,qe.expr,rhsVars),
            newNopats, newPats);
      break;
    }

    case TagConstants.RESEXPR:
    {
      newInstance = false;
      Expr to = (Expr)subst.get( resexpr );
      result = (to==null ? e : to);
      break;
    }
    case TagConstants.THISEXPR:
    {
      newInstance = false;
      Expr to = (Expr)subst.get( thisexpr );
      result = (to==null ? e : to);
      break;
    }
    case TagConstants.TYPEEXPR:
    case TagConstants.LOCKSETEXPR:
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
    case TagConstants.EVERYTHINGEXPR:
    case TagConstants.NOTHINGEXPR:
    {
      newInstance = false;
      result = e;
      break;
    }

    case TagConstants.NOTMODIFIEDEXPR:
    {
      NotModifiedExpr nme = (NotModifiedExpr)e;
      result = NotModifiedExpr.make(nme.loc, 
          doSubst(subst, nme.expr, rhsVars));
      break;
    }

    case TagConstants.VARIABLEACCESS:
    {
      //newInstance = false;
      VariableAccess va = (VariableAccess)e;
      Expr to = (Expr)subst.get( va.decl );

      if( to != null ) {
        // System.out.println("Doing subst on "+va.decl.id);
        result =  to;
      } else {
        // System.out.println("Not doing subst on "+va.decl.id + " " + va.decl);
        result = e;
        if (va.id == Identifier.intern("RES")) {
          to = (Expr)subst.get(resexpr);
          if (to != null) result = to;
        }
      }
      break;
    }

    default:
    {
      if (e instanceof NaryExpr) {

	      NaryExpr ne = (NaryExpr)e;
	      ExprVec nu = ExprVec.make( ne.exprs.size() );
	      for( int i=0; i<ne.exprs.size(); i++ ) {
	        nu.addElement( doSubst( subst, ne.exprs.elementAt(i), rhsVars ) );
	      }
	      result =  NaryExpr.make(ne.sloc, ne.eloc, ne.op, ne.methodName, nu);
	      ((NaryExpr)result).symbol = ne.symbol;
	    } else if (e instanceof BinaryExpr) {
        BinaryExpr be = (BinaryExpr)e;
        result = BinaryExpr.make(be.op, doSubst(subst,be.left,rhsVars),
            doSubst(subst,be.right,rhsVars), be.locOp);
      } else if (e instanceof SetCompExpr) {
        SetCompExpr se = (SetCompExpr)e;
        // FIXME - how do bound vars affect substitution?
        return e;
      } else if (e instanceof MethodInvocation) {
        MethodInvocation me = (MethodInvocation)e;
        ExprVec args = ExprVec.make(me.args.size());
        for (int i = 0; i< me.args.size(); ++i) {
          Expr ee = me.args.elementAt(i);
          args.addElement( doSubst(subst, ee, rhsVars));
        }
        ObjectDesignator od = me.od;
        if (od instanceof ExprObjectDesignator) {
          od = ExprObjectDesignator.make(od.locDot,doSubst(subst, ((ExprObjectDesignator)od).expr, rhsVars));
        }  // FIXME is it ok to ignore the other types of ObjectDesignator?
        MethodInvocation r = MethodInvocation.make(od, me.id, me.tmodifiers, me.locId, 
            me.locOpenParen, args);
        r.decl = me.decl;
        result = r;
      } else if (e instanceof NewArrayExpr) {
        NewArrayExpr me = (NewArrayExpr)e;
        ArrayInit init = me.init;
        // FIXME - need substitution
        //Expr init = doSubst(subst, me.init, rhsVars);
        result = NewArrayExpr.make(me.type,me.dims,init,
            me.loc,me.locOpenBrackets);
      } else if (e instanceof NewInstanceExpr) {
        NewInstanceExpr me = (NewInstanceExpr)e;
        ExprVec args = ExprVec.make(me.args.size());
        for (int i = 0; i< me.args.size(); ++i) {
          Expr ee = me.args.elementAt(i);
          args.addElement( doSubst(subst, ee, rhsVars));
        }
        Expr ee = me.enclosingInstance;
        if (ee != null) ee = doSubst(subst, ee, rhsVars);
        NewInstanceExpr r = NewInstanceExpr.make(ee,
            me.locDot, me.type, args, me.anonDecl,
            me.loc, me.locOpenParen);
        r.decl = me.decl;
        result = r;
      } else if (e instanceof AmbiguousVariableAccess) {
        // This will happen if there has been an undefined variable in
        // another compilation unit.  We ignore it here, rather than
        // throwing the Assertion failure, in order to continue as 
        // gracefully as possible.
        result = e;
      } else {

        Assert.fail("Bad expr in Substitute.doSubst: "+e+ " " 
            + Location.toString(e.getStartLoc()));
        return null; // dummy return
      }
    }
    }

    if (newInstance) escjava.tc.FlowInsensitiveChecks.copyType(e, result);
    return result;
  }

  final static public Expr resexpr = ResExpr.make(Location.NULL);
  final static public Expr thisexpr = ThisExpr.make(null,Location.NULL);

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
  //@ ensures \result != null;
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
      System.out.println("FA " + Location.toString(e.getStartLoc()) + " " + e);
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

