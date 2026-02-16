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
 ** Various useful routines for manipulating or querying
 ** Exprs.
 **
 ** Sanjit, July 2001
 **/

public class ExprUtil {


    /** Checks equality of Specification Exprs (including Exprs that
	occur in the modifies clause). For all other kinds of exprs,
	it fails.  */
    public static boolean areEqualSpecExprs(Expr e1, Expr e2) {
	int e1tag = e1.getTag();
	int e2tag = e2.getTag();
	if (e1tag == e2tag) {
	    int tag = e1tag; 
	    switch(tag) {
	    case TagConstants.TIDEXPR: 
	    case TagConstants.MAINEXPR: 
	    case TagConstants.WITNESSEXPR: 
	    case TagConstants.RESEXPR: 
	    case TagConstants.THISEXPR: 
	    case TagConstants.LOCKSETEXPR: 
            { 
		return true;
	    }

	    // check that they are the same literals
	    case TagConstants.BOOLEANLIT: {
		LiteralExpr le1 = (LiteralExpr) e1;
		LiteralExpr le2 = (LiteralExpr) e2;
		Boolean b1 = (Boolean) le1.value;
		Boolean b2 = (Boolean) le2.value;
		return b1.equals(b2);
	    }

	    case TagConstants.STRINGLIT: {
		LiteralExpr le1 = (LiteralExpr) e1;
		LiteralExpr le2 = (LiteralExpr) e2;
		String s1 = (String) le1.value;
		String s2 = (String) le2.value;
		return (s1 == s2);
	    }

	    case TagConstants.CHARLIT: {
		LiteralExpr le1 = (LiteralExpr) e1;
		LiteralExpr le2 = (LiteralExpr) e2;
		Character c1 = (Character) le1.value;
		Character c2 = (Character) le2.value;
		return c1.equals(c2);
	    }

	    case TagConstants.DOUBLELIT: {
		LiteralExpr le1 = (LiteralExpr) e1;
		LiteralExpr le2 = (LiteralExpr) e2;
		Double d1 = (Double) le1.value;
		Double d2 = (Double) le2.value;
		return d1.equals(d2);
	    }

	    case TagConstants.FLOATLIT: {
		LiteralExpr le1 = (LiteralExpr) e1;
		LiteralExpr le2 = (LiteralExpr) e2;
		Float f1 = (Float) le1.value;
		Float f2 = (Float) le2.value;
		return f1.equals(f2);
	    }

	    case TagConstants.INTLIT: {
		LiteralExpr le1 = (LiteralExpr) e1;
		LiteralExpr le2 = (LiteralExpr) e2;
		Integer i1 = (Integer) le1.value;
		Integer i2 = (Integer) le2.value;
		return i1.equals(i2);
	    }

	    case TagConstants.LONGLIT: {
		LiteralExpr le1 = (LiteralExpr) e1;
		LiteralExpr le2 = (LiteralExpr) e2;
		Long l1 = (Long) le1.value;
		Long l2 = (Long) le2.value;
		return l1.equals(l2);
	    }

	    case TagConstants.NULLLIT: {
		return true;
	    }

	    case TagConstants.VARIABLEACCESS: {
		VariableAccess va1 = (VariableAccess) e1;
		VariableAccess va2 = (VariableAccess) e2;
		return (va1.decl == va2.decl);
	    }

	    case TagConstants.FIELDACCESS: {
		FieldAccess fa1 = (FieldAccess) e1;
		FieldAccess fa2 = (FieldAccess) e2;
		return (fa1.decl == fa2.decl);
	    }

	    case TagConstants.ARRAYREFEXPR: {
		ArrayRefExpr ar1 = (ArrayRefExpr) e1;
		ArrayRefExpr ar2 = (ArrayRefExpr) e2;
		return (areEqualSpecExprs(ar1.array, ar2.array)
			&& areEqualSpecExprs(ar1.index, ar2.index));
	    }

	    case TagConstants.WILDREFEXPR: {
		WildRefExpr wr1 = (WildRefExpr) e1;
		WildRefExpr wr2 = (WildRefExpr) e2;
		return areEqualSpecExprs(wr1.expr, wr2.expr);
	    }

	    case TagConstants.PARENEXPR: {
		ParenExpr pe1 = (ParenExpr) e1;
		ParenExpr pe2 = (ParenExpr) e2;
		return areEqualSpecExprs(pe1.expr, pe2.expr);
	    }

	    case TagConstants.UNARYADD:
	    case TagConstants.UNARYSUB:
	    case TagConstants.NOT:
	    case TagConstants.BITNOT: {
		UnaryExpr ue1 = (UnaryExpr) e1;
		UnaryExpr ue2 = (UnaryExpr) e2;
		return areEqualSpecExprs(ue1.expr, ue2.expr);
	    }

	    case TagConstants.TYPEOF: 
	    case TagConstants.ELEMTYPE: 
	    case TagConstants.MAX: {
		return true;
	    }

	    case TagConstants.ELEMSNONNULL:
	    case TagConstants.DTTFSA: {
		NaryExpr ne1 = (NaryExpr)e1;
		NaryExpr ne2 = (NaryExpr)e2;
		
		int size1 = ne1.exprs.size();
		int size2 = ne2.exprs.size();

		if (size1 != size2) return false;

		for (int i = 0; i < size1; i++) {
		    Expr nei1 = ne1.exprs.elementAt(i);
		    Expr nei2 = ne2.exprs.elementAt(i);
		    if (!areEqualSpecExprs(nei1, nei2))
			return false;		    
		}
		return true;
	    }

	    // Binary operator expressions
	    
	    case TagConstants.OR:
	    case TagConstants.AND:
	    case TagConstants.BITOR:
	    case TagConstants.BITAND:
	    case TagConstants.BITXOR:
	    case TagConstants.EQ:
	    case TagConstants.NE:
	    case TagConstants.GE:
	    case TagConstants.GT:
	    case TagConstants.LE:
	    case TagConstants.LT:
	    case TagConstants.LSHIFT:
	    case TagConstants.RSHIFT:
	    case TagConstants.URSHIFT:
	    case TagConstants.ADD:
	    case TagConstants.SUB:
	    case TagConstants.DIV:
	    case TagConstants.MOD:
	    case TagConstants.STAR:
	    case TagConstants.IMPLIES: 
	    case TagConstants.SUBTYPE: {
		BinaryExpr be1 = (BinaryExpr) e1;
		BinaryExpr be2 = (BinaryExpr) e2;
		return (areEqualSpecExprs(be1.left, be2.left)
			&& areEqualSpecExprs(be1.right, be2.right));
	    }

	    case TagConstants.CONDEXPR: {
		CondExpr ce1 = (CondExpr) e1;
		CondExpr ce2 = (CondExpr) e2;
		return (areEqualSpecExprs(ce1.test, ce2.test)
			&& (areEqualSpecExprs(ce1.thn, ce2.thn) && areEqualSpecExprs(ce1.els, ce2.els)));
	    }

	    case TagConstants.INSTANCEOFEXPR: {
		InstanceOfExpr ie1 = (InstanceOfExpr) e1;
		InstanceOfExpr ie2 = (InstanceOfExpr) e2;
		return areEqualSpecExprs(ie1.expr, ie2.expr);
	    }

	    case TagConstants.CASTEXPR: {
		CastExpr ce1 = (CastExpr) e1;
		CastExpr ce2 = (CastExpr) e2;
		return areEqualSpecExprs(ce1.expr, ce2.expr);
	    }

	    // might need to change this case
	    case TagConstants.CLASSLITERAL: {
		ClassLiteral cl1 = (ClassLiteral) e1;
		ClassLiteral cl2 = (ClassLiteral) e2;
		return (cl1.type == cl2.type);
	    }

	    case TagConstants.TYPEEXPR: {
		TypeExpr te1 = (TypeExpr) e1;
		TypeExpr te2 = (TypeExpr) e2;
		return (te1.type == te2.type);
	    }

	    /* Equality of quantified exprs is
	       only syntactic - the exprs will be
	       judged unequal even if they only 
	       differ only in bound variables */
	    case TagConstants.FORALL:
	    case TagConstants.EXISTS: {
		QuantifiedExpr qe1 = (QuantifiedExpr) e1;
		QuantifiedExpr qe2 = (QuantifiedExpr) e2;
		return areEqualSpecExprs(qe1.expr, qe2.expr);
	    }

	    case TagConstants.LABELEXPR: {
		LabelExpr le1 = (LabelExpr) e1;
		LabelExpr le2 = (LabelExpr) e2;
		return areEqualSpecExprs(le1.expr, le2.expr);
	    }

	    case TagConstants.FRESH:
	    case TagConstants.PRE:{
		NaryExpr ne1 = (NaryExpr)e1;
		NaryExpr ne2 = (NaryExpr)e2;
		return areEqualSpecExprs(ne1.exprs.elementAt(0), ne2.exprs.elementAt(0));
	    }

	    default: // illegal tag
		//@ unreachable
		Assert.fail("Fall thru; tag = " + TagConstants.toString(tag));
		return (e1 == e2);
	    }
	}
	else // tags are different
	    return false;
    }


    /** Returns <code>true</code> if <code>ev</code> contains 
	<code>e</code>, <code>false</code> otherwise
     */
    public static boolean containsSpecExpr(ExprVec ev, Expr e) {
	for(int i = 0; i < ev.size(); i++) {
	    if (areEqualSpecExprs(ev.elementAt(i), e))
		return true;
	}
	return false;
    }


} // end of class

