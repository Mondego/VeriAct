/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.util.Set;
import javafe.util.Assert;

import escjava.ast.*;
import escjava.ast.TagConstants;

final class ATarget {
    GenericVarDecl x;
    Expr [] indices;

    ATarget(GenericVarDecl x, Expr [] indices) {
	this.x = x;
	this.indices = indices;
    }

    public static Set compute(GuardedCmd gc) {
	Set normalTargets = Targets.normal(gc);
	Set excTargets = new Set();
	System.out.println("normal targets are " + normalTargets.toString());
	Set answer = new Set();
	Set localVars = new Set();

	computeHelper(gc, answer, normalTargets, excTargets, localVars);
	System.out.println("atargets are " + answer.toString());
	return answer;	
    }

    public static boolean mentions(Expr expr, Set s) {
	if (expr instanceof VariableAccess) {
	    return s.contains(((VariableAccess) expr).decl);
	} else {
	    for (int i = 0; i < expr.childCount(); i++) {
		Object child = expr.childAt(i);
		if (child instanceof Expr && mentions((Expr) child, s)) {
		    return true;
		}
	    }
	    return false;
	}	
    }

    public static void computeHelper(GuardedCmd gc, Set set, Set localVars, Set nonConstN, Set nonConstX) {
	int tag = gc.getTag();
	switch (tag) {
	case TagConstants.SKIPCMD:
	case TagConstants.ASSERTCMD:
	case TagConstants.ASSUMECMD:
	    return;

	case TagConstants.RAISECMD: {
	    nonConstX.union(nonConstN);
	    nonConstN.clear();
	    return;
	}
	    
	case TagConstants.GETSCMD: {
	    GenericVarDecl vd = ((GetsCmd)gc).v.decl;
	    Expr rhs = ((GetsCmd)gc).rhs;

	    if (!localVars.contains(vd)) {
		set.add(new ATarget(vd, new Expr[0]));		    
	    }

	    if (mentions(rhs, nonConstN)) {
		nonConstN.add(vd);
	    } else {
		nonConstN.remove(vd);
	    }
	    return;
	}

	case TagConstants.SUBGETSCMD: {
	    GenericVarDecl vd = ((SubGetsCmd)gc).v.decl;
	    Expr expr = ((SubGetsCmd)gc).index;
	    Expr rhs = ((SubGetsCmd)gc).rhs;

	    if (!localVars.contains(vd)) {
		Expr [] exprs = new Expr[1];
		
		if (mentions(expr, nonConstN)) {
		    exprs[0] = null;
		} else {
		    exprs[0] = expr;
		}
		set.add(new ATarget(vd, exprs));
	    }

	    nonConstN.add(vd);
	    return;
	}

	case TagConstants.SUBSUBGETSCMD: {
	    GenericVarDecl vd = ((SubSubGetsCmd)gc).v.decl;
	    Expr expr1 = ((SubSubGetsCmd)gc).index1;
	    Expr expr2 = ((SubSubGetsCmd)gc).index2;
	    Expr rhs = ((SubSubGetsCmd)gc).rhs;

	    if (!localVars.contains(vd)) {
		Expr [] exprs = new Expr[2];
		
		if (mentions(expr1, nonConstN)) {
		    /*
		      System.out.println("expr1 = " + expr1);
		      System.out.println("vd.id = " + vd.id + ", 
		      mentions(expr1, simpleTargets) = " + 
		      mentions(expr1, simpleTargets) + ", 
		      mentions(expr1, localVars) = " + 
		      mentions(expr1, localVars));
		    */
		    exprs[0] = null;		    
		} else {
		    exprs[0] = expr1;
		}

		if (mentions(expr2, nonConstN)) {
		    exprs[1] = null;		    
		} else {
		    exprs[1] = expr2;
		}
		set.add(new ATarget(vd, exprs));
	    }

	    nonConstN.add(vd);
	    return;
	}
	
	case TagConstants.RESTOREFROMCMD: {
	    // no targets
	    return;
	}
	
	case TagConstants.VARINCMD: {
	    VarInCmd vc = (VarInCmd)gc;

	    for (int i = 0; i < vc.v.size(); i++) {
		GenericVarDecl vd = vc.v.elementAt(i);
		localVars.add(vd);
		nonConstN.add(vd);
	    }
	    computeHelper(vc.g, set, nonConstN, nonConstX, localVars);
	    return;
	}
	

	case TagConstants.DYNINSTCMD: {
	    DynInstCmd dc = (DynInstCmd)gc;

	    computeHelper(dc.g, set, nonConstN, nonConstX, localVars);
	    return;
	}
	
	case TagConstants.SEQCMD: {
	    SeqCmd sc = (SeqCmd)gc;
	    int len = sc.cmds.size();
	    for (int i = 0; i < len; i++) {
		computeHelper(sc.cmds.elementAt(i), set, nonConstN, nonConstX, localVars);
	    }
	    return;
	}
	
	case TagConstants.YIELDCMD: {
	    YieldCmd yc = (YieldCmd) gc;
	    computeHelper(yc.desugared, set, nonConstN, nonConstX, localVars);
	    return;
	}

	case TagConstants.CALL: {
	    Call call = (Call)gc;
	    if (call.inlined) {
		computeHelper(call.desugared, set, nonConstN, nonConstX, localVars);
	    } else {
		for (Enumeration enum = call.spec.preVarMap.keys(); enum.hasMoreElements();) {
		    GenericVarDecl vd = (GenericVarDecl)enum.nextElement();
		    set.add(new ATarget(vd, new Expr[0]));
		    nonConstN.add(vd);
		    nonConstX.add(vd);
		}
	    }
	    return;
	}
	
	case TagConstants.TRYCMD: {
	    CmdCmdCmd tc = (CmdCmdCmd)gc;
	    Set tmp = new Set();

	    computeHelper(tc.g1, set, nonConstN, tmp, localVars);
	    computeHelper(tc.g2, set, tmp, nonConstX, localVars);
	    nonConstN.union(tmp);	    
	    return;
	}
	
	case TagConstants.LOOPCMD: {
	    LoopCmd lp = (LoopCmd)gc;
	    for (int i = 0; i < lp.tempVars.size(); i++) {
		GenericVarDecl vd = lp.tempVars.elementAt(i);
		localVars.add(vd);
		nonConstN.add(vd);
	    }
	    computeHelper(lp.guard, set, nonConstN, nonConstX, localVars);
	    computeHelper(lp.body, set, nonConstN, nonConstX, localVars);
	    return;
	}
	
	case TagConstants.CHOOSECMD: {
	    CmdCmdCmd cc = (CmdCmdCmd)gc;
	    Set tmp = (Set) nonConstN.clone();

	    computeHelper(cc.g1, set, tmp, nonConstX, localVars);
	    computeHelper(cc.g2, set, nonConstN, nonConstX, localVars);
	    nonConstN.union(tmp);
	    return;
	}
	
	default:
	    //@ unreachable
	    Assert.fail("UnknownTag<" + tag + ":" +
			TagConstants.toString(tag) + ">");
	    return;
	}
    }

    public String toString() {
	String s = ("[aTarget: x =" + x + " indices are");

	for (int i = 0; i < indices.length; i++) {
	    s = s + " index[" + i + "] is " + indices[i];
	}
	return s + "\n";
    }
}
