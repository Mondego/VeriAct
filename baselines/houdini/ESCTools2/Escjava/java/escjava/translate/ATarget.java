/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.ast.GenericVarDecl;
import javafe.util.*;

import escjava.ast.*;
import escjava.ast.TagConstants;

/**
 ** Infers more precise loop targets.
 **
 ** @author Cormac Flanagan
 **/

public final class ATarget {
    public GenericVarDecl x;
    public Expr [] indices; // null if not loop invariant

    private static Set targets;

    public static Set compute(GuardedCmd gc) {
	targets = new Set();
	Hashtable env = new Hashtable();
	Hashtable[] out = new Hashtable[2];
	F( gc, env, out );
	//System.out.println("atargets are " + targets.toString());
	return targets;	
    }

    private static Expr nonConst = LiteralExpr.make(TagConstants.INTLIT, 
						    new Integer(0), Location.NULL);

    ATarget(GenericVarDecl x, Expr [] indices) {
	this.x = x;
	this.indices = indices;
    }

    private static void addTarget( GenericVarDecl vd ) {
	Expr [] indices = { };
	addTarget(vd, indices);
    }

    private static void addTarget( GenericVarDecl vd, Expr index ) {
	Expr [] indices = { index };
	addTarget(vd, indices);
    }

    private static void addTarget( GenericVarDecl vd, Expr index1, Expr index2 ) {
	Expr [] indices = { index1, index2 };
	addTarget(vd, indices);
    }

    private static void addTarget( GenericVarDecl vd, Expr[] indices ) {
	targets.add( new ATarget(vd, indices) );
    }
    
    public boolean equals(Object o) {
	if( ! (o instanceof ATarget) ) return false;
	ATarget t = (ATarget)o;
	if( x != t.x ) return false;
	for(int i=0; i<indices.length; i++) {
	    if( i >= t.indices.length ||
		!(indices[i]==t.indices[i])) {
		return false;
	    }
	}
	return true;
    }

    public int hashCode() {
	return x.hashCode();
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

    /* out[0] is normal, out[1] is exceptional.
       out[i] may be null if no such exit possible.

       call may mutate in.
       Each Hashtable maps GenericVarDecls to 
       1) Expr, if var assigned a loop-invariant expr;
       2) null, if variable loop-constant
       3) nonConst, if variable not loop-invariant
     */
    private static void F(/*@ non_null */ GuardedCmd g, Hashtable in, Hashtable[] out) {

	//System.out.println("F("+g+"), in = "+in);
	
	Assert.notNull(g);
	Assert.notNull(out);

        out[0] = in;
	out[1] = null;

	if( in == null ) {
	    // this code unreachable
	    return;
	}
	
	switch (g.getTag()) {

	  case TagConstants.ASSERTCMD:
	  case TagConstants.ASSUMECMD:
	  case TagConstants.RESTOREFROMCMD:
	  case TagConstants.SKIPCMD:
	      {
		  break;
	      }

	  case TagConstants.RAISECMD:
	      {
		  out[0] = null;
		  out[1] = in;
		  break;
	      }

	  case TagConstants.LOOPCMD:
	      {
		  LoopCmd lp= (LoopCmd)g;

		  Set t = Targets.normal(g);
		  // remove t from in
		  for(Enumeration e = t.elements(); e.hasMoreElements(); ) {
		      GenericVarDecl vd = (GenericVarDecl)e.nextElement();
		      in.put(vd, nonConst);
		  }
		  
		  F( lp.guard, in, out );
		  Hashtable ex=out[1];
		  F( lp.body, out[0], out );
		  out[1] = mergeEnv( out[1], ex );

		  break;
	      }
	      
	  case TagConstants.CALL:
	      {
		  Call call= (Call)g;
		  // TBW
		  F( call.desugared, in, out );
		  break;
	      }

	  case TagConstants.GETSCMD:
	      {
		  GetsCmd gc = (GetsCmd)g;
		  addTarget( gc.v.decl );
		  extendEnv( out[0], gc.v.decl, applyEnv( in, gc.rhs ));
		  break;
	      }

	  case TagConstants.SUBGETSCMD:
	      {
		  SubGetsCmd gc = (SubGetsCmd)g;
		  addTarget( gc.v.decl, applyEnv( in, gc.index ));
		  extendEnv( out[0], gc.v.decl, null );
		  break;
	      }

	  case TagConstants.SUBSUBGETSCMD:
	      {
		  SubSubGetsCmd gc = (SubSubGetsCmd)g;
		  addTarget( gc.v.decl, applyEnv( in, gc.index1 ), applyEnv( in, gc.index2 ));
		  extendEnv( out[0], gc.v.decl, null );
		  break;
	      }

	  case TagConstants.VARINCMD:
	      {
		  VarInCmd vc = (VarInCmd)g;
		  for (int i = 0; i < vc.v.size(); i++) {
		      in.put( vc.v.elementAt(i), nonConst );
		  }
		  F( vc.g, in, out );
		  // remove from targets
		  for (int i = 0; i < vc.v.size(); i++) {
		      for(Enumeration e = targets.elements(); e.hasMoreElements(); ) {
			  ATarget t = (ATarget)e.nextElement();
			  if( t.x == vc.v.elementAt(i) ) {
			      targets.remove(t);
			      break; //enum loop
			  }
		      }
		  }
		  break;
	      }

	  case TagConstants.DYNINSTCMD:
	      {
		  DynInstCmd dc = (DynInstCmd)g;
		  F( dc.g, in, out );
		  break;
	      }

	  case TagConstants.SEQCMD:
	      {
		  SeqCmd sc = (SeqCmd)g;
		  Hashtable ex = null;
		  for (int i = 0; i < sc.cmds.size(); i++) {
		      F( sc.cmds.elementAt(i), out[0], out);
		      ex = mergeEnv( ex, out[1] );
		  }
		  out[1] = ex;
		  break;
	      }

	  case TagConstants.TRYCMD:
	      {
		  CmdCmdCmd tc = (CmdCmdCmd)g;
		  F( tc.g1, in, out );
		  Hashtable norm = out[0];
		  F( tc.g2, out[1], out );
		  out[0] = mergeEnv( out[0], norm );
		  break;
	      }

	  case TagConstants.CHOOSECMD:
	      {
		  CmdCmdCmd cc = (CmdCmdCmd)g;
		  Hashtable in2 = (Hashtable)in.clone();
		  Hashtable[] out2 = new Hashtable[2];

		  F( cc.g1, in, out );
		  F( cc.g1, in2, out2 );
		  out[0] = mergeEnv( out[0], out2[0] );
		  out[1] = mergeEnv( out[1], out2[1] );
		  break;
	      }

	  default:
	    //@ unreachable;
	    Assert.fail("Fall thru on "+g);
	}
    }

    static private void extendEnv( Hashtable env, GenericVarDecl d, Expr e) {
	if( e == null ) e = nonConst;
	env.put(d,e);
    }

    static private Hashtable mergeEnv( Hashtable a, Hashtable b ) {
	if( a == null ) return b;
	if( b == null ) return a;

	Hashtable r = new Hashtable();
	for(Enumeration e = a.keys(); e.hasMoreElements(); ) {
	    Object key = e.nextElement();
	    Object o = a.get(key);
	    if( o.equals( b.get(key) ) ) {
		r.put( key, o );
	    } else {
		r.put( key, nonConst );
	    }
	}
	for(Enumeration e = b.keys(); e.hasMoreElements(); ) {
	    Object key = e.nextElement();
	    if( a.get(key) == nonConst ) {
		r.put( key, nonConst );
	    }
	}
	return r;
    }

    /** Returns null if expr not loop constant */

    static private Expr applyEnv( Hashtable env, Expr expr ) {
	Set vars = new Set();
	computeMentionsSet( expr, vars );
	for(Enumeration e = vars.elements(); e.hasMoreElements(); ) {
	    GenericVarDecl var = (GenericVarDecl)e.nextElement();
	    Expr val = (Expr)env.get( var );

	    if( val == nonConst ) {
		return null;
	    } else if( val != null ) {
		expr = GC.subst(Location.NULL, Location.NULL, var, val, expr );
	    } 
	    // else var is loop-invariant
	}
	return expr;
    }

    private static void computeMentionsSet(ASTNode n, Set s) {
	if( n instanceof VariableAccess ) {
	    VariableAccess va = (VariableAccess)n;
	    if( va.decl != null ) {
		s.add(va.decl);
	    }
	}
	for(int i=0; i<n.childCount(); i++) {
	    Object c = n.childAt(i);
	    if( c instanceof ASTNode ) {
		computeMentionsSet((ASTNode)c,s);
	    }
	}
    }

    public /*@non_null*/String toString() {
	StringBuffer s = new StringBuffer("[aTarget: x =" + x.id + "\n");

	for (int i = 0; i < indices.length; i++) {
	    s.append("     index[" + i + "] is " + indices[i]+"\n");
		// FIXME - fix end of line character
	}
	s.append("]\n"); // FIXME - fix end of line character
	return s.toString();
    }
}
