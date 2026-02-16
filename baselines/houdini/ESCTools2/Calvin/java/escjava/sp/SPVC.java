/* Copyright Hewlett-Packard, 2002 */


package escjava.sp;

import escjava.translate.GC;
import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import javafe.util.Location;
import javafe.util.Assert;
import javafe.util.Set;

import escjava.Main;


class NXW {
  /* predicate for normal, exceptional, and wrong outcomes of a command */
    public Expr n,x,w;
}

public class SPVC {

  public static Expr compute(GuardedCmd g) {
    SPVC spvc = new SPVC();
    return spvc.computeNotWrong(g);
  }

    private Hashtable cache = new Hashtable();
    private Set cacheHit = new Set();
    private DefPredVec preds = DefPredVec.make();
    private static int predNum = 0;

    private Expr name(Expr e) {
      if( Main.useDefpred && Util.size(e) >= Main.namePCsize) {
	  Identifier predId = Identifier.intern("PREDN"+predNum);
	  predNum++;
	  preds.addElement( DefPred.make( predId, GenericVarDeclVec.make(), e ));
	  return DefPredApplExpr.make( predId, ExprVec.make() );
      } else {
	  return e;
      }
    }


  private Expr computeNotWrong(GuardedCmd root) {
      Expr r = GC.nary(TagConstants.BOOLNOT, calcNxw(root).w);
      if( Main.useDefpred ) {
	  return DefPredLetExpr.make( preds, r);
    } else {
	return r;
    }
  }

  public static Expr computeN(GuardedCmd g) {
      SPVC spvc = new SPVC();
      return spvc.calcNxw(g).n;      
  }

  private NXW calcNxw(GuardedCmd g) {

    NXW nxw = (NXW)cache.get(g);
    if (nxw != null) {
	cacheHit.add(g);
	return nxw;
    }
    nxw = new NXW();

    switch (g.getTag()) {
    case TagConstants.SKIPCMD:
      /* norm(skip) == true
	 exc(skip) == false
	 wrong(skip) == false
      */
      nxw.n = GC.truelit;
      nxw.x = GC.falselit;
      nxw.w = GC.falselit;
      break;

    case TagConstants.RAISECMD:
      /* norm(raise) == false
	 exc(raise) == true
	 wrong(raise) == false
      */
      nxw.n = GC.falselit;
      nxw.x = GC.truelit;
      nxw.w = GC.falselit;
      break;

    case TagConstants.ASSERTCMD:
      /* norm(assert E) == E
	 exc(assert E) == false
	 wrong(assert E) == !E
      */
      {
	ExprCmd ec = (ExprCmd)g;
	if (Main.strongAssertPost == 2 ||
	    (Main.strongAssertPost == 1 && isSimpleConjunction(ec.pred))) {
	  nxw.n = ec.pred;
	} else {
	  nxw.n = GC.truelit;
	}
        nxw.x = GC.falselit;
        nxw.w = GC.nary(TagConstants.BOOLNOT, ec.pred);
        break;
      }

    case TagConstants.ASSUMECMD:
      /* norm(assume E) == E
	 exc(assume E) == false
	 wrong(assume E) == false
      */
      {
	ExprCmd ec = (ExprCmd)g;
        nxw.n = ec.pred;
        nxw.x = GC.falselit;
        nxw.w = GC.falselit;
	break;
      }

      case TagConstants.CHOOSECMD:
      /* norm(A [] B) == norm(A) | norm(B)
	 exc(A [] B) == exc(A) | exc(B)
	 wrong(A [] B) == wrong(A) | wrong(B)
      */
	{
	  CmdCmdCmd cc = (CmdCmdCmd)g;
          NXW a = calcNxw(cc.g1);
          NXW b = calcNxw(cc.g2);
          nxw.n = GC.or(a.n, b.n);
          nxw.x = GC.or(a.x, b.x);
          nxw.w = GC.or(a.w, b.w);
	  break;
	}

      case TagConstants.TRYCMD:
      /* norm(A ! B) == norm(A) | (exc(A) & norm(B )
	 exc(A ! B) == exc(A) & exc(B)
	 wrong(A ! B) == wrong(A) | (exc(A) & wrong(B))
      */
	{
	  CmdCmdCmd cc = (CmdCmdCmd)g;
          NXW a = calcNxw(cc.g1);
          NXW b = calcNxw(cc.g2);
	  Expr ax = name(a.x);
          nxw.n = GC.or(a.n, GC.and(ax, b.n));
          nxw.x = GC.and(ax, b.x);
          nxw.w = GC.or(a.w, GC.and(ax, b.w));
	  break;
	}

    case TagConstants.SEQCMD:
      /* norm(A ; B) == norm(A) & norm(B)
	 exc(A ; B) == exc(A) | (norm(A) & exc(B))
	 wrong(A ; B) == wrong(A) | (norm(A) & wrong(B))
      */
      {
	SeqCmd sc = (SeqCmd)g;

	nxw.n = GC.truelit;
	nxw.x = GC.falselit;
	nxw.w = GC.falselit;


	for (int i = sc.cmds.size() -1; 0 <= i; i--) {
	  NXW temp = calcNxw(sc.cmds.elementAt(i));
	  Expr tempn = name(temp.n);
          Expr n = GC.and(tempn, nxw.n);
          Expr x = GC.or(temp.x, GC.and(tempn, nxw.x));
          Expr w = GC.or(temp.w, GC.and(tempn, nxw.w));
          nxw.n = n;
          nxw.x = x;
          nxw.w = w;
	}
	break;
      }

    default:
      //@ unreachable
      Assert.fail("Fall thru on "+g);
      return null;
    }

    cache.put(g, nxw);
    return nxw;

  }

  public static boolean isSimpleConjunction(Expr e) {
    if (e instanceof NaryExpr) {
      NaryExpr ne = (NaryExpr)e;
      if (ne.op == TagConstants.BOOLAND) {
	for (int i = 0; i < ne.exprs.size(); i++) {
	  if (! isSimpleExpr(ne.exprs.elementAt(i))) {
	    return false;
	  }
	}
	return true;
      }
    }
    return isSimpleExpr(e);
  }

  private static boolean isSimpleExpr(Expr e) {
    switch (e.getTag()) {
    case TagConstants.LABELEXPR:
      {
	LabelExpr le = (LabelExpr)e;
	return isSimpleExpr(le.expr);
      }

    case TagConstants.GUARDEXPR:
      {
	GuardExpr ge = (GuardExpr)e;
	return isSimpleExpr(ge.expr);
      }

    case TagConstants.FORALL:
    case TagConstants.EXISTS:
      return false;
	
    case TagConstants.TYPEEXPR:
    case TagConstants.LOCKSETEXPR:
    case TagConstants.TIDEXPR: // added by sanjit
    case TagConstants.MAINEXPR: // added by shaz
    case TagConstants.WITNESSEXPR: // added by shaz
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

    case TagConstants.VARIABLEACCESS:
      return true;

    case TagConstants.BOOLAND:
    case TagConstants.BOOLOR:
    case TagConstants.DTTFSA:
      return false;

    case TagConstants.SUBSTEXPR:
      { SubstExpr se = (SubstExpr)e ;
        return isSimpleExpr(se.target) && isSimpleExpr(se.val) ;
      }

    default:
      {
	if (e instanceof NaryExpr) {
	  NaryExpr ne = (NaryExpr)e;
	  for (int i = 0; i < ne.exprs.size(); i++) {
	    if (! isSimpleExpr(ne.exprs.elementAt(i))) {
	      return false;
	    }
	  }
	  return true;
	} else {
	  Assert.fail("Unexpected expr in SPVC.isSimpleExpr: "+e);
	  return false; // dummy return
        }
      }
    }
  }
}
