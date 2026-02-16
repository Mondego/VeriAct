/* Copyright 2000, 2001, Compaq Computer Corporation */


package escjava.translate;

import java.util.Enumeration;
import java.util.Hashtable;

import javafe.ast.ASTNode;
import javafe.ast.Expr;
import javafe.ast.ExprVec;
import javafe.ast.FormalParaDecl;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.Identifier;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.LocalVarDeclVec;
import javafe.ast.Stmt;
import javafe.ast.Type;
import javafe.ast.VariableAccess;
import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.StackVector;
import escjava.Main;
import escjava.ast.CmdCmdCmd;
import escjava.ast.Condition;
import escjava.ast.ConditionVec;
import escjava.ast.DecreasesInfoVec;
import escjava.ast.ExprCmd;
import escjava.ast.GetsCmd;
import escjava.ast.GuardExpr;
import escjava.ast.GuardedCmd;
import escjava.ast.GuardedCmdVec;
import escjava.ast.LabelExpr;
import escjava.ast.LoopCmd;
import escjava.ast.NaryExpr;
import escjava.ast.QuantifiedExpr;
import escjava.ast.RestoreFromCmd;
import escjava.ast.SeqCmd;
import escjava.ast.SimpleCmd;
import escjava.ast.SubGetsCmd;
import escjava.ast.SubSubGetsCmd;
import escjava.ast.SubstExpr;
import escjava.ast.TagConstants;
import escjava.ast.TypeExpr;
import escjava.ast.VarInCmd;
import escjava.tc.Types;

public final class GC {

  //// Makers for guarded commands

  public static /*@non_null*/ GuardedCmd block(/*@non_null*/ GenericVarDeclVec v, 
				 /*@non_null*/ GuardedCmd g) 
  {
    if (v.size() == 0)
      return g;
    else
      return VarInCmd.make(v, g);
  }

  public static /*@non_null*/ GuardedCmd blockL(/*@non_null*/ Stmt label, /*@non_null*/ GuardedCmd g) {
    return trycmd(g, ifcmd( nary(TagConstants.ANYEQ,
				 ecvar,
				 symlit(label, "L_")),
			    skip(),
			    raise()));
  }

  /** Requires <code>0 < cmds.size()</code>. */
  public static /*@non_null*/ GuardedCmd seq(/*@non_null*/ GuardedCmd g1,
			       /*@non_null*/ GuardedCmd g2)
    { GuardedCmd[] cvec= {g1,g2};
      return seq(GuardedCmdVec.make(cvec)); }

  public static /*@non_null*/ GuardedCmd seq(/*@non_null*/ GuardedCmd g1,
			       /*@non_null*/ GuardedCmd g2,
			       /*@non_null*/ GuardedCmd g3 )
    { GuardedCmd[] cvec= {g1,g2,g3};
      return seq(GuardedCmdVec.make(cvec)); }

  public static /*@non_null*/ GuardedCmd seq(/*@non_null*/ GuardedCmd g1,
			       /*@non_null*/ GuardedCmd g2,
			       /*@non_null*/ GuardedCmd g3,
			       /*@non_null*/ GuardedCmd g4 )
    { GuardedCmd[] cvec= {g1,g2,g3,g4};
      return seq(GuardedCmdVec.make(cvec)); }

  /** May mutilate contents of <code>cmds</code>.  The caller is expected
    * not to use <code>cmds</code> after this call.
    **/

  public static /*@non_null*/ GuardedCmd seq(/*@non_null*/ GuardedCmdVec cmds) {
    int n;
    if (!Main.options().peepOptGC) {
      n = cmds.size();
    } else {
      n = 0;
      LOOP: for (int i = 0; i < cmds.size(); i++) {
	GuardedCmd g = cmds.elementAt(i);
	int tag = g.getTag();
	switch (tag) {
	  case TagConstants.SKIPCMD:
	    // don't keep (that is, don't copy and don't increment "n"
	    break;
	
	  case TagConstants.RAISECMD:
	    cmds.setElementAt(g, n);
	    n++;
	    // don't keep any further commands, since they won't
	    // be reached anyway
	    break LOOP;
	
	  case TagConstants.ASSERTCMD:
	  case TagConstants.ASSUMECMD:
	    {
	      cmds.setElementAt(g, n);
	      n++;
	      if ((tag != TagConstants.ASSERTCMD ||
		   !Main.options().noPeepOptGCAssertFalse) &&
		  isFalse(((ExprCmd)g).pred)) {
		// don't keep any further commands, since they won't
		// be reached anyway
		break LOOP;
	      }
	      break;
	    }
	
	  default:
	    cmds.setElementAt(g, n);
	    n++;
	    break;
	}
      }
    }
    if (n == 0)
      return SimpleCmd.make(TagConstants.SKIPCMD, Location.NULL);
    if (n == 1)
      return cmds.elementAt(0);
    for (int elementsToBeRemoved = cmds.size() - n;
	 elementsToBeRemoved != 0; elementsToBeRemoved--) {
      cmds.pop();
    }
    return SeqCmd.make(cmds);
  }

  public static /*@non_null*/ GuardedCmd gets(/*@non_null*/ VariableAccess lhs, 
				/*@ nullable @*/ Expr rhs) {
    return GetsCmd.make(lhs, rhs);
  }

  public static /*@non_null*/ GuardedCmd subgets(/*@non_null*/ VariableAccess lhs, 
				   /*@non_null*/ Expr index, 
				   /*@non_null*/ Expr rhs) {
    return SubGetsCmd.make(lhs, rhs, index);
  }

  public static /*@non_null*/ GuardedCmd subsubgets(/*@non_null*/ VariableAccess lhs,
				      /*@non_null*/ Expr array, 
				      /*@non_null*/ Expr index,
				      /*@non_null*/ Expr rhs) {
    return SubSubGetsCmd.make(lhs, rhs, array, index);
  }

  public static /*@non_null*/ GuardedCmd restoreFrom(/*@non_null*/ VariableAccess lhs, 
				       /*@non_null*/ Expr rhs) {
    return RestoreFromCmd.make(lhs, rhs);
  }

  public static /*@non_null*/ GuardedCmd raise() {
    return SimpleCmd.make(TagConstants.RAISECMD, Location.NULL);
  }

  public static /*@non_null*/ GuardedCmd skip() {
    return SimpleCmd.make(TagConstants.SKIPCMD, Location.NULL);
  }

  public static /*@non_null*/ LoopCmd loop(int sLoop, int eLoop, int locHotspot, 
			     /*@non_null*/ Hashtable oldmap,
			     /*@non_null*/ ConditionVec J, 
			     /*@non_null*/ DecreasesInfoVec decs,
			     /*@non_null*/ LocalVarDeclVec skolemConsts, 
			     /*@non_null*/ ExprVec P,
			     /*@non_null*/ GenericVarDeclVec tempVars,
			     /*@non_null*/ GuardedCmd B, 
			     /*@non_null*/ GuardedCmd S) {
    return LoopCmd.make(sLoop, eLoop, locHotspot, oldmap, J, decs,
			skolemConsts, P, tempVars, B, S);
  }

  public static /*@non_null*/ GuardedCmd ifcmd(/*@non_null*/ Expr t, 
				 /*@non_null*/ GuardedCmd c, 
				 /*@non_null*/ GuardedCmd a) 
  {
    GuardedCmd thn = GC.seq(GC.assume(t), c);
    GuardedCmd els = GC.seq(GC.assumeNegation(t), a);
    return choosecmd(thn, els);
  }

  /** Pops two command sequences off <code>s</code>, call them <code>a</code>
      and <code>b</code>.  Then returns the box composition of these
      commands, that is, <code>a [] b</code>. */

  public static /*@non_null*/ GuardedCmd boxPopFromStackVector(/*@non_null*/ StackVector s) {
    GuardedCmdVec b = GuardedCmdVec.popFromStackVector(s);
    GuardedCmdVec a = GuardedCmdVec.popFromStackVector(s);
    return choosecmd(GC.seq(a), GC.seq(b));
  }

  public static /*@non_null*/ GuardedCmd choosecmd(/*@non_null*/ GuardedCmd a, 
				     /*@non_null*/ GuardedCmd b) 
  {
    if (Main.options().peepOptGC) {
      if (a.getTag() == TagConstants.ASSUMECMD && isFalse(((ExprCmd)a).pred)) {
	return b;
      }
      if (b.getTag() == TagConstants.ASSUMECMD && isFalse(((ExprCmd)b).pred)) {
	return a;
      }
      if (a.getTag() == TagConstants.ASSERTCMD && isFalse(((ExprCmd)a).pred)) {
	return a;
      }
      if (b.getTag() == TagConstants.ASSERTCMD && isFalse(((ExprCmd)b).pred)) {
	return b;
      }
    }
    return CmdCmdCmd.make(TagConstants.CHOOSECMD, a, b);
  }

  public static /*@non_null*/ GuardedCmd trycmd(/*@non_null*/ GuardedCmd c, 
				  /*@non_null*/ GuardedCmd a) {
    if (Main.options().peepOptGC) {
      if (a.getTag() == TagConstants.RAISECMD) {
	return c;
      }
      // It would be nice to do the following:
      //     if (!canRaise(c)) {
      //       return c;
      //     }
      // but we don't keep track of the possible outcomes of expressions,
      // and thus we'd end up spending quadratic time.  Instead, we do:
      switch (c.getTag()) {
	case TagConstants.SKIPCMD:
	  return c;
	
	case TagConstants.RAISECMD:
	  return a;
	
	case TagConstants.GETSCMD:
	case TagConstants.SUBGETSCMD:
	case TagConstants.SUBSUBGETSCMD:
	case TagConstants.RESTOREFROMCMD:
	case TagConstants.ASSERTCMD:
	case TagConstants.ASSUMECMD:
	  return c;
	
	default:
	  break;
      }
    }
    return CmdCmdCmd.make(TagConstants.TRYCMD, c, a);
  }

  /** Returns <code>true</code> when <code>e</code> is a boolean
    * literal expression whose value is <code>b</code>.
    **/
  public static boolean isBooleanLiteral(/*@non_null*/ Expr e, boolean b) {
    if (e.getTag() == TagConstants.BOOLEANLIT) {
      LiteralExpr le = (LiteralExpr)e;
      if (le.value != null) {
	Boolean bb = (Boolean)le.value;
	return bb.booleanValue() == b;
      }
    } else if( e.getTag() == TagConstants.BOOLNOT) {
	return isBooleanLiteral( ((NaryExpr)e).exprs.elementAt(0), !b );
    }
    return false;
  }

  /** Returns <code>true</code> only if <code>e</code> represents an
    * expression equivalent to <code>false</code>.  This method may
    * return <code>false</code> under any circumstance, but tries to
    * use cheap methods to figure out whether <code>e</code> is equivalent
    * to <code>false</code>, in which case <code>true</code> is returned.
    **/

  public static boolean isFalse(/*@non_null*/ Expr e) {
    // first strip off any Simplify label
    while (e.getTag() == TagConstants.LABELEXPR) {
      LabelExpr le = (LabelExpr)e;
      e = le.expr;
    }

    return isBooleanLiteral(e, false);
  }

 /** Creates an assert, assume, or skip command, depending on
     the kind of given error name and locations, and depending on what
     checks are enabled where.  There are two versions of the
     <code>check</code> method.

     In the first version, <code>errorName</code> is the error name
     (that is, the tag constant of the type of check), <code>pred</code>
     evaluates to <code>false</code> if the check goes wrong,
     <code>locUse</code> is the source location of the command or
     expression that can go wrong, and <code>locPragmaDecl</code> is
     the location of the associated pragma, if any (or <code>Location.NULL</code>
     if not applicable).

     In the second version, <code>errorName</code>, <code>pred</code>,
     and <code>locPragmaDecl</code> are taken from the components of the
     given condition <code>cond</code>.  */

  public static /*@non_null*/ GuardedCmd check(int locUse,
				 int errorName, 
				 /*@non_null*/ Expr pred,
				 int locPragmaDecl) {
    return check(locUse, errorName, pred, locPragmaDecl, null);
  }

  public static /*@non_null*/ GuardedCmd check(int locUse,
				 int errorName, 
				 /*@non_null*/ Expr pred,
				 int locPragmaDecl,
				 /*@ nullable @*/ Object aux) {
	return check(locUse,errorName, pred, locPragmaDecl, Location.NULL,aux);
  }

  public static /*@non_null*/ GuardedCmd check(int locUse,
				 int errorName, 
				 /*@non_null*/ Expr pred,
				 int locPragmaDecl,
				 int auxLoc,
				 /*@ nullable @*/ Object aux) {
    //Assert.notFalse(locUse != Location.NULL);
    if (Main.options().guardedVC && locPragmaDecl != Location.NULL) {
      pred = GuardExpr.make(pred, locPragmaDecl);
    }
    switch( NoWarn.getChkStatus( errorName, locUse, locPragmaDecl) ) {
    case TagConstants.CHK_AS_ASSUME:
      LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
      return assume(pred);
    case TagConstants.CHK_AS_ASSERT:
      LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
      return assertPredicate(locUse, errorName, pred, locPragmaDecl, auxLoc, aux);
    case TagConstants.CHK_AS_SKIP:
      return skip();
    default:
      Assert.fail("unreachable");
      return null; // dummy return
    }

  }


  /** See description of <code>check</code> above. */

  public static /*@non_null*/ GuardedCmd check(int locUse, /*@non_null*/ Condition cond) {
    Assert.notFalse(locUse != Location.NULL);
    return check(locUse, cond.label, cond.pred, cond.locPragmaDecl, null);
  }

  /** See description of <code>check</code> above. */

  public static /*@non_null*/ GuardedCmd check(int locUse, 
				 /*@non_null*/ Condition cond, 
				 /*@ nullable @*/ Object aux) 
  {
    Assert.notFalse(locUse != Location.NULL);
    return check(locUse, cond.label, cond.pred, cond.locPragmaDecl, aux);
  }

  //@ requires label != TagConstants.CHKFREE;
  public static /*@non_null*/ Condition condition(int label, 
				    /*@non_null*/ Expr pred,
				    int locPragmaDecl) {
    Assert.notFalse(label != TagConstants.CHKFREE);
    return Condition.make(label, pred, locPragmaDecl);
  }

  public static /*@non_null*/ Condition freeCondition(/*@non_null*/ Expr pred, int locPragmaDecl) {
    return Condition.make(TagConstants.CHKFREE, pred, locPragmaDecl);
  }

  public static /*@non_null*/ Condition assumeCondition(/*@non_null*/ Expr pred, int locPragmaDecl) {
    return Condition.make(TagConstants.CHKASSUME, pred, locPragmaDecl);
  }

  //@ requires locPragmaDecl != Location.NULL;
  public static /*@non_null*/ GuardedCmd assumeAnnotation(int locPragmaDecl,
					    /*@non_null*/ Expr p) {
    if (Main.options().guardedVC && locPragmaDecl != Location.NULL) {
      p = GuardExpr.make(p, locPragmaDecl);
    }
    LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
    return assume(p);
  }

  public static /*@non_null*/ GuardedCmd assume(/*@non_null*/ Expr p) 
  {
    if (Main.options().peepOptGC && isBooleanLiteral(p, true)) {
      return skip();
    }
    if (p.getTag() == TagConstants.BOOLAND && (p instanceof NaryExpr)
           && ((NaryExpr)p).exprs.size() > 1) {
       // An optimization that makes ASSUME commands simpler - unpacks an AND
       // into multiple ASSUMEs to make reading easier.
        NaryExpr ne = (NaryExpr)p;
        GuardedCmdVec gcv = GuardedCmdVec.make(ne.exprs.size());
        for (int i=0; i<ne.exprs.size(); ++i) {
          Expr e = ne.exprs.elementAt(i);
          ExprCmd a = ExprCmd.make(TagConstants.ASSUMECMD, e, Location.NULL);
          gcv.addElement(a);
        }
        return GC.seq(gcv);
    }
    if (p.getTag() == TagConstants.FORALL && (p instanceof QuantifiedExpr)) {
      // This is an optimization that changes a Forall of a boolean and into
      // a sequence of foralls of the conjuncts.
      QuantifiedExpr qe = (QuantifiedExpr)p;
      if (qe.expr.getTag() == TagConstants.BOOLAND) {
        ExprVec ev = ((NaryExpr)qe.expr).exprs;
        if (ev.size() > 1) {
          GuardedCmdVec gcv = GuardedCmdVec.make(ev.size());
          for (int i=0; i<ev.size(); ++i) {
            Expr e =  QuantifiedExpr.make(qe.sloc,qe.eloc,qe.quantifier,
                         qe.vars,qe.rangeExpr,ev.elementAt(i),qe.nopats,qe.pats);
            ExprCmd a = ExprCmd.make(TagConstants.ASSUMECMD, e, Location.NULL);
            gcv.addElement(a);
          }
          return GC.seq(gcv);
        }
      }
    }
    return ExprCmd.make(TagConstants.ASSUMECMD, p, Location.NULL);
  }

  public static /*@non_null*/ GuardedCmd assumeNegation(/*@non_null*/ Expr p) {
    Expr non_p = not(p.getStartLoc(), p.getEndLoc(), p);
    return assume(non_p);
  }

  public static /*@non_null*/ GuardedCmd fail() {
    return assume(falselit);
  }

  private static int assertContinueCounter = 0;

//  //@ requires locUse != Location.NULL;
//  private static GuardedCmd assertPredicate(int locUse,
//                                            int errorName, Expr pred,
//                                            int locPragmaDecl,
//                                            Object aux) {
//	return assertPredicate(locUse, errorName, pred, locPragmaDecl,
//		Location.NULL, aux);
//  }

  private static /*@non_null*/ GuardedCmd assertPredicate(int locUse,
                                            int errorName, 
					    /*@non_null*/ Expr pred,
                                            int locPragmaDecl,
					    int auxLoc,
                                            /*@non_null*/ Object aux) {
    if (Main.options().assertContinue) {
      Identifier idn = Identifier.intern("assertContinue" +
					 assertContinueCounter);
      assertContinueCounter++;
      VariableAccess acVar = makeVar(idn, locUse);
      acVar.loc = Location.NULL;
      pred = or(pred, acVar);
    }
    if (errorName != TagConstants.CHKQUIET) {
	String name = TagConstants.toString(errorName);
	if (aux != null && Main.options().suggest) {
	  int n = AuxInfo.put(aux);
	  name += "/" + n;
	}
	Identifier en = makeLabel(name,locPragmaDecl,auxLoc,locUse);
	pred = LabelExpr.make(locUse, locUse, false, en, pred);
    }
    return ExprCmd.make(TagConstants.ASSERTCMD, pred, locUse);
  }

  public static /*@non_null*/ Identifier makeLabel(/*@non_null*/ String name, 
				     int locPragmaDecl, int auxLoc, int locUse) {
	String lab = name;
	if (locPragmaDecl != Location.NULL) {
	    lab = lab + ":" + UniqName.locToSuffix(locPragmaDecl);
	}
	if (auxLoc != Location.NULL) {
	    lab = lab + ":" + UniqName.locToSuffix(auxLoc);
	}
	if (locUse != Location.NULL)
	    lab += "@" + UniqName.locToSuffix(locUse);
	return Identifier.intern(lab);
  }

  public static /*@non_null*/ Identifier makeLabel(/*@non_null*/ String name, 
				     int locPragmaDecl, int locUse) {
	String lab = name;
	if (locPragmaDecl != Location.NULL) {
	    lab = lab + ":" + UniqName.locToSuffix(locPragmaDecl);
	}
	if (locUse != Location.NULL)
	    lab += "@" + UniqName.locToSuffix(locUse);
	return Identifier.intern(lab);
  }

  public static /*@non_null*/ Identifier makeFullLabel(/*@non_null*/ String name, 
					 int locPragmaDecl, int locUse) 
  {
	String lab = name;
	if (locPragmaDecl != Location.NULL) {
	    lab = lab + ":" + UniqName.locToSuffix(locPragmaDecl,false);
	}
	if (locUse != Location.NULL)
	    lab += "@" + UniqName.locToSuffix(locUse,false);
	return Identifier.intern(lab);
  }
	    
  //+@ requires subst.keyType == \type(GenericVarDecl) ;
  //+@ requires subst.elementType <: \type(Expr) ;
  public static /*@non_null*/ Expr subst(int sloc, int eloc, /*@non_null*/ Hashtable subst, /*@non_null*/ Expr target)
    {
      if ( !Main.options().lazySubst ) {
	// perform substitution eagerly

	return Substitute.doSubst( subst, target );

      } else {
	// create lazy substitutions
	
	for(Enumeration e = subst.keys(); e.hasMoreElements(); )
	  {
	    GenericVarDecl vd = (GenericVarDecl)e.nextElement();
	    Expr to = (Expr)subst.get( vd );
	    target = subst(sloc, eloc, vd, to, target);
	  }
	return target;
      }
    }

  public static /*@non_null*/ Expr subst(int sloc, int eloc,
			   /*@non_null*/ GenericVarDecl var, /*@non_null*/ Expr val, /*@non_null*/ Expr target) {
    if ( !Main.options().lazySubst ) {
      // perform substitution eagerly
      Hashtable t = new Hashtable();
      t.put( var, val );
      return subst( sloc, eloc, t, target );
    } else {
      return SubstExpr.make( sloc, eloc, var, val, target );
    }
  }

  public static /*@non_null*/ Expr subst(/*@non_null*/ GenericVarDecl var, /*@non_null*/ Expr val, /*@non_null*/ Expr target) {
    return subst( Location.NULL, Location.NULL, var, val, target );
  }


  //// Makers for literals

  public static final /*@non_null*/ Expr nulllit =
    LiteralExpr.make(TagConstants.NULLLIT, null, Location.NULL);

  public static final /*@non_null*/ Expr zerolit =
    LiteralExpr.make(TagConstants.INTLIT, new Integer(0), Location.NULL);

  public static final /*@non_null*/ Expr onelit =
    LiteralExpr.make(TagConstants.INTLIT, new Integer(1), Location.NULL);

  public static final /*@non_null*/ Expr truelit =
    LiteralExpr.make(TagConstants.BOOLEANLIT,Boolean.TRUE,Location.NULL);

  public static final /*@non_null*/ Expr falselit =
    LiteralExpr.make(TagConstants.BOOLEANLIT,Boolean.FALSE,Location.NULL);

  public static final /*@non_null*/ Expr dzerolit =
    LiteralExpr.make(TagConstants.DOUBLELIT, new Double(0.0), Location.NULL);

  public static /*@non_null*/ Expr symlit(/*@nullable*/ String s) {
    return LiteralExpr.make(TagConstants.SYMBOLLIT, s, Location.NULL);
  }

  public static /*@non_null*/ Expr symlit(/*@non_null*/ Stmt s, /*@non_null*/ String prefix) {
    return symlit(prefix + UniqName.locToSuffix(s.getStartLoc()));
  }

  public static /*@non_null*/ Expr zeroequiv(/*@non_null*/ Type t) {
    switch (t.getTag()) {
    case TagConstants.ARRAYTYPE:
    case TagConstants.NULLTYPE:
    case TagConstants.TYPENAME:
    case TagConstants.TYPESIG:
      return nulllit;

    case TagConstants.BOOLEANTYPE:
      return falselit;

    case TagConstants.INTTYPE:
    case TagConstants.LONGTYPE:
    case TagConstants.CHARTYPE:
    case TagConstants.BYTETYPE:
    case TagConstants.SHORTTYPE:
      return zerolit;

    case TagConstants.DOUBLETYPE:
    case TagConstants.FLOATTYPE:
      return dzerolit;
    }
    /*@ unreachable; */
    Assert.fail("Bad tag");
    return null;
  }


  //// Makers for variables

  public static /*@non_null*/ VariableAccess makeVar(/*@non_null*/ Identifier name, int locId) {
    LocalVarDecl v
      = LocalVarDecl.make(0, null, name, Types.anyType, locId,
			  null, Location.NULL);
    return VariableAccess.make(name, Location.NULL, v);
  }

  public static /*@non_null*/ VariableAccess makeVar(/*@non_null*/ String name, int locId) {
    return makeVar(Identifier.intern(name), locId);
  }

  public static /*@non_null*/ VariableAccess makeFormalPara(/*@non_null*/ String name, 
					      /*@non_null*/ Type type,
					      int locId) {
    Identifier nameId = Identifier.intern(name);
    FormalParaDecl v
      = FormalParaDecl.make(0, null, nameId, type, locId);
    return VariableAccess.make( Identifier.intern(name), Location.NULL, v);
  }


  public static /*@non_null*/ VariableAccess makeVar(/*@non_null*/ String name) {
    return makeVar(name, Location.NULL);
  }

  public static /*@non_null*/ VariableAccess makeVar(/*@non_null*/ Identifier name) {
    return makeVar(name, Location.NULL);
  }

  public static /*@non_null*/ VariableAccess makeFormalPara(/*@non_null*/ String name, 
					      /*@non_null*/ Type type) {
    return makeFormalPara(name, type, Location.NULL);
  }

  public static /*@non_null*/ VariableAccess makeFormalPara(/*@non_null*/ String name) {
    return makeFormalPara(name, Types.anyType);
  }


  public static final /*@non_null*/ VariableAccess allocvar = makeVar("alloc",
						  UniqName.specialVariable);
  public static final /*@non_null*/ VariableAccess ecvar = makeVar("EC",
					       UniqName.specialVariable);
  public static final /*@non_null*/ VariableAccess elemsvar = makeVar("elems",
						  UniqName.specialVariable);
  public static final /*@non_null*/ VariableAccess resultvar = makeVar("RES",
    					   UniqName.specialVariable);
  public static final /*@non_null*/ VariableAccess xresultvar = makeVar("XRES",
						    UniqName.specialVariable);
  public static final /*@non_null*/ VariableAccess objectTBCvar = makeVar("objectToBeConstructed",
							    UniqName.specialVariable);
  public static final /*@non_null*/ VariableAccess statevar = makeVar("state",
						  UniqName.specialVariable);

  // LSvar is not final because it is temporarily updated at
  // synchronized expressions. See trExpr
     public static /*@non_null*/ VariableAccess LSvar = makeVar("LS",
    			       UniqName.specialVariable);


  /*
   * The type of thisvar (thisvar.decl.type) is set by
   * Translate.trBody.  It is also temporarily changed by
   * GetSpec.trMethodDecl.
   */
  //@ invariant thisvar.decl.type instanceof TypeSig;
  public static final /*@non_null*/ VariableAccess thisvar =
      makeFormalPara("this", javafe.tc.Types.javaLangObject(),
		     UniqName.specialVariable);


  /*
   * These handle case 5 of ESCJ 23b:
   */
  public static final /*@non_null*/ Expr ec_throw = symlit("ecThrow");
  public static final /*@non_null*/ Expr ec_return = symlit("ecReturn");

  //// Makers for expressions

  public static /*@non_null*/ Expr typeExpr(/*@non_null*/ Type t)
    { return TypeExpr.make(Location.NULL, Location.NULL, t); }

  public static /*@non_null*/ Expr cast(/*@non_null*/ Expr e, 
			  /*@non_null*/ Type t) 
  {
	if (e instanceof LiteralExpr)
		return LiteralExpr.cast((LiteralExpr)e,
			t == Types.doubleType ? TagConstants.DOUBLELIT :
			t == Types.floatType ? TagConstants.FLOATLIT :
			t == Types.longType ? TagConstants.LONGLIT :
				TagConstants.INTLIT);
	return nary(TagConstants.CAST, e, typeExpr(t));
  }

  // Various forms of nary()

  public static /*@non_null*/ Expr nary(int tag, /*@non_null*/ Expr e) {
    return nary(Location.NULL, Location.NULL, tag, e);
  }

  public static /*@non_null*/ Expr nary(int sloc, int eloc, int tag,
			  /*@non_null*/ Expr e) {
    Expr[] args = { e };
    return nary( sloc, eloc, tag, ExprVec.make(args));
  }

  public static /*@non_null*/ Expr nary(int tag,
			  /*@non_null*/ Expr e1, /*@non_null*/ Expr e2) {
    return nary(Location.NULL, Location.NULL, tag, e1, e2);
  }

  //@ ensures \result != null;
  public static /*@non_null*/ Expr nary(int sloc, int eloc, int tag,
			  /*@non_null*/ Expr e1, /*@non_null*/ Expr e2) {
    Expr[] args = { e1, e2 };
    return nary(sloc,eloc,tag, ExprVec.make(args));
  }

  public static /*@non_null*/ Expr nary(int tag, /*@non_null*/ Expr e1,
			  /*@non_null*/ Expr e2, /*@non_null*/ Expr e3) {
    return nary(Location.NULL, Location.NULL, tag, e1, e2, e3);
  }

  public static /*@non_null*/ Expr nary(int sloc, int eloc, int tag,
			  /*@non_null*/ Expr e1, /*@non_null*/ Expr e2,
                          /*@non_null*/ Expr e3) {
    Expr[] args = { e1, e2, e3 };
    return nary(sloc,eloc,tag, ExprVec.make(args));
  }

  public static /*@non_null*/ Expr nary(int tag, /*@non_null*/ ExprVec ev) {
	return nary(Location.NULL, Location.NULL, tag, ev);
  }

  public static /*@non_null*/ Expr nary(/*@non_null*/ Identifier id, /*@non_null*/ ExprVec ev) {
	Expr e = nary(Location.NULL, Location.NULL, TagConstants.METHODCALL, ev);
	((NaryExpr)e).methodName = id;
	return e;
  }

  public static /*@non_null*/ Expr nary(/*@non_null*/ Identifier id, /*@non_null*/  ASTNode symbol, /*@non_null*/ ExprVec ev) {
	Expr e = nary(Location.NULL, Location.NULL, TagConstants.METHODCALL, ev);
	((NaryExpr)e).methodName = id;
	((NaryExpr)e).symbol = symbol;
	return e;
  }

  public static /*@non_null*/ Expr nary(/*@non_null*/ Identifier id, /*@non_null*/ Expr e) {
	ExprVec ev = ExprVec.make(1);
	ev.addElement(e);
	return nary(id,ev);
}

  public static /*@non_null*/ Expr nary(/*@non_null*/ Identifier id, 
			  /*@non_null*/ Expr e1, 
			  /*@non_null*/ Expr e2) 
  {
	ExprVec ev = ExprVec.make(2);
	ev.addElement(e1);
	ev.addElement(e2);
	return nary(id,ev);
  }

  public static /*@non_null*/ Expr nary(int sloc, int eloc, 
			  /*@non_null*/ Identifier id, 
			  /*@non_null*/ ExprVec ev) 
  {
	Expr e = nary(sloc, eloc, TagConstants.METHODCALL, ev);
	((NaryExpr)e).methodName = id;
	return e;
  }

  public static /*@non_null*/ Expr nary(int sloc, int eloc, int tag, 
			  /*@non_null*/ ExprVec ev) 
  {
    if( Main.options().peepOptE ) {
      // Do some optimizations ...

      switch( tag ) {
	case TagConstants.BOOLAND:
	case TagConstants.BOOLANDX:
	  {
	    ExprVec w = ExprVec.make( ev.size() );
	    if( selectiveAdd( w, ev, truelit, falselit,
			      tag ) )
	      {
		return falselit;
	      }
	
	    if( w.size() == 0 )
	      return truelit;
	    else if( w.size() == 1 )
	      return w.elementAt(0);
	    else
	      return NaryExpr.make( sloc, eloc, tag, null, w);
	  }

	case TagConstants.BOOLOR:
	  {
	    ExprVec w = ExprVec.make( ev.size() );
	    if( selectiveAdd( w, ev, falselit, truelit,
			      TagConstants.BOOLOR ) )
	      {
		return truelit;
	      }

	    if( w.size() == 0 )
	      return falselit;
	    else if( w.size() == 1 )
	      return w.elementAt(0);
	    else
	      return NaryExpr.make( sloc, eloc, TagConstants.BOOLOR, null, w);
	  }

	case TagConstants.BOOLIMPLIES:
	  {
	    Expr c0 = ev.elementAt(0);
	    Expr c1 = ev.elementAt(1);

	    if( Main.options().bubbleNotDown ) {
	      return or( sloc, eloc,
			 not( sloc, eloc, c0 ),
			 c1 );
	    } else {
		// Change a ==> (b ==> c) to (a ^ b) ==> c
		if(c1.getTag() == TagConstants.BOOLIMPLIES ) {
		    NaryExpr ne = (NaryExpr)c1;
		    return implies( sloc, eloc,
				    and( sloc, eloc, c0, ne.exprs.elementAt(0) ),
				    ne.exprs.elementAt(1) );
		} else if (isBooleanLiteral(c0, false)) {
		// false ==> X --> true
		return GC.truelit;
	      } else if (isBooleanLiteral(c1, true)) {
		// X ==> true --> true
		return GC.truelit;
	      } else if (isBooleanLiteral(c0, true)) {
		// true ==> X --> X
		return c1;
	      } else if (isBooleanLiteral(c1, false)) {
		// X ==> false --> !X
		return nary(sloc, eloc, TagConstants.BOOLNOT, c0);
	      } else {
		break; // go to default case
	      }
	    }
	  }

	case TagConstants.BOOLNOT:
	  // Change (not (and a b)) -> (or (not a) (not b)), etc
	  // Also (not (not a)) -> a
	  {
	    Expr c0 = ev.elementAt(0);
	    if (isBooleanLiteral(c0, false)) {
	      return truelit;
	    } else if (isBooleanLiteral(c0, true)) {
	      return falselit;
	    } else if ( c0.getTag() == TagConstants.BOOLNOT ) {
		return ((NaryExpr)c0).exprs.elementAt(0);
	    } else if( Main.options().bubbleNotDown ) {
	      switch( c0.getTag() ) {
	      case TagConstants.BOOLOR:
	      case TagConstants.BOOLAND:
	      case TagConstants.BOOLANDX:
		{
		  ExprVec w = ((NaryExpr)c0).exprs;
		  ExprVec r = ExprVec.make();
		  for(int i=0; i<w.size(); i++) {
		    r.addElement( not( sloc, eloc, w.elementAt(i)));
		  }
		  return nary( sloc, eloc,
			       c0.getTag() == TagConstants.BOOLOR ?
			         TagConstants.BOOLAND : TagConstants.BOOLOR,
			       r );
		}
	      }
	    }

	    break; // go to default case
	  }

	case TagConstants.ANYEQ:
	  // Change (ANYEQ a a) -> true
	  {
	      if( ev.size() == 2 &&
		  ev.elementAt(0) instanceof VariableAccess &&
		  ev.elementAt(1) instanceof VariableAccess &&
		  ((VariableAccess)ev.elementAt(0)).decl ==
		  ((VariableAccess)ev.elementAt(1)).decl ) {
		  return GC.truelit;
	      }

	      if( ev.size() == 2 &&
		  ev.elementAt(0) instanceof LiteralExpr &&
		  ev.elementAt(1) instanceof LiteralExpr &&
		  ((LiteralExpr)ev.elementAt(0)).value.equals(
			 ((LiteralExpr)ev.elementAt(1)).value )) {
		  return GC.truelit;
	      }

	    break; // go to default case
	  }

      }
    }

    // No special case, so do default
    return NaryExpr.make(sloc,eloc,tag, null, ev);
  }


  //// Makers for other GCExpr nodes

  public static /*@non_null*/ Expr select(/*@non_null*/ Expr c0, /*@non_null*/ Expr c1) {
    return nary(TagConstants.SELECT, c0, c1);
  }

  public static /*@non_null*/ Expr not(/*@non_null*/ Expr c) {
    return not(Location.NULL, Location.NULL, c);
  }

  public static /*@non_null*/ Expr not(int sloc, int eloc, /*@non_null*/ Expr c) {
    return nary(sloc, eloc, TagConstants.BOOLNOT, c);
  }

  public static /*@non_null*/ Expr and(/*@non_null*/ Expr c1, /*@non_null*/ Expr c2) {
    return and(Location.NULL, Location.NULL, c1, c2);
  }

  public static /*@non_null*/ Expr andx(/*@non_null*/ Expr c1, /*@non_null*/ Expr c2) {
    ExprVec es = ExprVec.make(2);
    es.addElement(c1);
    es.addElement(c2);
    return nary(Location.NULL, Location.NULL, TagConstants.BOOLANDX, es);
  }

  public static /*@non_null*/ Expr and(int sloc, int eloc, /*@non_null*/ Expr c1, /*@non_null*/ Expr c2) {
    Expr[] es = {c1, c2};
    return and( sloc, eloc, ExprVec.make(es) );
  }

  public static /*@non_null*/ Expr and(/*@non_null*/ ExprVec v) {
    return and(Location.NULL, Location.NULL, v);
  }

  public static /*@non_null*/ Expr and(int sloc, int eloc, /*@non_null*/ ExprVec v) {
    return nary( sloc, eloc, TagConstants.BOOLAND, v );
  }

  public static /*@non_null*/ Expr or(/*@non_null*/ Expr c1, /*@non_null*/ Expr c2) {
    return or(Location.NULL, Location.NULL, c1, c2);
  }

  public static /*@non_null*/ Expr or(int sloc, int eloc, 
			/*@non_null*/ Expr c1, /*@non_null*/ Expr c2) {
    Expr[] es = {c1, c2};
    return or( sloc, eloc, ExprVec.make(es) );
  }

  public static /*@non_null*/ Expr or(/*@non_null*/ ExprVec v) {
    return or(Location.NULL, Location.NULL, v);
  }

  public static /*@non_null*/ Expr or(int sloc, int eloc, /*@non_null*/ ExprVec v) {
    return nary( sloc, eloc, TagConstants.BOOLOR, v );
  }

  public static /*@non_null*/ Expr implies(/*@non_null*/ Expr c0, /*@non_null*/ Expr c1) {
    return implies( Location.NULL, Location.NULL, c0, c1 );
  }

  public static /*@non_null*/ Expr implies(int sloc, int eloc, /*@non_null*/ Expr c0, /*@non_null*/ Expr c1) {
    return nary( sloc, eloc, TagConstants.BOOLIMPLIES, c0, c1);
  }
  
  public static /*@non_null*/ Expr equiv(/*@non_null*/ Expr c0, /*@non_null*/ Expr c1) {
      return equiv(Location.NULL, Location.NULL, c0, c1);
  }
  
  public static /*@non_null*/ Expr equiv(int sloc, int eloc, /*@non_null*/ Expr c0, /*@non_null*/ Expr c1) {
      return nary(sloc, eloc, TagConstants.BOOLEQ, c0, c1);
  }

  public static /*@non_null*/ Expr forall(/*@non_null*/ GenericVarDecl v, /*@non_null*/ Expr e) {
    return forall( Location.NULL, Location.NULL, v, GC.truelit, e, null );
  }

  public static /*@non_null*/ Expr forall(/*@non_null*/ GenericVarDecl v, /*@nullable*/ Expr range, /*@non_null*/ Expr e) {
    return forall( Location.NULL, Location.NULL, v, range, e, null );
  }

  public static /*@non_null*/ Expr forall(/*@non_null*/ GenericVarDeclVec v, 
			    /*@nullable*/  Expr range, 
			    /*@non_null*/ Expr e) 
  {
    return quantifiedExpr( Location.NULL, Location.NULL, 
		TagConstants.FORALL, v, range, e, null, null );
  }

  public static /*@non_null*/ Expr forall(/*@non_null*/ GenericVarDecl v, 
			    /*@non_null*/ Expr e, 
			    /*@nullable*/ ExprVec nopats) 
  {
    return forall( Location.NULL, Location.NULL, v, GC.truelit, e, nopats );
  }

  public static /*@non_null*/ Expr forallwithpats(/*@non_null*/ GenericVarDecl v, 
				    /*@non_null*/ Expr e, 
				    /*@non_null*/ ExprVec pats) {
    return quantifiedExpr( Location.NULL, Location.NULL, 
		TagConstants.FORALL, v, GC.truelit, e, null, pats );
  }

  public static /*@non_null*/ Expr forall(int sloc, int eloc, /*@non_null*/ GenericVarDecl v,
			    /*@nullable*/ Expr range, /*@non_null*/ Expr e) {
    return forall(sloc, eloc, v, range, e, null);
  }

  public static /*@non_null*/ Expr forall(int sloc, int eloc, 
			    /*@non_null*/ GenericVarDecl v, 
			    /*@nullable*/ Expr range, 
			    /*@non_null*/ Expr e,
			    /*@nullable*/ExprVec nopats) {
    Assert.notNull(v);
    Assert.notNull(e);

    if( Main.options().peepOptE ) {

      // Change forall (a) ((a==b) ==> e) -> e[b/a] if b a variable

      if( e.getTag() == TagConstants.BOOLIMPLIES ) {

	NaryExpr nary = (NaryExpr)e;
	Expr impliesLhs = nary.exprs.elementAt(0);
	Expr impliesRhs = nary.exprs.elementAt(1);
	switch( impliesLhs.getTag() ) {
	  case TagConstants.ANYEQ:
	  case TagConstants.BOOLEQ:
	  case TagConstants.INTEGRALEQ:
	  case TagConstants.REFEQ:
	  case TagConstants.TYPEEQ:
	
	    NaryExpr lhsNary = (NaryExpr)impliesLhs;
	    Expr eqLhs = lhsNary.exprs.elementAt(0);
	    Expr eqRhs = lhsNary.exprs.elementAt(1);
	    if( eqLhs.getTag() == TagConstants.VARIABLEACCESS
		&& ((VariableAccess)eqLhs).decl == v
		&& isSimple( eqRhs ))
	      {
		// Can replace the forall with a substitution
		return subst( v, eqRhs, impliesRhs );
	      }
	}
      }
    }

    // could not do the substitution
    return quantifiedExpr(sloc, eloc, TagConstants.FORALL, v, range, e, nopats, null);
  }

  public static /*@non_null*/ Expr quantifiedExpr(int sloc, int eloc, int tag,
				    /*@non_null*/ GenericVarDecl v, 
				    /*@nullable*/ Expr range, 
				    /*@non_null*/ Expr e,
				    /*@nullable*/ExprVec nopats, 
				    /*@nullable*/ExprVec pats)
    {
      GenericVarDeclVec vs = GenericVarDeclVec.make();
      vs.addElement(v);
      return quantifiedExpr(sloc, eloc, tag, vs, range, e, nopats, pats );
    }

  public static /*@non_null*/ Expr quantifiedExpr(int sloc, int eloc, int tag,
				    /*@non_null*/ GenericVarDeclVec vs, 
				    /*@nullable*/	Expr range, 
				    /*@non_null*/ Expr e,
				    /*@nullable*/	ExprVec nopats, 
				    /*@nullable*/	ExprVec pats)
    {
      Assert.notFalse( tag == TagConstants.FORALL
		       || tag == TagConstants.EXISTS );

      if( tag == TagConstants.FORALL && vs.size() == 0 ) {
	  // empty forall
	  return e;
      }

      if( Main.options().peepOptE ) {

	// change Q(a)Q(b)e -> Q(a b) e, where Q is a quantifier

	if( e.getTag() == tag ) {
	  QuantifiedExpr qe = (QuantifiedExpr)e;
	  GenericVarDeclVec copy = vs.copy();
	  copy.append( qe.vars );
	  if (qe.nopats != null) {
	    if (nopats == null) {
	      nopats = qe.nopats;
	    } else {
	      nopats = nopats.copy();
	      nopats.append(qe.nopats);
	    }
	  }
          if (range == null) range = qe.rangeExpr;
          else if (qe.rangeExpr != null) range = GC.and(range,qe.rangeExpr);
	  return QuantifiedExpr.make( sloc, eloc, tag, copy, 
				      range, qe.expr,
				      nopats, qe.pats );
	}
      }

      // No optimization done
      return QuantifiedExpr.make( sloc, eloc, tag, vs, range, e, nopats, pats );
    }

  public static boolean isSimple(/*@non_null*/ Expr e) {
    switch( e.getTag() ) {
      case TagConstants.VARIABLEACCESS:
      case TagConstants.TYPEEXPR:
      case TagConstants.BOOLEANLIT:
      case TagConstants.CHARLIT:
      case TagConstants.DOUBLELIT:
      case TagConstants.FLOATLIT:
      case TagConstants.INTLIT:
      case TagConstants.LONGLIT:
      case TagConstants.NULLLIT:
      case TagConstants.STRINGLIT:
      case TagConstants.SYMBOLLIT:
	return true;

      default:
	return false;
    }
  }

  /**
   * Adds elements to <code>to</code> from <code>from</code>.
   * Elements equal to bot are dropped. If an element equal to top is
   * encountered, true is returned and to is undefined. If top is
   * never encountered, false is returned. If from contains an
   * NaryExpr with tag naryTagMerge, the components of that NaryExpr
   * are treated in a similar manner.
   */
  private static boolean selectiveAdd(/*@non_null*/ ExprVec to, 
				      /*@non_null*/ ExprVec from,
				      /*@non_null*/ Expr bot, /*@non_null*/ Expr top, int naryTagMerge)
    {
      for(int i=0; i<from.size(); i++) {
	Expr e = from.elementAt(i);
	if( e == bot ) {
	  // drop e
	} else if( e == top ) {
	  return true;
	} else if( e.getTag() == naryTagMerge ) {
	  ExprVec exprs = ((NaryExpr)e).exprs;
	  if( selectiveAdd( to, exprs, bot, top, naryTagMerge ) )
	    return true;
	} else {
	  // nothing special
	  to.addElement(e);
	}
      }
      return false;
    }

}
