/* Copyright Hewlett-Packard, 2002 */


package escjava.translate;

import java.util.Hashtable;
import java.util.Vector;
import java.util.Enumeration;
import javafe.ast.*;
import javafe.tc.*;
import javafe.util.Location;
import javafe.util.Assert;
import javafe.util.Info;
import javafe.util.StackVector;
import javafe.util.Set;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.tc.Types;
import escjava.Main;

public final class GC {

  //// Makers for guarded commands

  public static GuardedCmd block(GenericVarDeclVec v, GuardedCmd g) {
    if (v.size() == 0)
      return g;
    else
      return VarInCmd.make(v, g);
  }

  public static GuardedCmd blockL(Stmt label, GuardedCmd g) {
    return trycmd(g, ifcmd( nary(TagConstants.ANYEQ,
				 ecvar,
				 symlit(label, "L_")),
			    skip(),
			    raise()));
  }

  /** Requires <code>0 < cmds.size()</code>. */
  public static GuardedCmd seq(GuardedCmd g1,GuardedCmd g2)
    { GuardedCmd[] cvec= {g1,g2};
      return seq(GuardedCmdVec.make(cvec)); }

  public static GuardedCmd seq(GuardedCmd g1,GuardedCmd g2, GuardedCmd g3 )
    { GuardedCmd[] cvec= {g1,g2,g3};
      return seq(GuardedCmdVec.make(cvec)); }

  public static GuardedCmd seq(GuardedCmd g1,GuardedCmd g2, GuardedCmd g3, GuardedCmd g4 )
    { GuardedCmd[] cvec= {g1,g2,g3,g4};
      return seq(GuardedCmdVec.make(cvec)); }

  /** May mutilate contents of <code>cmds</code>.  The caller is expected
    * not to use <code>cmds</code> after this call.
    **/

  public static GuardedCmd seq(/*@ non_null */ GuardedCmdVec cmds) {
    int n;
    if (!Main.peepOptGC) {
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
		   !Main.noPeepOptGCAssertFalse) &&
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

  public static GuardedCmd gets(VariableAccess lhs, Expr rhs) {
    return GetsCmd.make(lhs, rhs);
  }

  public static GuardedCmd subgets(VariableAccess lhs, Expr index, Expr rhs) {
    return SubGetsCmd.make(lhs, rhs, index);
  }

  public static GuardedCmd subsubgets(VariableAccess lhs,
				      Expr array, Expr index,
				      Expr rhs) {
    return SubSubGetsCmd.make(lhs, rhs, array, index);
  }

  public static GuardedCmd restoreFrom(VariableAccess lhs, Expr rhs) {
    return RestoreFromCmd.make(lhs, rhs);
  }

  public static GuardedCmd raise() {
    return SimpleCmd.make(TagConstants.RAISECMD, Location.NULL);
  }

  public static GuardedCmd skip() {
    return SimpleCmd.make(TagConstants.SKIPCMD, Location.NULL);
  }

  public static YieldCmd yield(int loc, Vector localVarDecls) {
      Set readVars = new Set();
      Set writeVars = new Set();      
      ConditionVec conds = ConditionVec.make();
      ConditionVec rmConds = ConditionVec.make();
      ConditionVec lmConds = ConditionVec.make();
      return YieldCmd.make(loc, null, readVars, writeVars, conds, rmConds, lmConds, localVarDecls);
  }

  public static YieldCmd yield(int loc, GuardedCmd gc, Vector localVarDecls) {
      Set readVars = new Set();
      Set writeVars = new Set();
      ConditionVec conds = ConditionVec.make();
      ConditionVec rmConds = ConditionVec.make();
      ConditionVec lmConds = ConditionVec.make();
      return YieldCmd.make(loc, gc, readVars, writeVars, conds, rmConds, lmConds, localVarDecls);
  }


  public static YieldCmd yield(int loc,
			       GuardedCmd code,
			       Set readVars, 
			       Set writeVars, Vector localVarDecls) {
      ConditionVec conds = ConditionVec.make();
      ConditionVec rmConds = ConditionVec.make();
      ConditionVec lmConds = ConditionVec.make();
      return YieldCmd.make(loc, code, readVars, writeVars, conds, rmConds, lmConds, localVarDecls);
  }

  public static LoopCmd loop(int sLoop, int eLoop, int locHotspot, Hashtable oldmap,
			     ConditionVec J, LocalVarDeclVec skolemConsts,
			     ExprVec P, GenericVarDeclVec tempVars,
			     GuardedCmd B, GuardedCmd S) {
    return LoopCmd.make(sLoop, eLoop, locHotspot, oldmap, J, skolemConsts, P, tempVars, B, S);
  }

  public static GuardedCmd ifcmd(Expr t, GuardedCmd c, GuardedCmd a) {
    GuardedCmd thn = GC.seq(GC.assume(t), c);
    GuardedCmd els = GC.seq(GC.assumeNegation(t), a);
    return choosecmd(thn, els);
  }

  /** Pops two command sequences off <code>s</code>, call them <code>a</code>
      and <code>b</code>.  Then returns the box composition of these
      commands, that is, <code>a [] b</code>. */

  public static GuardedCmd boxPopFromStackVector(StackVector s) {
    GuardedCmdVec b = GuardedCmdVec.popFromStackVector(s);
    GuardedCmdVec a = GuardedCmdVec.popFromStackVector(s);
    return choosecmd(GC.seq(a), GC.seq(b));
  }

  public static GuardedCmd choosecmd(GuardedCmd a, GuardedCmd b) {
    if (Main.peepOptGC) {
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

  public static GuardedCmd trycmd(GuardedCmd c, GuardedCmd a) {
    if (Main.peepOptGC) {
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
  public static boolean isBooleanLiteral(/*@ non_null */ Expr e, boolean b) {
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

  public static boolean isFalse(Expr e) {
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

  public static GuardedCmd check(int locUse,
				 int errorName, Expr pred,
				 int locPragmaDecl) {
    return check(locUse, errorName, pred, locPragmaDecl, null);
  }

  public static GuardedCmd check(int locUse,
				 int errorName, Expr pred,
				 int locPragmaDecl,
				 Object aux) {
    Assert.notFalse(locUse != Location.NULL);
    if (Main.guardedVC && locPragmaDecl != Location.NULL) {
      pred = GuardExpr.make(pred, locPragmaDecl);
    }
    switch( NoWarn.getChkStatus( errorName, locUse, locPragmaDecl) ) {
    case TagConstants.CHK_AS_ASSUME:
      LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
      return assume(pred);
    case TagConstants.CHK_AS_ASSERT:
      LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
      return assert(locUse, errorName, pred, locPragmaDecl, aux);
    case TagConstants.CHK_AS_SKIP:
      return skip();
    default:
      Assert.fail("unreachable");
      return null; // dummy return
    }

  }


  /** See description of <code>check</code> above. */

  public static GuardedCmd check(int locUse, Condition cond) {
    Assert.notFalse(locUse != Location.NULL);
    return check(locUse, cond.label, cond.pred, cond.locPragmaDecl, null);
  }

  /** See description of <code>check</code> above. */

  public static GuardedCmd check(int locUse, Condition cond, Object aux) {
    Assert.notFalse(locUse != Location.NULL);
    return check(locUse, cond.label, cond.pred, cond.locPragmaDecl, aux);
  }

  //@ requires label != TagConstants.CHKFREE
  public static Condition condition(int label, Expr pred,
				    int locPragmaDecl) {
    Assert.notFalse(label != TagConstants.CHKFREE);
    return Condition.make(label, pred, locPragmaDecl);
  }

  public static Condition freeCondition(Expr pred, int locPragmaDecl) {
    return Condition.make(TagConstants.CHKFREE, pred, locPragmaDecl);
  }

  //@ requires locPragmaDecl != Location.NULL;
  public static GuardedCmd assumeAnnotation(int locPragmaDecl,
					    /*@ non_null */ Expr p) {
    if (Main.guardedVC && locPragmaDecl != Location.NULL) {
      p = GuardExpr.make(p, locPragmaDecl);
    }
    LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
    return assume(p);
  }

  public static GuardedCmd assume(Expr p) {
    if (Main.peepOptGC && isBooleanLiteral(p, true)) {
      return skip();
    }
    return ExprCmd.make(TagConstants.ASSUMECMD, p, Location.NULL);
  }

  public static GuardedCmd assumeNegation(Expr p) {
    Expr non_p = not(p.getStartLoc(), p.getEndLoc(), p);
    return assume(non_p);
  }

  public static GuardedCmd fail() {
    return assume(falselit);
  }

  private static int assertContinueCounter = 0;

  //@ requires locUse != Location.NULL;
  private static GuardedCmd assert(int locUse,
				   int errorName, Expr pred,
				   int locPragmaDecl,
				   Object aux) {
    if (Main.assertContinue) {
      Identifier idn = Identifier.intern("assertContinue" +
					 assertContinueCounter);
      assertContinueCounter++;
      VariableAccess acVar = makeVar(idn, locUse);
      acVar.loc = Location.NULL;
      pred = or(pred, acVar);
    }
    String name = TagConstants.toString(errorName);
    if (aux != null && Main.suggest) {
      int n = AuxInfo.put(aux);
      name += "/" + n;
    }
    if (locPragmaDecl != Location.NULL) {
      name += ":" + UniqName.locToSuffix(locPragmaDecl);
    }
    name += "@" + UniqName.locToSuffix(locUse);
    Identifier en = Identifier.intern(name);
    Expr p = LabelExpr.make(locUse, locUse, false, en, pred);
    return ExprCmd.make(TagConstants.ASSERTCMD, p, locUse);
  }

  //@ requires subst != null && target != null ;
  //@ requires subst.keyType == \type(GenericVarDecl) ;
  //@ requires subst.elementType <: \type(Expr) ;
  public static Expr subst(int sloc, int eloc, Hashtable subst, Expr target)
    {
      if ( !Main.lazySubst ) {
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

  //@ requires target != null ;
  //@ requires var!=null && val!=null
  public static Expr subst(int sloc, int eloc,
			   GenericVarDecl var, Expr val, Expr target) {
    if ( !Main.lazySubst ) {
      // perform substitution eagerly
      Hashtable t = new Hashtable();
      t.put( var, val );
      return subst( sloc, eloc, t, target );
    } else {
      return SubstExpr.make( sloc, eloc, var, val, target );
    }
  }

  //@ requires var!=null && val!=null
  public static Expr subst(GenericVarDecl var, Expr val, Expr target) {
    return subst( Location.NULL, Location.NULL, var, val, target );
  }


  //// Makers for literals

  public static final Expr nulllit =
    LiteralExpr.make(TagConstants.NULLLIT, null, Location.NULL);

  public static final Expr zerolit =
    LiteralExpr.make(TagConstants.INTLIT, new Integer(0), Location.NULL);

  public static final Expr onelit =
    LiteralExpr.make(TagConstants.INTLIT, new Integer(1), Location.NULL);

  public static final Expr truelit =
    LiteralExpr.make(TagConstants.BOOLEANLIT,new Boolean(true),Location.NULL);

  public static final Expr falselit =
    LiteralExpr.make(TagConstants.BOOLEANLIT,new Boolean(false),Location.NULL);

  public static final Expr dzerolit =
    LiteralExpr.make(TagConstants.DOUBLELIT, new Double(0.0), Location.NULL);

  public static Expr intlit(int n) {
      return LiteralExpr.make(TagConstants.INTLIT, new Integer(n), Location.NULL);
  }

  public static Expr symlit(String s) {
    return LiteralExpr.make(TagConstants.SYMBOLLIT, s, Location.NULL);
  }

  public static Expr symlit(Stmt s, String prefix) {
    return symlit(prefix + UniqName.locToSuffix(s.getStartLoc()));
  }
  public static Expr zeroequiv(Type t) {
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
    /*@ unreachable */
    Assert.fail("Bad tag");
    return null;
  }


  //// Makers for variables

  public static VariableAccess makeVar(Identifier name, int locId) {
    LocalVarDecl v
      = LocalVarDecl.make(0, null, name, Types.anyType, locId,
			  null, Location.NULL);
    return VariableAccess.make(name, Location.NULL, v);
  }

  public static VariableAccess makeVar(String name, int locId) {
    return makeVar(Identifier.intern(name), locId);
  }

  public static VariableAccess makeFormalPara(String name, Type type,
					      int locId) {
    Identifier nameId = Identifier.intern(name);
    FormalParaDecl v
      = FormalParaDecl.make(0, null, nameId, type, locId);
    return VariableAccess.make( Identifier.intern(name), Location.NULL, v);
  }


  public static VariableAccess makeVar(String name) {
    return makeVar(name, Location.NULL);
  }

  public static VariableAccess makeVar(Identifier name) {
    return makeVar(name, Location.NULL);
  }

  public static VariableAccess makeFormalPara(String name, Type type) {
    return makeFormalPara(name, type, Location.NULL);
  }

  public static VariableAccess makeFormalPara(String name) {
    return makeFormalPara(name, Types.anyType);
  }



  public static final VariableAccess thisThread = makeVar("tid",
						  UniqName.specialVariable);

  public static final VariableAccess mainThread = makeVar("mainThread",
						  UniqName.specialVariable);

  public static final VariableAccess witness = makeVar("witness",
						       UniqName.specialVariable);

  public static final VariableAccess notThisThread = makeVar("notTid", 
							     UniqName.specialVariable);

  public static final VariableAccess allocvar = makeVar("alloc",
						  UniqName.specialVariable);
  public static final VariableAccess ecvar = makeVar("EC",
					       UniqName.specialVariable);
  public static final VariableAccess elemsvar = makeVar("elems",
						  UniqName.specialVariable);
  public static final VariableAccess resultvar = makeVar("RES",
    					   UniqName.specialVariable);
  public static final VariableAccess xresultvar = makeVar("XRES",
						    UniqName.specialVariable);
  public static final VariableAccess objectTBCvar = makeVar("objectToBeConstructed",
							    UniqName.specialVariable);

  // LSvar is not final because it is temporarily updated at
  // synchronized expressions. See trExpr
     public static VariableAccess LSvar = makeVar("LS",
    			       UniqName.specialVariable);


  /*
   * The type of thisvar (thisvar.decl.type) is set by
   * Translate.trBody.  It is also temporarily changed by
   * GetSpec.trMethodDecl.
   */
  //@ invariant thisvar.decl.type instanceof TypeSig
  public static final VariableAccess thisvar =
      makeFormalPara("this", javafe.tc.Types.javaLangObject(),
		     UniqName.specialVariable);


  /*
   * These handle case 5 of ESCJ 23b:
   */
  public static final Expr ec_throw = symlit("ecThrow");
  public static final Expr ec_return = symlit("ecReturn");

  //// Makers for expressions

  public static Expr typeExpr(Type t)
    { return TypeExpr.make(Location.NULL, Location.NULL, t); }

  // Various forms of nary()

  public static Expr nary(int tag, /*@ non_null */ Expr e) {
    return nary(Location.NULL, Location.NULL, tag, e);
  }

  public static Expr nary(int sloc, int eloc, int tag,
			  /*@ non_null */ Expr e) {
    Expr[] args = { e };
    return nary( sloc, eloc, tag, ExprVec.make(args));
  }

  public static Expr nary(int tag,
			  /*@ non_null */ Expr e1, /*@ non_null */ Expr e2) {
    return nary(Location.NULL, Location.NULL, tag, e1, e2);
  }

  public static Expr nary(int sloc, int eloc, int tag,
			  /*@ non_null */ Expr e1, /*@ non_null */ Expr e2) {
    Expr[] args = { e1, e2 };
    return nary(sloc,eloc,tag, ExprVec.make(args));
  }

  public static Expr nary(int tag, /*@ non_null */ Expr e1,
			  /*@ non_null */ Expr e2, /*@ non_null */ Expr e3) {
    return nary(Location.NULL, Location.NULL, tag, e1, e2, e3);
  }

  public static Expr nary(int sloc, int eloc, int tag,
			  /*@ non_null */ Expr e1, /*@ non_null */ Expr e2,
                          /*@ non_null */ Expr e3) {
    Expr[] args = { e1, e2, e3 };
    return nary(sloc,eloc,tag, ExprVec.make(args));
  }

  public static Expr nary(int sloc, int eloc, int tag, ExprVec ev) {

    if( Main.peepOptE ) {
      // Do some optimizations ...

      switch( tag ) {
	case TagConstants.BOOLAND:
	  {
	    ExprVec w = ExprVec.make( ev.size() );
	    if( selectiveAdd( w, ev, truelit, falselit,
			      TagConstants.BOOLAND ) )
	      {
		return falselit;
	      }
	
	    if( w.size() == 0 )
	      return truelit;
	    else if( w.size() == 1 )
	      return w.elementAt(0);
	    else
	      return NaryExpr.make( sloc, eloc, TagConstants.BOOLAND, w);
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
	      return NaryExpr.make( sloc, eloc, TagConstants.BOOLOR, w);
	  }

	case TagConstants.BOOLIMPLIES:
	  {
	    Expr c0 = ev.elementAt(0);
	    Expr c1 = ev.elementAt(1);

	    if( Main.bubbleNotDown ) {
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
	    } else if( Main.bubbleNotDown ) {
	      switch( c0.getTag() ) {
	      case TagConstants.BOOLOR:
	      case TagConstants.BOOLAND:
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
    return NaryExpr.make(sloc,eloc,tag, ev);
  }


  //// Makers for other GCExpr nodes

  public static Expr select(Expr c0, Expr c1) {
    return nary(TagConstants.SELECT, c0, c1);
  }

  public static Expr store(Expr c0, Expr c1, Expr c2) {
    return nary(TagConstants.STORE, c0, c1, c2);
  }

  public static Expr not(Expr c) {
    return not(Location.NULL, Location.NULL, c);
  }

  public static Expr not(int sloc, int eloc, Expr c) {
    return nary(sloc, eloc, TagConstants.BOOLNOT, c);
  }

  public static Expr and(Expr c1, Expr c2) {
    return and(Location.NULL, Location.NULL, c1, c2);
  }

  public static Expr and(int sloc, int eloc, Expr c1, Expr c2) {
    Expr[] es = {c1, c2};
    return and( sloc, eloc, ExprVec.make(es) );
  }

  public static Expr and(ExprVec v) {
    return and(Location.NULL, Location.NULL, v);
  }

  public static Expr and(int sloc, int eloc, ExprVec v) {
    return nary( sloc, eloc, TagConstants.BOOLAND, v );
  }

  public static Expr or(Expr c1, Expr c2) {
    return or(Location.NULL, Location.NULL, c1, c2);
  }

  public static Expr or(int sloc, int eloc, Expr c1, Expr c2) {
    Expr[] es = {c1, c2};
    return or( sloc, eloc, ExprVec.make(es) );
  }

  public static Expr or(ExprVec v) {
    return or(Location.NULL, Location.NULL, v);
  }

  public static Expr or(int sloc, int eloc, ExprVec v) {
    return nary( sloc, eloc, TagConstants.BOOLOR, v );
  }

  public static Expr implies(Expr c0, Expr c1) {
    return implies( Location.NULL, Location.NULL, c0, c1 );
  }

  public static Expr implies(int sloc, int eloc, Expr c0, Expr c1) {
    return nary( sloc, eloc, TagConstants.BOOLIMPLIES, c0, c1);
  }

  public static Expr forall(GenericVarDecl v, Expr e) {
    return forall( Location.NULL, Location.NULL, v, e, null );
  }

  public static Expr forall(GenericVarDecl v, Expr e, ExprVec nopats) {
    return forall( Location.NULL, Location.NULL, v, e, nopats );
  }

  public static Expr forall(int sloc, int eloc, GenericVarDecl v, Expr e) {
    return forall(sloc, eloc, v, e, null);
  }

  public static Expr forall(int sloc, int eloc, GenericVarDecl v, Expr e,
			    ExprVec nopats) {
    Assert.notNull(v);
    Assert.notNull(e);

    if( Main.peepOptE ) {

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
    return quantifiedExpr(sloc, eloc, TagConstants.FORALL, v, e, nopats);
  }

  public static Expr quantifiedExpr(int sloc, int eloc, int tag,
				    GenericVarDecl v, Expr e,
				    ExprVec nopats)
    {
      GenericVarDeclVec vs = GenericVarDeclVec.make();
      vs.addElement(v);
      return quantifiedExpr(sloc, eloc, tag, vs, e, nopats );
    }

  public static Expr quantifiedExpr(int sloc, int eloc, int tag,
				    GenericVarDeclVec vs, Expr e,
				    ExprVec nopats)
    {
      Assert.notFalse( tag == TagConstants.FORALL
		       || tag == TagConstants.EXISTS );

      if( tag == TagConstants.FORALL && vs.size() == 0 ) {
	  // empty forall
	  return e;
      }

      if( Main.peepOptE ) {

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
	  return QuantifiedExpr.make( sloc, eloc, tag, copy, qe.expr,
				      nopats );
	}
      }

      // No optimization done
      return QuantifiedExpr.make( sloc, eloc, tag, vs, e, nopats );
    }

  public static boolean isSimple(Expr e) {
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

  /** Adds elements to <code>to</code> from <code>from</code>.
      Elements equal to bot are dropped. If an element equal to top
      is encountered, true is returned and to is undefined. If top is
      never encountered, false is returned. If from contains an
      NaryExpr with tag naryTagMerge, the components of that NaryExpr
      are treated in a similar manner. */

  private static boolean selectiveAdd( ExprVec to, ExprVec from,
				       Expr bot, Expr top, int naryTagMerge )
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
