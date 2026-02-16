/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;

import escjava.Main;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import escjava.backpred.FindContributors;

import javafe.tc.TypeSig;
import escjava.tc.Types;
import escjava.tc.TypeCheck;

import javafe.util.*;



/** Responsible for getting Spec for calls.
    Includes ... from ESCJ16c.
    */

public final class GetSpec {

  public static Spec getSpecForCall(/*@ non_null */ RoutineDecl rd,
				    /*@ non_null */ FindContributors scope,
				    Set predictedSynTargs) {
    Spec spec = getCommonSpec( rd, scope, null );
    return extendSpecForCall( spec, scope, predictedSynTargs);
  }


  /* used for calls that are inlined */
  public static Spec getSpecForInline(/*@ non_null */ RoutineDecl rd,
				      /*@ non_null */ FindContributors scope) {
    return getCommonSpec( rd, scope, null );
    // TBW:  Should also add NonNullInit checks somewhere!
  }


  public static Spec getSpecForBody(/*@ non_null */ RoutineDecl rd,
				    /*@ non_null */ FindContributors scope,
				    /*@ non_null */ Set synTargs,
				    Hashtable premap) {
      Spec spec = getCommonSpec( rd, scope, premap );
      return extendSpecForBody( spec, scope, synTargs );
    }

    /** The methods getCommonSpecMT and the second
     *  version of getSpecForBody below are used in
     *  the multithreaded context - Sanjit
     **/
    public static Spec getCommonSpecMT(/*@ non_null */ RoutineDecl rd,
				    /*@ non_null */ FindContributors scope,
				     Hashtable premap) {
      return getCommonSpec( rd, scope, premap );
    }

    public static Spec getSpecForBody(/*@ non_null */ Spec spec,
				      /*@ non_null */ FindContributors scope,
				      /*@ non_null */ Set synTargs) {
	return extendSpecForBody( spec, scope, synTargs );
    }


    //@ ensures RES != null;
    private static Spec getCommonSpec(/*@ non_null */ RoutineDecl rd,
				      /*@ non_null */ FindContributors scope,
				      Hashtable premap)
    {
	/* Need to typecheck TypeDecl containing callee so that
	   requires/ensures/modifies clauses etc are resolved. */

	TypeSig T = TypeSig.getSig( rd.parent );
	T.typecheck();
	
	DerivedMethodDecl combined = getCombinedMethodDecl(rd);
	DerivedMethodDecl filtered = filterMethodDecl(combined, scope);

	/*
	 * If we are translating the spec for an inner-class
	 * constructor, then we need to substitute this$0arg for
	 * this.this$0 everywhere:
	 */
	Spec spec = null;
	try {
	    if (filtered.isConstructor() && !T.isTopLevelType()) {
		Inner.firstThis0 = Inner.getEnclosingInstanceArg(
					   (ConstructorDecl)filtered.original);
	    }

	    spec = trMethodDecl(filtered, premap);
	} finally {
	    Inner.firstThis0 = null;
	}

	return spec;
    }


    /**
     ** Implement GetCombinedMethodDecl from ESCJ 16c section 7:<p>
     **
     ** Precondition: the type declaring rd has been typechecked.<p>
     **
     ** Note: this routine may typecheck the supertypes of the type that
     ** declares rd.
     **/
    //@ ensures RES != null;
    public static DerivedMethodDecl getCombinedMethodDecl(/*@ non_null */ RoutineDecl rd) {
	DerivedMethodDecl dmd = new DerivedMethodDecl(rd);

	dmd.throwsSet = rd.raises;
	dmd.requires  = ExprModifierPragmaVec.make();
	dmd.modifies  = ExprModifierPragmaVec.make();
	dmd.ensures   = ExprModifierPragmaVec.make();
	dmd.exsures   = VarExprModifierPragmaVec.make();
	dmd.performs  = null;

	if (rd instanceof ConstructorDecl) {
	    // Handle constructor case:
	    dmd.args = rd.args;
	    TypeSig thisType = TypeSig.getSig(rd.parent);
	    if (!thisType.isTopLevelType()) {
		// An Inner class; add this$0 argument:
		dmd.args = rd.args.copy();
		FormalParaDecl this0arg =
		    Inner.getEnclosingInstanceArg((ConstructorDecl)rd);
		dmd.args.insertElementAt(this0arg, 0);
	    }

	    dmd.returnType = thisType;
	    addModifiersToDMD(dmd, rd);

	} else {
	  // Handle method case:
	  //@ assume rd instanceof MethodDecl;

	  MethodDecl md = (MethodDecl)rd;
	  dmd.returnType = md.returnType;

	  if (Modifiers.isStatic(rd.modifiers)) {
	    // static method
	    dmd.args = rd.args;
	  } else {
	    // instance method
	    dmd.args = md.args.copy();
	    dmd.args.insertElementAt((FormalParaDecl)GC.thisvar.decl, 0);
	  }
    
	  /*
	   * Add modifiers from this method as well as all methods it
	   * overrides; also handle non_null:
	   */
	  addModifiersToDMD(dmd, md);
	  Set overrides = escjava.tc.FlowInsensitiveChecks.getAllOverrides(md);
	  Enumeration enum = overrides.elements();
	  while (enum.hasMoreElements()) {
	    MethodDecl smd = (MethodDecl)enum.nextElement();
	    TypeSig.getSig(smd.parent).typecheck();

	    addModifiersToDMD(dmd, smd);
	  }
	}

	dmd.computeFreshUsage();

	return dmd;
    }

    /**
     ** Add the modifiers of rd to dmd, making any necessary substitions
     ** of parameter names.  Also propagates non_null's.<p>
     **
     ** Precondition: rd has been typechecked.<p>
     **/
    private static void addModifiersToDMD(/*@ non_null */ DerivedMethodDecl dmd,
					  /*@ non_null */ RoutineDecl rd) {
	/*
	 * Compute the substitution on parameter names to change a spec
	 * for an overridden method to refer to our method's parameter
	 * names instead of its; propagate non_nulls:
	 */
	Hashtable subst = new Hashtable();
	if (rd != dmd.original) {
	     for (int i=0; i<rd.args.size(); i++) {
		GenericVarDecl newDecl = dmd.original.args.elementAt(i);
		GenericVarDecl oldDecl = rd.args.elementAt(i);

		SimpleModifierPragma nonnull = NonNullPragma(oldDecl);
		if (nonnull!=null)
		    setNonNullPragma(newDecl, nonnull);

		VariableAccess va = VariableAccess.make(newDecl.id,
							Location.NULL,
							newDecl);

		subst.put(oldDecl, va);
	     }
	}

	if (rd.pmodifiers == null)
	    return;

	for (int i = 0; i < rd.pmodifiers.size(); i++) {
	    ModifierPragma mp = rd.pmodifiers.elementAt(i);
	    switch (mp.getTag()) {
	    case TagConstants.PERFORMS: 
		{
		    PerformsModifierPragma amp = (PerformsModifierPragma) mp;
		    if (dmd.performs != null) {
			ErrorSet.error(mp.getStartLoc(),
				       "Can only have one performs annotation per method.");
		    }
		    amp.stmt = doSubst(subst, amp.stmt);
		    dmd.performs = amp;
		    break;
		}
	      case TagConstants.REQUIRES:
	      case TagConstants.ALSO_REQUIRES:
	        {
		   ExprModifierPragma emp = (ExprModifierPragma)mp;
		   emp = doSubst(subst, emp);
		   dmd.requires.addElement(emp);
		   break;
		}
	      case TagConstants.MODIFIES:
	      case TagConstants.ALSO_MODIFIES:
		{
		  ExprModifierPragma emp = (ExprModifierPragma)mp;
		  emp = doSubst(subst, emp);
		  dmd.modifies.addElement(emp);
		  break;
		}
	      case TagConstants.ENSURES:
	      case TagConstants.ALSO_ENSURES:
		{
		  ExprModifierPragma emp = (ExprModifierPragma)mp;
		  emp = doSubst(subst, emp);
		  dmd.ensures.addElement(emp);
		  break;
		}
	      case TagConstants.EXSURES:
	      case TagConstants.ALSO_EXSURES:
		{
		  VarExprModifierPragma vemp = (VarExprModifierPragma)mp;
		  vemp = doSubst(subst, vemp);
		  dmd.exsures.addElement(vemp);
		  break;
		}
	      default:
		break;
	    }
	}

	/** Now check that every expr that is indicated
	    to be modifiable in the performs pragma is
	    also an entry in the modifies pragma **/
	// Sanjit
	//checkPerformsModifiesPragmas(dmd.performs, dmd.modifies);

	return;
    }

    /** Methods for checking equality and containment
	of Exprs that occur in the modifies clause.
	For all other kinds of exprs, it returns the
	result of object equality.
    **/    
    private static boolean areEqualExprs(Expr e1, Expr e2) {
	int e1tag = e1.getTag();
	int e2tag = e2.getTag();
	if (e1tag == e2tag) {
	    int tag = e1tag; 
	    switch(tag) {
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
		return (areEqualExprs(ar1.array, ar2.array)
			&& areEqualExprs(ar1.index, ar2.index));
	    }
	    case TagConstants.WILDREFEXPR: {
		WildRefExpr wr1 = (WildRefExpr) e1;
		WildRefExpr wr2 = (WildRefExpr) e2;
		return areEqualExprs(wr1.expr, wr2.expr);
	    }
	    default:
		//@ unreachable
		Assert.fail("Fall thru; tag = " + TagConstants.toString(tag));
		return false;
	    }
	}
	else 
	    return false;
    }

    private static boolean containsExpr(ExprVec ev, Expr e) {
	for(int i = 0; i < ev.size(); i++) {
	    if (areEqualExprs(ev.elementAt(i), e))
		return true;
	}
	return false;
    }

	/*
    private static void checkPerformsModifiesPragmas(PerformsModifierPragma performs, 
						     ExprModifierPragmaVec modifies) {

	int s1 = performs.size(); 
	int s2 = modifies.size(); 
	if (s2 == 0 && s1 > 0) { 
	    int loc = performs.elementAt(0).loc;
	    ErrorSet.error(loc, "Performs pragma cannot occur without matching modifies pragma");
	    return;
	}

	if (s1 == 0) return; // nothing to check

	// make the exprvec out of all the exprs in the modifies pragmas
	ExprVec mVec = ExprVec.make();
	for(int i=0; i < s2; i++) {
	    Expr e = modifies.elementAt(i).expr;
	    mVec.addElement(e);
	}

	// run through all the performs pragma lists and check them 
	for(int i=0; i < s1; i++) {
	    ActionModifierPragma amp = performs.elementAt(i);
	    ExprVec exprs = amp.exprs;
	    for(int j=0; j < exprs.size(); j++) {
		Expr e = exprs.elementAt(j);
		if (!ExprUtil.containsSpecExpr(mVec, e)) {
		    ErrorSet.error(amp.loc, "Set of expressions in performs pragma list must be a subset of those in the modifies pragmas");
		    return;
		}
	    }
	}

	return;
    }
	*/

    /** Perform a substitution on an ActionModifierPragma **/
    private static PerformsStmt doSubst(Hashtable subst, PerformsStmt p) {
	switch (p.getTag()) {
	  case TagConstants.CHOICE: {
	      PerformsChoice samp = (PerformsChoice)p;
	      return PerformsChoice.make(p.loc,
					 doSubst(subst, samp.left),
					 doSubst(subst, samp.right));

	  }
	  case TagConstants.SEMICOLON: {
	      PerformsSequence samp = (PerformsSequence)p;
	      PerformsStmtVec sv = PerformsStmtVec.make();
	      int i;
	      for (i = 0; i < samp.seq.size(); i++) {
		  sv.addElement(doSubst(subst, samp.seq.elementAt(i)));
	      }
	      return PerformsSequence.make(p.loc, sv);
	  }

	case TagConstants.ACTION: {
	      PerformsAction samp = (PerformsAction)p;
	      
	      Expr ampPred = Substitute.doSubst(subst, samp.pred);
	      ExprVec ampExprs = ExprVec.make();
	      for(int k=0; k < samp.exprs.size(); k++) {
		  ampExprs.addElement(Substitute.doSubst(subst, samp.exprs.elementAt(k)));
	      }
	      
	      Assert.notFalse(ampExprs.size() == samp.exprs.size());

	      return PerformsAction.make(p.loc,
					 ampExprs,
					 ampPred, 
					 samp.label);
	  }
	
	default:
	    System.out.println("UNKNOWN ACTION MODIFIER PRAGMA: " + p);
	    Assert.notFalse(false);
	    return null;
	}
	
    }


    /** Perform a substitution on an ExprModifierPragma **/
    private static ExprModifierPragma doSubst(Hashtable subst,
					      ExprModifierPragma emp) {
	return ExprModifierPragma.make(emp.tag,
				       Substitute.doSubst(subst, emp.expr),
				       emp.loc);
    }

    /** Perform a substitution on a VarExprModifierPragma **/
    private static VarExprModifierPragma doSubst(Hashtable subst,
					         VarExprModifierPragma vemp) {
	return VarExprModifierPragma.make(vemp.tag,
					  vemp.arg,
					  Substitute.doSubst(subst, vemp.expr),
					  vemp.loc);
    }


  //@ ensures RES != null;
  public static DerivedMethodDecl filterMethodDecl(/*@ non_null */ DerivedMethodDecl dmd,
						   /*@ non_null */ FindContributors scope) {
    if (!Main.filterMethodSpecs) {
      return dmd;
    }

    DerivedMethodDecl dmdFiltered = new DerivedMethodDecl(dmd.original);
    dmdFiltered.args = dmd.args;
    dmdFiltered.returnType = dmd.returnType;
    dmdFiltered.throwsSet = dmd.throwsSet;

    dmdFiltered.requires = dmd.requires;
    dmdFiltered.modifies = filterExprModPragmas(dmd.modifies, scope);
    dmdFiltered.ensures = filterExprModPragmas(dmd.ensures, scope);
    dmdFiltered.exsures = filterVarExprModPragmas(dmd.exsures, scope);

    dmdFiltered.computeFreshUsage();

    return dmdFiltered;
  }

  private static ExprModifierPragmaVec filterExprModPragmas(/*@ non_null */ ExprModifierPragmaVec vec,
							    /*@ non_null */ FindContributors scope) {
    int n = vec.size();
    ExprModifierPragmaVec vecNew = null;  // lazily allocated
    for (int i = 0; i < n; i++) {
      ExprModifierPragma emp = vec.elementAt(i);
      if (exprIsVisible(scope.originType, emp.expr)) {
	// keep this pragma
	if (vecNew != null) {
	  vecNew.addElement(emp);
	}
      } else {
	// filter out this pragma
	if (vecNew == null) {
	  vecNew = ExprModifierPragmaVec.make(n-1);
	  for (int j = 0; j < i; j++) {
	    vecNew.addElement(vec.elementAt(j));
	  }
	}
      }
    }
    if (vecNew == null) {
      return vec;
    } else {
      return vecNew;
    }
  }


  private static VarExprModifierPragmaVec filterVarExprModPragmas(/*@ non_null */ VarExprModifierPragmaVec vec,
								  /*@ non_null */ FindContributors scope) {
    int n = vec.size();
    VarExprModifierPragmaVec vecNew = null;  // lazily allocated
    for (int i = 0; i < n; i++) {
      VarExprModifierPragma vemp = vec.elementAt(i);
      if (exprIsVisible(scope.originType, vemp.expr)) {
	// keep this pragma
	if (vecNew != null) {
	  vecNew.addElement(vemp);
	}
      } else {
	// filter out this pragma
	if (vecNew == null) {
	  vecNew = VarExprModifierPragmaVec.make(n-1);
	  for (int j = 0; j < i; j++) {
	    vecNew.addElement(vec.elementAt(j));
	  }
	}
      }
    }
    if (vecNew == null) {
      return vec;
    } else {
      return vecNew;
    }
  }

  //-------------------------------------------------------------------------
  //-------------------------------------------------------------------------
  //-------------------------------------------------------------------------


  /** Translates a MethodDecl to a Spec. */

  //@ ensures RES != null;
  private static Spec trMethodDecl(/*@ non_null */ DerivedMethodDecl dmd,
				   Hashtable premap) {
      Assert.notNull(dmd);

      /*
       * Unlike any body we may be translating, for these
       * translations this's type is that of the type that contains
       * the method or constructor dmd.original.
       */
      TypeSig T = TypeSig.getSig(dmd.getContainingClass());
      Type savedType = GC.thisvar.decl.type;
      GC.thisvar.decl.type = T;
      // (we restore GC.thisvar.decl.type just before returning)


      ConditionVec pre = trMethodDeclPreconditions(dmd);
      
      ExprVec targets  = ExprVec.make();

      if( dmd.usesFresh )
	targets.addElement( GC.allocvar );

      // translates modifies list

      for (int i = 0; i < dmd.modifies.size(); i++) {
	Expr designator = dmd.modifies.elementAt(i).expr;
	Expr gcDesignator = TrAnExpr.trSpecExpr(designator);
	targets.addElement(gcDesignator);
      }

      // handle targets stuff, and create preVarMap

      Set roots = new Set(); // set of GenericVarDecls
      
      for(int i=0; i<targets.size(); i++)
	{
	  Expr gcDesignator = targets.elementAt(i);
	  VariableAccess shaved = shave( gcDesignator );
	  roots.add( shaved.decl );
	}

      Hashtable preVarMap;
      if (premap == null)
	preVarMap = makeSubst( roots.elements(), "pre" );
      else
	preVarMap = restrict( premap, roots.elements() );

      // Now create the postconditions

      ConditionVec post = trMethodDeclPostcondition(dmd, preVarMap);

      Spec spec = Spec.make(dmd, targets, preVarMap, pre, post);

      GC.thisvar.decl.type = savedType;
      return spec;
    }

  /** Computes the preconditions, according to section 7.2.0 of ESCJ 16. */
  
  private static ConditionVec trMethodDeclPreconditions(/*@ non_null */ DerivedMethodDecl dmd) {
    ConditionVec pre = ConditionVec.make();
      
    // Account for properties about parameters
    for (int i = 0; i < dmd.args.size(); i++) {
      FormalParaDecl arg = dmd.args.elementAt(i);
      
      if (i == 0 && arg == GC.thisvar.decl) {
	// the special parameter "this"
	addFreeTypeCorrectAs(arg, TypeSig.getSig(dmd.getContainingClass()),
			     pre);
	VariableAccess argAccess = TrAnExpr.makeVarAccess(arg, Location.NULL);
	Expr nonnull = GC.nary(TagConstants.REFNE, argAccess, GC.nulllit);
	Condition cond = GC.freeCondition(nonnull, Location.NULL);
	pre.addElement(cond);
	
      } else {
	// regular parameters
	addFreeTypeCorrectAs(arg, arg.type, pre);
	SimpleModifierPragma nonNullPragma = NonNullPragma(arg);
	if (nonNullPragma != null) {
	  VariableAccess argAccess = TrAnExpr.makeVarAccess(arg,
							    Location.NULL);
	  Expr nonnull = GC.nary(TagConstants.REFNE, argAccess, GC.nulllit);
	  Condition cond = GC.freeCondition(nonnull,
					    nonNullPragma.getStartLoc());
	  pre.addElement(cond);
	}
      }
    }

    // Add the declared preconditions
    for (int i = 0; i < dmd.requires.size(); i++) {
      ExprModifierPragma prag = dmd.requires.elementAt(i);
      Expr gcExpr = TrAnExpr.trSpecExpr(prag.expr);
      Condition cond = GC.condition(TagConstants.CHKPRECONDITION, gcExpr,
				    prag.getStartLoc());
      pre.addElement(cond);
    }

    return pre;
  }

  /** Computes the postconditions, according to section 7.2.2 of ESCJ 16. */
  
  private static ConditionVec trMethodDeclPostcondition(/*@ non_null */ DerivedMethodDecl dmd,
							/*@ non_null */ Hashtable wt) {
    ConditionVec post = ConditionVec.make();

    // type correctness of targets (including "alloc", if "alloc" is a target)
    Enumeration wtEnum = wt.keys();
    while (wtEnum.hasMoreElements()) {
      GenericVarDecl vd = (GenericVarDecl)wtEnum.nextElement();
      Expr e = TrAnExpr.targetTypeCorrect(vd, wt);
      Condition cond = GC.freeCondition(e, Location.NULL);
      post.addElement(cond);
    }

    if (dmd.isConstructor()) {
      // Free:  RES != null && !isAllocated(RES, wt[[alloc]])
      Expr nonnull = GC.nary(TagConstants.REFNE, GC.resultvar, GC.nulllit);
      Expr allocated = GC.nary(TagConstants.ISALLOCATED,
			       GC.resultvar,
			       (VariableAccess)wt.get(GC.allocvar.decl));
      Expr notAllocated = GC.not(allocated);
      Condition cond = GC.freeCondition(GC.and(nonnull, notAllocated),
					Location.NULL);
      post.addElement(cond);
    }

    if (!Types.isVoidType(dmd.returnType)) {
      // Free:  TypeCorrectAs[[ RES, T ]]
      addFreeTypeCorrectAs(GC.resultvar.decl, dmd.returnType, post);
    }

    TypeSig T = TypeSig.getSig(dmd.getContainingClass());
    if (dmd.isConstructor() && !T.isTopLevelType()) {
	// Free: RES.this$0 == this$0arg
	Expr this0 = TrAnExpr.makeVarAccess(Inner.getEnclosingInstanceField(T),
					    Location.NULL);
	FormalParaDecl this0arg =
	    Inner.getEnclosingInstanceArg((ConstructorDecl)dmd.original);
	Expr thisSet = GC.nary(TagConstants.REFEQ,
			       GC.select(this0, GC.resultvar),
			       TrAnExpr.makeVarAccess(this0arg,Location.NULL));
	Condition cond = GC.freeCondition(thisSet, Location.NULL);
	post.addElement(cond);
    }


    if (dmd.throwsSet.size() == 0) {
      // UnexpectedException:  EC == ecReturn
      Expr pred = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return);
      Condition cond = GC.condition(TagConstants.CHKUNEXPECTEDEXCEPTION, pred,
				    Location.NULL);
      post.addElement(cond);
    } else {
      // Free:  EC == ecThrow ==>
      //          XRES != null && typeof(XRES) <: Throwable &&
      //          isAllocated(XRES, alloc)
      Expr antecedent = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_throw);
      ExprVec ev = ExprVec.make();
      // XRES != null
      Expr p = GC.nary(TagConstants.REFNE, GC.xresultvar, GC.nulllit);
      ev.addElement(p);
      // typeof(XRES) <: Throwable
      p = GC.nary(TagConstants.TYPELE,
		  GC.nary(TagConstants.TYPEOF, GC.xresultvar),
		  GC.typeExpr(Types.javaLangThrowable()));
      ev.addElement(p);
      // isAllocated(XRES, alloc)
      p = GC.nary(TagConstants.ISALLOCATED,
		  GC.xresultvar,
		  GC.allocvar);
      ev.addElement(p);
      // conjoin and add free postcondition
      Expr consequence = GC.and(ev);
      Condition cond = GC.freeCondition(GC.implies(antecedent, consequence),
					Location.NULL);
      post.addElement(cond);

      // UnexpectedException:
      //   EC == ecReturn ||
      //   (EC == ecThrow &&
      //     (typeof(XRES) <: X1 && ... &&& typeof(XRES) <: Xx))
      Expr normal = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return);
      Expr except = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_throw);
      Expr typeofXRES = GC.nary(TagConstants.TYPEOF, GC.xresultvar);
      ev.removeAllElements();
      for (int i = 0; i < dmd.throwsSet.size(); i++) {
	TypeName x = dmd.throwsSet.elementAt(i);
	p = GC.nary(TagConstants.TYPELE, typeofXRES, GC.typeExpr(x));
	ev.addElement(p);
      }
      Expr eOutcomes;
      eOutcomes = GC.or(ev);

      p = GC.or(normal, GC.and(except, eOutcomes));

      Assert.notFalse(dmd.original.locThrowsKeyword != Location.NULL);
      cond = GC.condition(TagConstants.CHKUNEXPECTEDEXCEPTION, p,
			  Location.NULL);
      post.addElement(cond);
    }

    // constructors ensure that the new object's owner field is null
    if (dmd.isConstructor()) {
	// get java.lang.Object
	TypeSig obj = Types.javaLangObject();
	
	FieldDecl owner = null; // make the compiler happy
	boolean found = true;
	try {
	    owner = Types.lookupField(obj, Identifier.intern("owner"),
				      obj);
	}
	catch (javafe.tc.LookupException e) {
	    found = false;
	}
	// if we couldn't find the owner ghost field, there's nothing to do
	if (found) {
	    VariableAccess ownerVA = 
		TrAnExpr.makeVarAccess(owner, Location.NULL);
	    Expr everything;
	    Expr ownerNull =  GC.nary(TagConstants.REFEQ, 
				      GC.select(ownerVA, GC.resultvar), 
				      GC.nulllit);
	    // if there are no exceptions thrown, it is sufficient to do
	    // RES.owner == null
	    // SHAZ: Note that code for local and local_holder needs to be added here
	    if (dmd.throwsSet.size() == 0)
		everything = ownerNull;
	    // else we need to do (EC == ECReturn) ==> (RES.owner == null)
	    else {	    
		Expr ret = GC.nary(TagConstants.ANYEQ, GC.ecvar, 
				   GC.ec_return);
		everything = GC.implies(ret, ownerNull);
	    }
	    Condition cond = GC.condition(TagConstants.CHKOWNERNULL,
					  everything,
					  Location.NULL);
	    post.addElement(cond);
	}
    }
	    
    // Finally, add declared postconditions
    // First normal postconditions
    {
      // EC == ecReturn
      Expr ante = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return);

      Hashtable map;
      if (dmd.isConstructor()) {
	map = new Hashtable();
	map.put(GC.thisvar.decl, GC.resultvar);
      } else {
	map = null;
      }
      for (int i = 0; i < dmd.ensures.size(); i++) {
	ExprModifierPragma prag = dmd.ensures.elementAt(i);
	Expr pred = TrAnExpr.trSpecExpr(prag.expr, map, wt);
	pred = GC.implies(ante, pred);
	Condition cond = GC.condition(TagConstants.CHKPOSTCONDITION,
				      pred, prag.getStartLoc());
	post.addElement(cond);
      }
    }
    // Then exceptional postconditions
    {
      // EC == ecThrow
      Expr ante0 = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_throw);
      // typeof(XRES)
      Expr typeofXRES = GC.nary(TagConstants.TYPEOF, GC.xresultvar);

      for (int i = 0; i < dmd.exsures.size(); i++) {
	// Pragma has the form:  exsures (T v) E
	VarExprModifierPragma prag = dmd.exsures.elementAt(i);
	// TrSpecExpr[[ E, {v-->XRES}, wt ]]
	Hashtable map = new Hashtable();
	map.put(prag.arg, GC.xresultvar);
	Expr pred = TrAnExpr.trSpecExpr(prag.expr, map, wt);
	// typeof(XRES) <: T
	Expr ante1 = GC.nary(TagConstants.TYPELE,
			     typeofXRES, GC.typeExpr(prag.arg.type));
	//     EC == ecThrow && typeof(XRES) <: T
	// ==> TrSpecExpr[[ E, {v-->XRES}, wt ]]
	pred = GC.implies(GC.and(ante0, ante1), pred);
	Condition cond = GC.condition(TagConstants.CHKPOSTCONDITION,
				      pred, prag.getStartLoc());
	post.addElement(cond);
      }
    }

    return post;
  }

  //-------------------------------------------------------------------------
  //-------------------------------------------------------------------------
  //-------------------------------------------------------------------------

  /** Implements ExtendSpecForCall, section 7.3 of ESCJ 16. */
  
  private static Spec extendSpecForCall(/*@ non_null */ Spec spec,
					/*@ non_null */ FindContributors scope,
					Set predictedSynTargs) {

    // The set of variables modified by *this* GC call:
    Set modifiedVars = new Set( spec.preVarMap.keys() );
    
    ParamAndGlobalVarInfo vars = null;

    InvariantInfo tmpii = collectInvariants(scope);
    /*
    if (Main.multiThreaded) {
	tmpii = collectGlobalAndObjectInvariants(scope);
    }
    else {
	tmpii = collectInvariants(scope);
    }
    */

    for (InvariantInfo ii = mergeInvariants(tmpii);
	 ii != null; ii = ii.next) {

      /*
       * Does ii mention a variable that this GC call will modify?
       */
      Set invFV = Substitute.freeVars( ii.J );
      boolean mentionsModifiedVars = Main.useAllInvPostCall ||
	                             invFV.containsAny(modifiedVars);

      /*
       * Does ii mention a variable that the body that is making the
       * GC call ever modifies?
       */
      boolean falsifiable = true;
      if (predictedSynTargs!=null) {
	Assert.notFalse(!Main.useAllInvPreBody);
	falsifiable = invFV.containsAny(predictedSynTargs);
      }


      if (ii.isStatic) {
	// static invariant

	// PRECONDITION for static invariant
	Condition cond = GC.condition(TagConstants.CHKOBJECTINVARIANT, ii.J,
				      ii.prag.getStartLoc());
	if (falsifiable)
	  spec.pre.addElement(cond);

	// POSTCONDITION for static invariant

	if( mentionsModifiedVars ) {
	  // The free variables of "J" overlap with "synTargs", so add "J"
	  cond = GC.freeCondition(ii.J, ii.prag.getStartLoc());
	  spec.post.addElement(cond);
	}
      
      } else {
	// instance invariant
	Assert.notNull(ii.sdecl);
	Assert.notNull(ii.s);

	if (falsifiable) {
	  // PRECONDITION for instance invariant
	
	  // Gather parameters and static fields, unless already cached
	  if (vars == null) {
	    vars = collectParamsAndGlobals(spec, scope);
	  }

	  // Instead of generating one precondition:
	  //   p == null || TrSpecExpr[[ J, {this-->p}, {} ]]
	  // for each parameter or static field "p", generate one precondition:
	  //   (FORALL s :: (s == p0 || s == p1 || ...)
	  //                ==> s == null || TrSpecExpr[[ J, {this-->p}, {} ]] )
	  // where the "p0", "p1", ... are the parameters and static fields.
	  // If the list "p0", "p1", ... is empty, don't generate a precondition.
	  // If each "p0", "p1", ... is "non_null", then drop the
	  // disjunct "s == null" from the consequent.
	  boolean allAreNonnull = true;
	  ExprVec alternatives = null;  // alternatives == null ||
	                                // alternatives.size() != 0
	  for (ParamAndGlobalVarInfo info = vars;
	       info != null; info = info.next) {
	    if (Types.isSubclassOf(info.U, ii.U)) {
	      Expr eq = GC.nary(TagConstants.REFEQ,
				ii.s, TrAnExpr.makeVarAccess(info.vdecl,
							     Location.NULL));
	      if (alternatives == null)
		alternatives = ExprVec.make();
	      alternatives.addElement(eq);
	      if (! info.isNonnull)
		allAreNonnull = false;
	    }
	  }
	  // Now we have built the equalities of the antecedent
	  if (alternatives != null) {
	    Expr ante = GC.or(alternatives);
	    Expr cons = ii.J;
	    if (!allAreNonnull) {
	      cons = GC.or(GC.nary(TagConstants.REFEQ, ii.s, GC.nulllit), cons);
	    }
	    Expr quant = GC.forall(ii.sdecl, GC.implies(ante, cons));
	    Condition cond = GC.condition(TagConstants.CHKOBJECTINVARIANT,
					  quant, ii.prag.getStartLoc());

	    spec.pre.addElement(cond);
	  }
	}

	// POSTCONDITION for instance invariant
	
	if (mentionsModifiedVars) {
	  // TypeCorrectNoallocAs[[ s, U ]] && s != null
	  ExprVec ev = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
							     null, true,
							     null, false);
	  ExprVec nopats = ev.copy();

	  Expr p = TrAnExpr.trSpecExpr(ii.prag.expr,
				       TrAnExpr.union(spec.preVarMap, ii.map),
				       null);
	  ev.addElement(p);

	  Expr ante = GC.and(ev);
	  Expr impl = GC.implies(ante, ii.J);

	  spec.post.addElement(GC.freeCondition(GC.forall(ii.sdecl, impl,
							  nopats),
						ii.prag.getStartLoc()));
	}
      }
    }
    
    return spec;
  }

  /** Implements ExtendSpecForBody, section 7.4 of ESCJ 16. */
  
  private static Spec extendSpecForBody(/*@ non_null */ Spec spec,
					/*@ non_null */ FindContributors scope,
					/*@ non_null */ Set synTargs) {
    
    // Add NonNullInit checks
    if (spec.dmd.isConstructor() &&
	!spec.dmd.isConstructorThatCallsSibling()) {
      ClassDecl cd = (ClassDecl)spec.dmd.getContainingClass();
      // first check fields in first-inherited interfaces
      Enumeration enum = getFirstInheritedInterfaces(cd);
      while (enum.hasMoreElements()) {
	TypeDecl tdSuperInterface = (TypeDecl)enum.nextElement();
	nonNullInitChecks(tdSuperInterface, spec.post);
      }
      // then check fields in the current class
      nonNullInitChecks(cd, spec.post);
    }

    InvariantInfo tmpii = collectInvariants(scope);
    /*
    if (Main.multiThreaded) {
	tmpii = collectGlobalAndObjectInvariants(scope);
    }
    else {
	tmpii = collectInvariants(scope);
    }
    */

    for (InvariantInfo ii = mergeInvariants(tmpii);
	 ii != null; ii = ii.next) {
      addInvariantBody(ii, spec, synTargs);
    }

    ExprVec axioms = collectAxioms(scope);
    
    for(int i=0; i<axioms.size(); i++) {
      spec.pre.addElement( GC.freeCondition( axioms.elementAt(i),
					     Location.NULL ) );
    }

    return spec;
  }

  /** Add to <code>post</code> all NonNullInit checks for non_null
    * instance fields and instance ghost fields declared in <code>td</code>.
    **/
  private static void nonNullInitChecks(/*@ non_null */ TypeDecl td,
					/*@ non_null */ ConditionVec post) {
    TypeDeclElemVec tdes = td.elems;

    // check that non_null instance fields have been initialized
    for (int i = 0; i < tdes.size(); i++) {
      TypeDeclElem tde = tdes.elementAt(i);
      FieldDecl fd;
      if (tde.getTag() == TagConstants.FIELDDECL) {
	fd = (FieldDecl)tde;
      } else if (tde.getTag() == TagConstants.GHOSTDECLPRAGMA) {
	fd = ((GhostDeclPragma)tde).decl;
      } else {
	continue;
      }

      if (!Modifiers.isStatic(fd.modifiers)) {
	SimpleModifierPragma smp = NonNullPragma(fd);
	if (smp != null) {
	  // NonNullInit: EC==ecReturn ==> f[RES] != null

	  Expr left = GC.nary(TagConstants.ANYEQ,GC.ecvar, GC.ec_return);
		
	  VariableAccess f = TrAnExpr.makeVarAccess(fd, Location.NULL);
	  Expr right = GC.nary(TagConstants.REFNE,
			       GC.select(f, GC.resultvar),
			       GC.nulllit);
	  Expr pred = GC.implies(left, right);
	  Condition cond = GC.condition(TagConstants.CHKNONNULLINIT,
					pred, smp.getStartLoc());
	  post.addElement(cond);
	}
      }
    }
  }
  
  //@ ensures RES != null && RES.elementType == type(InterfaceDecl);
  static public Enumeration getFirstInheritedInterfaces(
					   /*@ non_null @*/ ClassDecl cd) {
    Set interfaces = new Set();
    addSuperInterfaces(cd, interfaces);
    if (interfaces.size() != 0 && cd.superClass != null) {
      TypeDecl tdSuper = TypeSig.getSig(cd.superClass).getTypeDecl();
      Set superClassInterfaces = new Set();
      addSuperInterfaces(tdSuper, superClassInterfaces);
      Enumeration enum = superClassInterfaces.elements();
      while (enum.hasMoreElements()) {
	interfaces.remove(enum.nextElement());
      }
    }
    return interfaces.elements();
  }

  //@ requires set.elementType == type(InterfaceDecl);
  private static void addSuperInterfaces(/*@ non_null */ TypeDecl td,
					 /*@ non_null */ Set set) {
    if (td instanceof InterfaceDecl) {
      set.add(td);
    }
    // add superinterfaces of "td"
    TypeNameVec si = td.superInterfaces;
    for (int i = 0; i < si.size(); i++) {
      TypeName superName = si.elementAt(i);
      TypeDecl superDecl = TypeSig.getSig(superName).getTypeDecl();
      addSuperInterfaces(superDecl, set);
    }
  }


  /** Extend <code>spec</code>, in a way appropriate for checking the
    * body of a method or constructor, to account for invariant <code>ii.J</code>
    * declared in class <code>ii.U</code>.
    * The body to be checked has syntactic targets <code>synTargs</code>.
    **/

  private static void addInvariantBody(/*@ non_null */ InvariantInfo ii,
					/*@ non_null */ Spec spec,
					/*@ non_null */ Set synTargs) {

    Set invFV = Substitute.freeVars( ii.J );

    /* Include invariant in post only if intersection of vars of
       invariant and vars modified in the method body is nonempty. */

    boolean includeInPost = Main.useAllInvPostBody ||
                            invFV.containsAny(synTargs);

    if (ii.isStatic) {
      // static invariant

      Condition cond = GC.freeCondition(ii.J, ii.prag.getStartLoc());

      spec.pre.addElement(cond);
      
      if (includeInPost) {
	cond = GC.condition(TagConstants.CHKOBJECTINVARIANT, ii.J,
			    ii.prag.getStartLoc());
	spec.post.addElement(cond);
      }
      
    } else {
      // instance invariant

      // Do the precondition
      {
	// Method, or constructor not declared below:
	//   (FORALL s :: TypeCorrectNoallocAs[[ s, U ]] && s != null ==> J)
	//
	// Constructor of a class T, where either
	//   *  U is a subtype of T, and
	//      either U is not T or the constructor does not call a sibling
	// or
	//   *  U is an interface, and
	//        +  m calls sibling constructor and U is not a
	//           superinterface of T
	//        or
	//        +  m does not call sibling constructor and U is not a
	//           superinterface of a proper superclass of T
	//   (FORALL s :: TypeCorrectNoallocAs[[ s, U ]] && s != null &&
	//                s != objectToBeConstructed
        //            ==> J)
	//
	// In either case, NOPATS is the same as the antecedent.
	ExprVec ante = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
							     null, true,
							     null, false);
	if (spec.dmd.isConstructor()) {
	  TypeSig tU = ii.U;
	  TypeSig tT = TypeSig.getSig(spec.dmd.getContainingClass());
	  boolean includeAntecedent = false;
	  if (Types.isSubclassOf(tU, tT)) {
	    if (!Types.isSameType(tU, tT) ||
		!spec.dmd.isConstructorThatCallsSibling()) {
	      includeAntecedent = true;
	    }
	  } else if (ii.prag.parent instanceof InterfaceDecl) {
	    if (spec.dmd.isConstructorThatCallsSibling()) {
	      if (!Types.isSubclassOf(tT, tU)) {
		includeAntecedent = true;
	      }
	    } else {
	      ClassDecl cd = (ClassDecl)spec.dmd.getContainingClass();
	      if (!Types.isSubclassOf(TypeSig.getSig(cd.superClass), tU)) {
		includeAntecedent = true;
	      }
	    }
	  }
	  if (includeAntecedent) {
	    Expr p = GC.nary(TagConstants.REFNE, ii.s, GC.objectTBCvar);
	    ante.addElement(p);
	  }
	}
	Expr body = GC.implies(GC.and(ante), ii.J);
	Expr quant = GC.forall(ii.sdecl, body, ante);
	Condition cond = GC.freeCondition(quant, ii.prag.getStartLoc());

	spec.pre.addElement(cond);
      }

      // Do the postcondition
      
      // Include the invariant as a checked postcondition if its free variables
      // overlap with the syntactic targets of the body (as indicated by the
      // current value of "includeInPost"), or if the routine is a constructor
      // (that does not call a sibling) of some class "T" and the invariant is
      // declared in "T" or in one of "T"'s first-inherited interfaces.
      if (!includeInPost && spec.dmd.isConstructor() &&
	  !spec.dmd.isConstructorThatCallsSibling()) {
	TypeSig tU = ii.U;
	ClassDecl cd = (ClassDecl)spec.dmd.getContainingClass();
	TypeSig tT = TypeSig.getSig(cd);
	if (Types.isSubclassOf(tT, tU) &&
	    (cd.superClass == null ||
	     !Types.isSubclassOf(TypeSig.getSig(cd.superClass), tU))) {
	  // "U" is "T" or a first-inherited interface of "T"
	  includeInPost = true;
	}
      }

      if (includeInPost) {
	// TypeCorrectAs[[ s, U ]] && s != null
	ExprVec ante = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
							     null, true,
							     null, true);
	
	if (spec.dmd.isConstructor()) {
	  TypeSig tU = ii.U;
	  TypeSig tT = TypeSig.getSig(spec.dmd.getContainingClass());
	  if (!Types.isSubclassOf(tT, tU)) {
	    // "m" is a constructor, and "U" is not a superclass or
	    // superinterface of "T"
	    // Add to antecedent the conjunct:  s != this
	    ante.addElement(GC.nary(TagConstants.REFNE, ii.s, GC.thisvar));
	  } else if (Types.isSameType(tU, tT) &&
		     spec.dmd.throwsSet.size() != 0) {
	    // "m" is a constructor, and "U" is "T", and throws set is nonempty
	    // Add to antecedent the conjunct:  (EC == ecReturn || s != this)
	    ante.addElement(
		GC.or(GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return),
		      GC.nary(TagConstants.REFNE, ii.s, GC.thisvar)));
	  }
	}
	Expr body = GC.implies(GC.and(ante), ii.J);
	Expr quant = GC.forall(ii.sdecl, body);
	Condition cond = GC.condition(TagConstants.CHKOBJECTINVARIANT, quant,
				      ii.prag.getStartLoc());
	spec.post.addElement(cond);
	if (Info.on) {
	  Info.out("[addInvariantBody: Including body-post-invariant at " +
		   Location.toString(ii.prag.getStartLoc()) + "]");
	}
      } else {
	if (Info.on) {
	  Info.out("[addInvariantBody: Skipping body-post-invariant at " +
		   Location.toString(ii.prag.getStartLoc()) + "]");
	}
      }
    }
  }
  
  /** Collects the axioms in <code>scope</code>. */
  
  private static ExprVec collectAxioms(/*@ non_null */ FindContributors scope) {

    ExprVec r = ExprVec.make();

    for( Enumeration enum = scope.typeSigs();
	 enum.hasMoreElements(); )
      {

	TypeDecl td = ((javafe.tc.TypeSig)enum.nextElement()).getTypeDecl();

	for (int i = 0; i < td.elems.size(); i++) {
	  TypeDeclElem tde = td.elems.elementAt(i);
	  if (tde.getTag() == TagConstants.AXIOM) {
	    ExprDeclPragma axiom = (ExprDeclPragma)tde;
	    if (!Main.filterInvariants ||
		exprIsVisible(scope.originType, axiom.expr)) {
	      r.addElement( TrAnExpr.trSpecExpr( axiom.expr, null, null ) );
	    }
	  }
	}
      }

    return r;
  }

    private static Enumeration collectOtis(TypeDeclElemVec vec) {
	Hashtable h = new Hashtable();
	for (int i = 0; i < vec.size(); i++) {
	    TypeDeclElem td = vec.elementAt(i);
	    if (td.getTag() == TagConstants.OTI) {
		h.put(td, td);
	    }
	}
	return h.elements();
    }

    private static Enumeration collectInvariants(TypeDeclElemVec vec) {
	Hashtable h = new Hashtable();
	for (int i = 0; i < vec.size(); i++) {
	    TypeDeclElem td = vec.elementAt(i);
	    if (td.getTag() == TagConstants.INVARIANT) {
		h.put(td, td);
	    }
	}
	return h.elements();
    }

    private static Enumeration collectGlobalInvariants(TypeDeclElemVec vec) {
	Hashtable h = new Hashtable();
	for (int i = 0; i < vec.size(); i++) {
	    TypeDeclElem td = vec.elementAt(i);
	    if (td.getTag() == TagConstants.GLOBAL_INVARIANT) {
		h.put(td, td);
	    }
	}
	return h.elements();
    }

    private static Enumeration collectGlobalAndObjectInvariants(TypeDeclElemVec vec) {
	Hashtable h = new Hashtable();
	for (int i = 0; i < vec.size(); i++) {
	    TypeDeclElem td = vec.elementAt(i);
	    int tag = td.getTag();
	    if (tag == TagConstants.GLOBAL_INVARIANT ||
		tag == TagConstants.INVARIANT) {
		h.put(td, td);
	    }
	}
	return h.elements();
    }

    /* Collects the object invariants in <code>scope</code> */
    public static InvariantInfo collectInvariants(/*@ non_null */ FindContributors scope) {
	Enumeration enum;
	if (Main.restrictScope) {
	    TypeDecl td = scope.originType.getTypeDecl();
	    enum = collectInvariants(td.elems);
	} else {
	    enum = scope.invariants();
	}
	return fillInvariantInfo(scope, enum);
    }

    /* Collects the global invariants in <code>scope</code> */
    public static InvariantInfo collectGlobalInvariants(/*@ non_null */ FindContributors scope) {
	Enumeration enum;
	if (Main.restrictScope) {
	    TypeDecl td = scope.originType.getTypeDecl();
	    enum = collectGlobalInvariants(td.elems);
	} else {
	    enum = scope.globalInvariants();
	}
	return fillInvariantInfo(scope, enum);	
    }

    /* Collects the otis in <code>scope</code> */
    public static InvariantInfo collectOtis(/*@ non_null */ FindContributors scope) {
	Enumeration enum;
	if (Main.restrictScope) {
	    TypeDecl td = scope.originType.getTypeDecl();
	    enum = collectOtis(td.elems);
	} else {
	    enum = scope.otis();
	}
	return fillInvariantInfo(scope, enum);	
    }

    /* Collects the global and object invariants in <code>scope</code> */
    public static InvariantInfo collectGlobalAndObjectInvariants(/*@ non_null */ FindContributors scope) {
	Enumeration enum;
	if (Main.restrictScope) {
	    TypeDecl td = scope.originType.getTypeDecl();
	    enum = collectGlobalAndObjectInvariants(td.elems);
	} else {
	    enum = scope.globalAndObjInvariants();
	}
	return fillInvariantInfo(scope, enum);
    }

    /*
    public static InvariantInfo collectInvariants(FindContributors scope) {
	Enumeration enum = scope.invariants();
	return fillInvariantInfo(scope, enum);
    }

    public static InvariantInfo collectGlobalInvariants(FindContributors scope) {
	Enumeration enum = scope.globalInvariants();
	return fillInvariantInfo(scope, enum);	
    }

    public static InvariantInfo collectOtis(FindContributors scope) {
	Enumeration enum = scope.otis();
	return fillInvariantInfo(scope, enum);	
    }

    public static InvariantInfo collectGlobalAndObjectInvariants(FindContributors scope) {
	Enumeration enum = scope.globalAndObjInvariants();
	return fillInvariantInfo(scope, enum);
    }
    */

  /** Helper method for collecting the object invariants, global invariants or 
      otis in <code>scope</code>. */  
  private static InvariantInfo fillInvariantInfo(/*@ non_null */ FindContributors scope,
								 Enumeration enum) {
    InvariantInfo ii = null;
    InvariantInfo iiPrev = null;
 
   while (enum.hasMoreElements()) {
	  ExprDeclPragma ep = (ExprDeclPragma)enum.nextElement();
	  Expr J = ep.expr;

	  boolean Jvisible = !Main.filterInvariants ||
				exprIsVisible( scope.originType, J );
	  
	  // System.out.print( (Jvisible?"Visible":"Invisible")+": ");
	  // javafe.ast.PrettyPrint.inst.print(System.out, J );
	  // System.out.println();
	  
	  if( Jvisible ) {

	    // Add a new node at the end of "ii"
	    InvariantInfo invinfo = new InvariantInfo();
	    invinfo.prag = ep;
	    invinfo.U = TypeSig.getSig(ep.parent);
	    if (iiPrev == null)
	      ii = invinfo;
	    else
	      iiPrev.next = invinfo;
	    iiPrev = invinfo;
	    
	    // The following determines whether or not an invariant is a
	    // static invariant by, in essence, checking for occurrence
	    // of "this" in the guarded-command translation of "J", not
	    // in "J" itself.  (These yield different results in the
	    // unusual case that "J" mentioned "this" in a subexpression
	    // "E.g", where "g" is a static field.)
	    //  First, build the map "{this-->s}" for a new "s".

	    LocalVarDecl sdecl = UniqName.newBoundThis();
	    VariableAccess s = TrAnExpr.makeVarAccess(sdecl, Location.NULL);
	    Hashtable map = new Hashtable();
	    map.put(GC.thisvar.decl, s);

	    int cReplacementsBefore = TrAnExpr.getReplacementCount();

	    /*
	     * Unlike any body we may be translating, for these
	     * translations this's type is that of the type that contains
	     * the invariant J.
	     */
	    Type savedType = GC.thisvar.decl.type;
	    GC.thisvar.decl.type = TypeSig.getSig(ep.getParent());
	    /* Commented out by Shaz. Changed it to the statement underneath.
	       Since the hashtable in the second argument is applied everywhere 
	       the hashtable in the third argument is applied only inside \old(),
	       the new statement should be equivalent to the old one.  
	       Basically, I am doing this to get rid of the cautions generated 
	       while translating otis which can have \old() inside them.
	    */
	    /*
	       invinfo.J = TrAnExpr.trSpecExpr(J, map, null); 
	    */
	    invinfo.J = TrAnExpr.trSpecExpr(J, map, map); 
	    GC.thisvar.decl.type = savedType;


	    if (cReplacementsBefore == TrAnExpr.getReplacementCount()) {
	      // static invariant
	      invinfo.isStatic = true;
	      invinfo.sdecl = null;
	      invinfo.s = null;
	      invinfo.map = null;
	    } else {
	      invinfo.isStatic = false;
	      invinfo.sdecl = sdecl;
	      invinfo.s = s;
	      invinfo.map = map;
	    }
	  }
    }

    return ii;
  }

  /** Collects the parameters of <code>spec.args</code> and the static
    * fields in <code>scope</code>, whose types are class types.
    **/
  
  private static ParamAndGlobalVarInfo collectParamsAndGlobals(
				     /*@ non_null */ Spec spec,
				     /*@ non_null */ FindContributors scope) {
    ParamAndGlobalVarInfo vars = null;
    ParamAndGlobalVarInfo varPrev = null;

    // Add the parameters
    for (int i = 0; i < spec.dmd.args.size(); i++) {
      FormalParaDecl arg = spec.dmd.args.elementAt(i);
      TypeSig classSig;
      boolean isNonnull;
      if (i == 0 && arg == GC.thisvar.decl) {
	classSig = TypeSig.getSig(spec.dmd.getContainingClass());
	isNonnull = true;
      } else {
	classSig = Types.toClassTypeSig(arg.type);
	isNonnull = NonNullPragma(arg) != null;
      }
      
      if (classSig != null) {
	// The parameter is of a class type
	ParamAndGlobalVarInfo info = new ParamAndGlobalVarInfo();
	if (varPrev == null)
	  vars = info;
	else
	  varPrev.next = info;
	varPrev = info;

	info.vdecl = arg;
	info.U = classSig;
	info.isNonnull = isNonnull;
      }
    }

    // Add the static fields
    Enumeration enum = scope.fields();
    while (enum.hasMoreElements()) {
      FieldDecl fd = (FieldDecl)enum.nextElement();

      TypeSig classSig = Types.toClassTypeSig(fd.type);
	  
      if (classSig != null && Modifiers.isStatic(fd.modifiers) ) {
	    // the field is a static field of a class type
	    ParamAndGlobalVarInfo info = new ParamAndGlobalVarInfo();
	    if (varPrev == null)
	      vars = info;
	    else
	      varPrev.next = info;
	    varPrev = info;

	    info.vdecl = fd;
	    info.U = classSig;
	    info.isNonnull = NonNullPragma(fd) != null;
      }
    }

    return vars;
  }

  /** Shaves a GC designator. */
  
  private static VariableAccess shave(/*@ non_null */ Expr e)
    {
      switch( e.getTag() )
	{
	  case TagConstants.VARIABLEACCESS:
	    return (VariableAccess)e;

	  case TagConstants.SELECT:
	    return shave( ((NaryExpr)e).exprs.elementAt(0) );

	  default:
	    Assert.fail("Unexpected case");
	    return null;
	}
    }

  /** Finds and returns the first modifier pragma of <code>vdecl</code>
    * that has the tag <code>tag</code>, if any.  If none, returns
    * <code>null</code>.<p>
    *
    * Note, if you want to know whether a variable is <code>non_null</code>,
    * use method <code>NonNullPragma</code> instead, for it properly
    * handles inheritance of <code>non_null</code> pragmas.
    **/

  static public ModifierPragma findModifierPragma(/*@ non_null */ GenericVarDecl vdecl,
						  int tag) {
    if (vdecl.pmodifiers != null) {
      for (int j = 0; j < vdecl.pmodifiers.size(); j++) {
	ModifierPragma prag= vdecl.pmodifiers.elementAt(j);
	if (prag.getTag() == tag)
	  return prag;
      }
    }
    return null;  // not present
  }

  /** Creates and returns a new map that is <code>map</code> restricted
    * to the domain <code>e</code>.  Assumes that every element in
    * <code>e</code> is in the domain of <code>map</code>.
    **/

  //@ requires map.keyType == type(GenericVarDecl);
  //@ requires map.elementType == type(VariableAccess);
  //@ requires e.elementType == type(GenericVarDecl);
  static Hashtable restrict(/*@ non_null */ Hashtable map,
			    /*@ non_null */ Enumeration e) {
    Hashtable r = new Hashtable();
    while (e.hasMoreElements()) {
      GenericVarDecl vd = (GenericVarDecl)e.nextElement();
      VariableAccess variant = (VariableAccess)map.get(vd);
      Assert.notNull(variant);
      r.put(vd, variant);
    }
    return r;
  }
  
  /** Converts a GenericVarDecl enumeration to a mapping from those
      variables to new Variableaccesses. */
  
  //@ requires e.elementType == type(GenericVarDecl);
  static Hashtable makeSubst(/*@ non_null */ Enumeration e,
			     /*@ non_null */ String postfix )
    {
      Hashtable r = new Hashtable();
      while( e.hasMoreElements() )
	{
	  GenericVarDecl vd = (GenericVarDecl)e.nextElement();
	  VariableAccess variant = createVarVariant( vd, postfix );
	  r.put( vd, variant );
	}
      return r;
    }

  static Hashtable makeSubst(/*@ non_null */ FormalParaDeclVec args,
			     /*@ non_null */ String postfix) {
    Hashtable r = new Hashtable();
    for (int i = 0; i < args.size(); i++) {
      GenericVarDecl vd = args.elementAt(i);
      VariableAccess variant = createVarVariant(vd, postfix);
      r.put(vd, variant);
    }
    return r;
  }


  /**
   ** Given a GenericVarDecl named "x@old", returns a VariableAccess to a
   ** fresh LocalVarDecl named "x@<postfix>".
   **
   ** This handles ESCJ 23b case 13.
   **/
  static VariableAccess createVarVariant(/*@ non_null */ GenericVarDecl vd,
					 /*@ non_null */ String postfix) {
      String name = vd.id.toString();
      if (name.indexOf('@')!= -1)
	name = name.substring(0, name.indexOf('@'));

      Identifier id = Identifier.intern( name+"@"+postfix );
      LocalVarDecl ld =
	LocalVarDecl.make( vd.modifiers,
			   vd.pmodifiers,
			   id,
			   vd.type,
			   vd.locId,
			   null, Location.NULL );
      VariableAccess va = VariableAccess.make( id, vd.locId, ld );

      return va;
    }


  /** Adds to <code>cv</code> a condition stating that <code>vd</code>
    * has type <code>type</code>.
    **/
      
  private static void addFreeTypeCorrectAs(GenericVarDecl vd, Type type,
					   ConditionVec cv) {
    Expr e = TrAnExpr.typeCorrectAs(vd, type);
    Condition cond = GC.freeCondition(e, Location.NULL);
    cv.addElement(cond);
  }


  public static GuardedCmd surroundBodyBySpec(Translate gctranslator,
					      GuardedCmd body, Spec spec,
					      FindContributors scope,
					      Set syntargets,
					      Hashtable premap,
					      int locBeginCurlyBrace,
					      int locEndCurlyBrace) {
    StackVector code = new StackVector();
    GuardedCmd gc;
    YieldCmd yc;

    FormalParaDeclVec vec = gctranslator.rdCurrent.args;
    Vector localVarDeclVec = new Vector();
    for (int i = 0; i < vec.size(); i++) {
	FormalParaDecl d = vec.elementAt(i);
	localVarDeclVec.addElement(d);
    }

    code.push();

    /* Add the yield corresponding to the precondition */

    code.push();
    assumeConditions(spec.pre, code);
    gc = GC.seq(GuardedCmdVec.popFromStackVector(code));
    yc = GC.yield(locBeginCurlyBrace, gc, localVarDeclVec);
    code.addElement(yc);
    for (int i = 0; i < spec.pre.size(); i++) {
      Condition cond = spec.pre.elementAt(i);
      gctranslator.addRHS(false, yc, cond.pred);
    }    

    /*
    code.push();
    assumeConditions(spec.pre, code);
    yc = GC.yield(locBeginCurlyBrace, localVarDeclVec);
    for (int i = 0; i < spec.pre.size(); i++) {
      Condition cond = spec.pre.elementAt(i);
      gctranslator.addRHS(yc, cond.pred);
    }    
    yc.code = GC.seq(GuardedCmdVec.popFromStackVector(code));
    code.addElement(yc);
    */

    /* Add the body */
    code.addElement(body);

    /* Add the yield corresponding to the postcondition */
    code.push();
    checkConditions(spec.post, locEndCurlyBrace, code);
    gc = GC.seq(GuardedCmdVec.popFromStackVector(code));
    yc = GC.yield(locEndCurlyBrace, gc, localVarDeclVec);
    code.addElement(yc);
    
    for (int i = 0; i < spec.post.size(); i++) {
      Condition cond = spec.post.elementAt(i);
      //System.out.println(TagConstants.toString(cond.label));
      Assert.notFalse(/*cond.label == TagConstants.CHKGLOBALINVARIANT || */
		      cond.label == TagConstants.CHKOBJECTINVARIANT ||
		      cond.label == TagConstants.CHKPOSTCONDITION ||
		      cond.label == TagConstants.CHKUNEXPECTEDEXCEPTION ||
		      cond.label == TagConstants.CHKFREE ||
		      cond.label == TagConstants.CHKOWNERNULL);
      /* Don't add global invariants to postconditions */
      if (cond.label == TagConstants.CHKOBJECTINVARIANT ||
	  cond.label == TagConstants.CHKPOSTCONDITION) {
	  gctranslator.addRHS(false, yc, cond.pred);
      }
    }    
	
    /*
    // Nothing to do here since modifies constraints are not 
    // being checked currently.
    checkModifiesConstraints(spec, scope, syntargets, premap,
			     locEndCurlyBrace, code);
    */

    return GC.seq(GuardedCmdVec.popFromStackVector(code));
  }

  /** Returns a command that first does an <code>assume</code> for
    * each precondition in <code>spec</code>, then does <code>body</code>,
    * then checks the postconditions of <code>spec</code>, and finally
    * checks the modifies constraints implied by <code>spec</code>.
    **/
  
  public static GuardedCmd surroundBodyBySpec(GuardedCmd body, Spec spec,
					      FindContributors scope,
					      Set syntargets,
					      Hashtable premap,
					      int locEndCurlyBrace) {
    StackVector code = new StackVector();
    code.push();

    assumeConditions(spec.pre, code);
    code.addElement(body);
    checkConditions(spec.post, locEndCurlyBrace, code);
    checkModifiesConstraints(spec, scope, syntargets, premap,
			     locEndCurlyBrace, code);

    return GC.seq(GuardedCmdVec.popFromStackVector(code));
  }

  /** Appends <code>code</code> with an <code>assume C</code> command
    * for every condition <code>C</code> in <code>cv</code>.
    **/
  
  private static void assumeConditions(ConditionVec cv, StackVector code) {
    for (int i = 0; i < cv.size(); i++) {
      Condition cond = cv.elementAt(i);
      code.addElement(GC.assumeAnnotation(cond.locPragmaDecl, cond.pred));
    }
  }
  
  /** Appends <code>code</code> with an <code>check loc: C</code>
    * command for every condition <code>C</code> in <code>cv</code>.
    **/
  
  private static void checkConditions(ConditionVec cv, int loc, StackVector code) {
    for (int i = 0; i < cv.size(); i++) {
      Condition cond = cv.elementAt(i);
      // if the condition is an object invariant, send its guarded command
      // translation as auxiliary info to GC.check
      if (cond.label == TagConstants.CHKOBJECTINVARIANT)
	code.addElement(GC.check(loc, cond, cond.pred));
      else
	  code.addElement(GC.check(loc, cond));
    }
  }

  private static void checkModifiesConstraints(Spec spec,
					       FindContributors scope,
					       Set syntargets,
					       Hashtable premap,
					       int loc,
					       StackVector code) {
    // TBW
  }

  private static InvariantInfo mergeInvariants(InvariantInfo ii) {
    if( !Main.mergeInv )
      return ii;
    
    // Combined static/dynamic invariants, indexed by TypeSIg
    Hashtable staticInvs = new Hashtable();
    Hashtable dynInvs = new Hashtable();

    while( ii != null ) {

      Hashtable table = ii.isStatic ? staticInvs : dynInvs;

      InvariantInfo old = (InvariantInfo)table.get( ii.U );

      if( old == null ) {

	table.put( ii.U, ii );

      } else {

	// Add ii to old
	old.J = GC.and( old.J,
			ii.isStatic ? ii.J
				    : GC.subst( ii.sdecl, old.s, ii.J ));
      }

      ii = ii.next;
    }

    // Now pull out merged invariants from tables
    Hashtable[] tables = { staticInvs, dynInvs };
    ii = null;

    for( int i=0; i<2; i++ ) {
      Hashtable table = tables[i];

      for( Enumeration e = table.elements(); e.hasMoreElements(); ) {
	InvariantInfo t = (InvariantInfo)e.nextElement();
	t.next = ii;
	ii = t;
      }
    }

    return ii;
  }

  private static boolean exprIsVisible(TypeSig originType, Expr e) {

    switch( e.getTag() ) {

      case TagConstants.FIELDACCESS:
	{
	  FieldAccess fa = (FieldAccess)e;
	  FieldDecl decl = fa.decl;

	  if( fa.od instanceof ExprObjectDesignator
	      && !exprIsVisible( originType,
				 ((ExprObjectDesignator)fa.od).expr ) )
	    return false;

	  if( decl.parent == null ) {
	    // for array.length "field", there is no parent
	    return true;
	  } else if (GetSpec.findModifierPragma(decl,
						TagConstants.SPEC_PUBLIC) != null) {
	    return true;
	  } else {
	    TypeSig sigDecl = TypeSig.getSig( decl.parent );
	    return TypeCheck.inst.canAccess( originType, sigDecl,
					     decl.modifiers,
					     decl.pmodifiers );
	  }
	}

      default:
	{
	  // Recurse over all children
	  for(int i=0; i<e.childCount(); i++ ) {
	    Object o = e.childAt(i);
	    if( o instanceof Expr ) {
	      if( !exprIsVisible(originType, (Expr)o) )
		return false;
	    }
	  }

	  return true;
	}
    }
  }


    /***************************************************
     *                                                 *
     * Handling NON_NULL:			       *
     *                                                 *
     ***************************************************/


    /**
     ** Decorates <code>GenericVarDecl</code>'s to point to
     ** NonNullPragmas (SimpleModifierPragma's).
     **/
    //@ invariant nonnullDecoration!=null
    /*@ invariant nonnullDecoration.decorationType ==
				type(SimpleModifierPragma)  */
    private static ASTDecoration nonnullDecoration
      = new ASTDecoration("nonnullDecoration");


    /**
     ** Mark v as non_null because of non_null pragma nonnull.
     **
     ** Precondition: nonnull is a NON_NULL pragma.
     **
     ** (Used to implement inheritence of non_null's.)
     **/
    private static void setNonNullPragma(GenericVarDecl v,
					 SimpleModifierPragma nonnull) {
	nonnullDecoration.set(v, nonnull);
    }


    /**
     ** Has a particular declaration been declared non_null?  If so,
     ** return the non_null pragma responsible.  Otherwise, return null.<p>
     **
     ** Precondition: if the declaration is a formal parameter, then the
     ** spec of it's routine has already been computed.<p>
     **
     **
     ** WARNING: this is the only authorized way to determine this
     ** information.  Do *not* attempt to search for NON_NULL modifiers
     ** directly.  (This is needed to handle inherited NON_NULL's
     ** properly.)
     **/
    public static SimpleModifierPragma NonNullPragma(GenericVarDecl v) {
	// First check for a mark:
	SimpleModifierPragma mark = (SimpleModifierPragma)
			nonnullDecoration.get(v);
	if (mark!=null)
	    return mark;

	// Else fall back on a direct search of local modifiers:
	return (SimpleModifierPragma)
			findModifierPragma(v, TagConstants.NON_NULL);
    }
}


/** This class is used by <code>collectParamsAndGlobalVars</code> and its
 ** caller, <code>extendSpecForCall</code>.
 **/

class ParamAndGlobalVarInfo {
  GenericVarDecl vdecl;
  TypeSig U;
  boolean isNonnull;
  ParamAndGlobalVarInfo next;
}
