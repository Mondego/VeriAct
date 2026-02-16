/* $Id: GetSpec.java 72039 2008-07-23 03:54:08Z chalin $
 * Copyright 2000, 2001, Compaq Computer Corporation 
 */

package escjava.translate;

import escjava.Main;
import escjava.ast.*;
import escjava.ast.Modifiers;
import escjava.ast.TagConstants;
import escjava.backpred.FindContributors;
import escjava.tc.TypeCheck;
import escjava.tc.Types;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.Hashtable;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Iterator;
import java.util.Vector;
import javafe.ast.*;
import javafe.tc.TypeSig;

import javafe.util.*;

/**
 * Responsible for getting Spec for calls. Includes ... from ESCJ16c.
 */

public final class GetSpec {

	public static /*@ non_null */ Spec getSpecForCall(/*@ non_null */RoutineDecl rd,
			/*@ non_null */FindContributors scope, /*@ nullable */ Set predictedSynTargs,
			/*@ non_null */ RoutineDecl callingRoutine) {
		Spec spec = getCommonSpec(rd, scope, null);
		return extendSpecForCall(spec, scope, predictedSynTargs,callingRoutine);
	}

	/* used for calls that are inlined */
	public static  /*@ non_null */ Spec getSpecForInline(/*@ non_null */RoutineDecl rd,
			/*@ non_null */FindContributors scope) {
		Spec spec = getCommonSpec(rd, scope, null);
		return spec;
		// TBW: Should also add NonNullInit checks somewhere!
	}

	public static /*@ non_null */ Spec getSpecForBody(/*@ non_null */RoutineDecl rd,
			/*@ non_null */FindContributors scope,
			/*@ non_null */Set synTargs, /*@ nullable */ Hashtable premap) {
		Spec spec = getCommonSpec(rd, scope, premap);
		if(Main.options().idc)
			DefGCmd.addInvConds(scope, spec);
		if (rd.body==null && Main.options().idc) {
			return(spec);
		}	else {
			return extendSpecForBody(spec, scope, synTargs);
		}
	}

	private static /*@ non_null @*/ Spec getCommonSpec(
			/*@ non_null */RoutineDecl rd,
			/*@ non_null */FindContributors scope, /*@ nullable */ Hashtable premap) {
		/*
		 * Need to typecheck TypeDecl containing callee so that
		 * requires/ensures/modifies clauses etc are resolved.
		 */

		TypeSig T = TypeSig.getSig(rd.parent);
		T.typecheck();

		DerivedMethodDecl combined = getCombinedMethodDecl(rd);
		DerivedMethodDecl filtered = filterMethodDecl(combined, scope);

		/*
		 * If we are translating the spec for an inner-class constructor, then we
		 * need to substitute this$0arg for this.this$0 everywhere:
		 */
		Spec spec = null;
		try {
			if (filtered.isConstructor() && !T.isTopLevelType()) {
				Inner.firstThis0 = Inner
				.getEnclosingInstanceArg((ConstructorDecl)filtered.original);
			}

			spec = trMethodDecl(filtered, premap);
		} finally {
			Inner.firstThis0 = null;
		}

		//System.out.println("COMMON END " + spec.postAssumptions.size());
		return spec;
	}

	static private /*@ non_null @*/ ASTDecoration dmdDecoration = new ASTDecoration("dmd");

	/**
	 * * Implement GetCombinedMethodDecl from ESCJ 16c section 7:
	 * <p>* * Precondition: the type declaring rd has been typechecked.
	 * <p>* * Note: this routine may typecheck the supertypes of the type that *
	 * declares rd.
	 */
	//@ ensures \result != null;
	public static /*@ non_null */ DerivedMethodDecl getCombinedMethodDecl(
			/*@ non_null */RoutineDecl rd) 
	{
		DerivedMethodDecl dmd = (DerivedMethodDecl)dmdDecoration.get(rd);
		//if (dmd != null) { System.out.println("RETRIEVED SPECS"); ((EscPrettyPrint)EscPrettyPrint.inst).printDMD(System.out,0,dmd);}
		if (dmd != null) return dmd;

		dmd = new DerivedMethodDecl(rd);
		dmdDecoration.set(rd, dmd);

		dmd.throwsSet = rd.raises;
		dmd.requires = ExprModifierPragmaVec.make();
		dmd.modifies = ModifiesGroupPragmaVec.make();
		dmd.ensures = ExprModifierPragmaVec.make();
		dmd.exsures = VarExprModifierPragmaVec.make();

		if (rd instanceof ConstructorDecl) {
			// Handle constructor case:
			dmd.args = rd.args;
			TypeSig thisType = TypeSig.getSig(rd.parent);
			if (!thisType.isTopLevelType()) {
				// An Inner class; add this$0 argument:
				dmd.args = rd.args.copy();
				FormalParaDecl this0arg = Inner
				.getEnclosingInstanceArg((ConstructorDecl)rd);
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
			 * Add modifiers from this method as well as all methods it overrides;
			 * also handle non_null:
			 */

			addModifiersToDMD(dmd, md);
			//{ System.out.println("SELF SPECS"); ((EscPrettyPrint)EscPrettyPrint.inst).printDMD(System.out,0,dmd);}
			Set overrides = escjava.tc.FlowInsensitiveChecks.getAllOverrides(md);
			Enumeration overridden_methods = overrides.elements();
			while (overridden_methods.hasMoreElements()) {
				MethodDecl smd = (MethodDecl)overridden_methods.nextElement();
				TypeSig.getSig(smd.parent).typecheck();

				addModifiersToDMD(dmd, smd);
			}
			//{ System.out.println("COMBINED SPECS"); ((EscPrettyPrint)EscPrettyPrint.inst).printDMD(System.out,0,dmd);}
		}

		dmd.computeFreshUsage();

		return dmd;
	}

	/**
	 * * Add the modifiers of rd to dmd, making any necessary substitions * of
	 * parameter names. Also propagates non_null's.
	 * <p>* * Precondition: rd has been typechecked.
	 * <p>
	 */
	private static void addModifiersToDMD(/*@ non_null */DerivedMethodDecl dmd,
			/*@ non_null */RoutineDecl rd) {
		/*
		 * Compute the substitution on parameter names to change a spec for an
		 * overridden method to refer to our method's parameter names instead of
		 * its; propagate non_nulls:
		 */
		Hashtable subst = new Hashtable();
		//System.out.println("SUBST " + rd.parent.id.toString());
		if (rd != dmd.original) {
			for (int i = 0; i < rd.args.size(); i++) {
				GenericVarDecl newDecl = dmd.original.args.elementAt(i);
				GenericVarDecl oldDecl = rd.args.elementAt(i);

				// This may no longer be necessary, but it doesn't hurt
				SimpleModifierPragma nonnull = NonNullPragma(oldDecl);
				if (nonnull != null) setNonNullPragma(newDecl, nonnull);

				VariableAccess va = VariableAccess.make(newDecl.id, Location.NULL,
						newDecl);

				subst.put(oldDecl, va);
				//        escjava.ast.EscPrettyPrint.inst.print(System.out,oldDecl);
				//        System.out.print(" -> ");
				//        escjava.ast.EscPrettyPrint.inst.print(System.out,0,va);
				//        System.out.println("");
			}
		}

		if (rd.pmodifiers == null) return;

		for (int i = 0; i < rd.pmodifiers.size(); i++) {
			// We omit pragmas that have something notimplemented, but carry on
			// with the rest
			try {
				ModifierPragma mp = rd.pmodifiers.elementAt(i);
				switch (mp.getTag()) {
				case TagConstants.REQUIRES:
				case TagConstants.ALSO_REQUIRES:
				case TagConstants.PRECONDITION: {
					ExprModifierPragma emp = (ExprModifierPragma)mp;
					emp = doSubst(subst, emp);
					dmd.requires.addElement(emp);
					break;
				}
				case TagConstants.MODIFIESGROUPPRAGMA: {
					ModifiesGroupPragma em = (ModifiesGroupPragma)mp;
					if (em.precondition != null) em.precondition = doSubst(subst,em.precondition); // FIXME - don't like modifiying the original
					for (int ii = 0; ii < em.items.size(); ++ii) {
						CondExprModifierPragma emp = em.items.elementAt(ii);
						if (emp.expr == null) {
							em.items.removeElementAt(i);
							--ii;
							continue;
						}
						int t = emp.expr.getTag();
						// FIXME - no contribution to spec for these keywords
						if (t == TagConstants.EVERYTHINGEXPR) {
							dmd.modifiesEverything = true;
						} else if (t == TagConstants.NOTSPECIFIEDEXPR) {
							dmd.modifiesEverything = true;
							emp.expr = EverythingExpr.make(emp.expr.getStartLoc());
							//} else if (t == TagConstants.NOTHINGEXPR ) {
							// no action
						}
						emp = doSubst(subst, emp);
						em.items.setElementAt(emp, ii);  // FIXME - I don't like this modification of the original -- DRC
					}
					dmd.modifies.addElement(em);
					break;
				}
				/*
				 * case TagConstants.MODIFIES: case TagConstants.ALSO_MODIFIES: case
				 * TagConstants.MODIFIABLE: case TagConstants.ASSIGNABLE: {
				 * CondExprModifierPragma emp = (CondExprModifierPragma)mp; if
				 * (emp.expr == null) break; // ignore - informal int t =
				 * emp.expr.getTag(); // FIXME - no contribution to spec for these
				 * keywords if (t == TagConstants.EVERYTHINGEXPR) {
				 * dmd.modifiesEverything = true; } else if (t ==
				 * TagConstants.NOTSPECIFIEDEXPR) { dmd.modifiesEverything = true;
				 * //emp = doSubst(subst, // EverythingExpr.make(emp.getStartLoc()) );
				 * break; // FIXME } else if (t == TagConstants.NOTHINGEXPR ) { // no
				 * action } else { } emp = doSubst(subst, emp);
				 * dmd.modifies.addElement(emp); break; }
				 */
				case TagConstants.ENSURES:
				case TagConstants.ALSO_ENSURES:
				case TagConstants.POSTCONDITION: {
					ExprModifierPragma emp = (ExprModifierPragma)mp;
					int t = emp.errorTag;
					emp = doSubst(subst, emp);
					emp.errorTag = t;
					dmd.ensures.addElement(emp);
					break;
				}
				case TagConstants.NON_NULL:
					/*
					 * if (dmd.nonnull == null) { dmd.nonnull =
					 * (SimpleModifierPragma)mp; }
					 */
					break;
				case TagConstants.EXSURES:
				case TagConstants.ALSO_EXSURES:
				case TagConstants.SIGNALS: {
					VarExprModifierPragma vemp = (VarExprModifierPragma)mp;
					vemp = doSubst(subst, vemp);
					dmd.exsures.addElement(vemp);
					break;
				}
				default:
					break;
				}
			} catch (NotImplementedException e) {
				// Error message already printed
			}
		}
	}

	/** Perform a substitution on an Expr * */
	private static /*@ non_null */ Expr doSubst(/*@ non_null */ Hashtable subst,
			/*@ non_null @*/ Expr expr) {
		return 
		Substitute.doSubst(subst, expr);
	}

	/** Perform a substitution on an ExprModifierPragma * */
	private static /*@ non_null */ ExprModifierPragma doSubst(/*@ non_null */ Hashtable subst,
			/*@ non_null @*/ ExprModifierPragma emp) {
		return ExprModifierPragma.make(emp.tag,
				Substitute.doSubst(subst, emp.expr), emp.loc);
	}

	/** Perform a substitution on a CondExprModifierPragma * */
	private static /*@ non_null */ CondExprModifierPragma doSubst(/*@ non_null */ Hashtable subst,
			/*@ non_null @*/ CondExprModifierPragma emp) 
	{
		return CondExprModifierPragma.make(emp.tag, Substitute.doSubst(subst,
				emp.expr), emp.loc, emp.cond == null ? null : Substitute.doSubst(subst,
						emp.cond));
	}

	/** Perform a substitution on a VarExprModifierPragma * */
	//@ ensures \result != null;
	private static /*@ non_null */ VarExprModifierPragma doSubst(/*@ non_null */ Hashtable subst,
			/*@ non_null @*/ VarExprModifierPragma vemp) 
	{
		VarExprModifierPragma v =
			VarExprModifierPragma.make(vemp.tag, vemp.arg, Substitute.doSubst(
					subst, vemp.expr), vemp.loc);
		v.setOriginalTag(vemp.originalTag());
		return v;
	}

	//@ ensures \result != null;
	public static /*@ non_null */ DerivedMethodDecl filterMethodDecl(
			/*@ non_null */DerivedMethodDecl dmd,
			/*@ non_null */FindContributors scope) {
		if (!Main.options().filterMethodSpecs) { return dmd; }

		DerivedMethodDecl dmdFiltered = new DerivedMethodDecl(dmd.original);
		dmdFiltered.args = dmd.args;
		dmdFiltered.returnType = dmd.returnType;
		dmdFiltered.throwsSet = dmd.throwsSet;

		dmdFiltered.requires = dmd.requires;
		dmdFiltered.modifies = filterModifies(dmd.modifies, scope);
		dmdFiltered.ensures = filterExprModPragmas(dmd.ensures, scope);
		dmdFiltered.exsures = filterVarExprModPragmas(dmd.exsures, scope);

		dmdFiltered.computeFreshUsage();

		return dmdFiltered;
	}

	//@ ensures \result != null;
	private static /*@ non_null */ ExprModifierPragmaVec filterExprModPragmas(
			/*@ non_null */ExprModifierPragmaVec vec,
			/*@ non_null */FindContributors scope) {
		int n = vec.size();
		ExprModifierPragmaVec vecNew = null; // lazily allocated
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
					vecNew = ExprModifierPragmaVec.make(n - 1);
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

	//@ ensures \result != null;
	private static /*@ non_null */ ModifiesGroupPragmaVec filterModifies(
			/*@ non_null */ModifiesGroupPragmaVec mvec,
			/*@ non_null */FindContributors scope) {
		ModifiesGroupPragmaVec result = ModifiesGroupPragmaVec.make();
		int mn = mvec.size();
		for (int j = 0; j < mn; ++j) {
			ModifiesGroupPragma mv = mvec.elementAt(j);
			CondExprModifierPragmaVec vec = mv.items;
			CondExprModifierPragmaVec vecNew = null; // lazily allocated
			int n = vec.size();
			for (int i = 0; i < n; i++) {
				CondExprModifierPragma vemp = vec.elementAt(i);
				if (exprIsVisible(scope.originType, vemp.expr)
						&& exprIsVisible(scope.originType, vemp.cond)) {
					// keep this pragma
					if (vecNew != null) {
						vecNew.addElement(vemp);
					}
				} else {
					// filter out this pragma
					if (vecNew == null) {
						vecNew = CondExprModifierPragmaVec.make(n - 1);
						for (int k = 0; k < i; k++) {
							vecNew.addElement(vec.elementAt(k));
						}
					}
				}
			}
			result.addElement(ModifiesGroupPragma.make(mv.tag, mv.clauseLoc).append(
					vecNew == null ? vec : vecNew));
		}
		return result;
	}

	//@ ensures \result != null;
	private static /*@ non_null */ VarExprModifierPragmaVec filterVarExprModPragmas(
			/*@ non_null */VarExprModifierPragmaVec vec,
			/*@ non_null */FindContributors scope) {
		int n = vec.size();
		VarExprModifierPragmaVec vecNew = null; // lazily allocated
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
					vecNew = VarExprModifierPragmaVec.make(n - 1);
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

	//@ ensures \result != null;
	private static /*@ non_null */ Spec trMethodDecl(/*@ non_null */DerivedMethodDecl dmd,
			/*@ nullable */ Hashtable premap) {
		Assert.notNull(dmd);

		/*
		 * Unlike any body we may be translating, for these translations this's type
		 * is that of the type that contains the method or constructor dmd.original.
		 */
		TypeSig T = TypeSig.getSig(dmd.getContainingClass());
		Type savedType = GC.thisvar.decl.type;
		GC.thisvar.decl.type = T;

		// (we restore GC.thisvar.decl.type just before returning)

		ExprVec preAssumptions = ExprVec.make();
		ConditionVec pre = trMethodDeclPreconditions(dmd, preAssumptions);

		ExprVec targets = ExprVec.make();
		ExprVec specialTargets = ExprVec.make();

		if (!Utils.isPure(dmd.original)) {
			targets.addElement(GC.statevar);
			specialTargets.addElement(GC.statevar);
		}
		if (dmd.usesFresh) targets.addElement(GC.allocvar);
		if (dmd.usesFresh) specialTargets.addElement(GC.allocvar);

		// translates modifies list

		for (int k = 0; k < dmd.modifies.size(); ++k) {
			Frame.ModifiesIterator ii = new Frame.ModifiesIterator(
					dmd.getContainingClass(),dmd.modifies.elementAt(k).items, true, true);
			while (ii.hasNext()) {
				Expr designator = (Expr)ii.next();
				//if (Utils.isModel(designator)) continue;
				Expr gcDesignator = TrAnExpr.trSpecExpr(designator);
				// Returns null for modifies \nothing, \everything FIXME?
				// array-range, wild-ref expressions FIXME!!
				if (gcDesignator != null) {
					targets.addElement(gcDesignator);
				} else if (designator instanceof ArrayRangeRefExpr) {
					targets.addElement(GC.elemsvar);
				} else if (designator instanceof EverythingExpr) {
					targets.addElement(GC.elemsvar);
				}
			}
		}

		// handle targets stuff, and create preVarMap

		Set roots = new Set(); // set of GenericVarDecls

		for (int i = 0; i < targets.size(); i++) {
			Expr gcDesignator = targets.elementAt(i);
			VariableAccess shaved = shave(gcDesignator);
			roots.add(shaved.decl);
		}

		Hashtable preVarMap = premap;
		if (premap == null) preVarMap = makeSubst(roots.elements(), "pre");
		//else
		//    preVarMap = restrict( premap, roots.elements() );

		/*
		 * Re the change above: premap is a map from variables with a @pre suffix to
		 * their declarations; preVarMap is the relevant piece of this for the
		 * currnet method. However, that was determined by the set of locations
		 * specified in modifies clauses. That leads to erroneous behavior if the
		 * modifies clause is incorrect.
		 * 
		 * The change is to use the premap without restriction. That allows the
		 * verification of a body of a method to proceed without dependence on the
		 * accuracy of the modifies clause. However it also adds a lot of conjuncts
		 * into the verification condition - and the premap is accumulated from the
		 * entire class declaration. An improvement would be to simply use the
		 * premap generated from the uses of \old in the body of the method + the
		 * spec of the method + the spec of the class.
		 */
		// Now create the postconditions
		ExprVec postAssumptions = ExprVec.make();
		ConditionVec post = trMethodDeclPostcondition(dmd, preVarMap,
				postAssumptions);
		java.util.Set postlocs = new java.util.HashSet();
		int size = post.size();
		for (int ic = 0; ic < size; ++ic) {
			collectFields(post.elementAt(ic).pred, postlocs);
		}

		Spec spec = Spec.make(dmd, targets, specialTargets, preVarMap,
				preAssumptions, pre, postAssumptions, post,
				false && dmd.modifiesEverything, postlocs); // FIXME - turning off
				// modifies everything for
		// now

		GC.thisvar.decl.type = savedType;

		//System.out.println("TRMETHOD END " + spec.postAssumptions.size());
		return spec;
	}

	/** Computes the preconditions, according to section 7.2.0 of ESCJ 16. */

	//@ ensures \result != null;
	private static /*@ non_null */ ConditionVec trMethodDeclPreconditions(
			/*@ non_null */DerivedMethodDecl dmd, /*@ non_null */ ExprVec preAssumptions) {
		ConditionVec pre = ConditionVec.make();

		// Account for properties about parameters
		for (int i = 0; i < dmd.args.size(); i++) {
			FormalParaDecl arg = dmd.args.elementAt(i);

			if (i == 0 && arg == GC.thisvar.decl) {
				// the special parameter "this"
				addFreeTypeCorrectAs(arg, TypeSig.getSig(dmd.getContainingClass()), pre);
				VariableAccess argAccess = TrAnExpr.makeVarAccess(arg, Location.NULL);
				Expr nonnull = GC.nary(TagConstants.REFNE, argAccess, GC.nulllit);
				Condition cond = GC.freeCondition(nonnull, Location.NULL);
				pre.addElement(cond);

			} else {
				// regular parameters
				addFreeTypeCorrectAs(arg, arg.type, pre);
				/*
				 * non_null is handled in the desugaring SimpleModifierPragma
				 * nonNullPragma = NonNullPragma(arg); if (nonNullPragma != null) {
				 * VariableAccess argAccess = TrAnExpr.makeVarAccess(arg,
				 * Location.NULL); Expr nonnull = GC.nary(TagConstants.REFNE, argAccess,
				 * GC.nulllit); Condition cond = GC.freeCondition(nonnull,
				 * nonNullPragma.getStartLoc()); pre.addElement(cond); }
				 */
			}
		}

		if (dmd.isConstructor()) {
			// Free: RES != null && !isAllocated(RES, wt[[alloc]])
			Expr nonnull = GC.nary(TagConstants.REFNE, GC.resultvar, GC.nulllit);
			Expr allocated = GC.nary(TagConstants.ISALLOCATED, GC.resultvar,
					GC.allocvar);
			//(VariableAccess)wt.get(GC.allocvar.decl));
			Expr notAllocated = GC.not(allocated);
			preAssumptions.addElement(nonnull);
			preAssumptions.addElement(notAllocated);
		}
		// Add the declared preconditions

		// Make the disjunction of all of the preconditions

		java.util.Set axsToAdd = new java.util.HashSet();
		if (dmd.requires.size() != 0) {
			Expr expr = dmd.requires.elementAt(0).expr;
			int loc = dmd.requires.elementAt(0).getStartLoc();
			for (int i = 1; i < dmd.requires.size(); ++i) {
				ExprModifierPragma e = dmd.requires.elementAt(i);
				if (loc == Location.NULL) loc = e.getStartLoc();
				expr = BinaryExpr.make(TagConstants.OR, expr, e.expr, loc);
				javafe.tc.FlowInsensitiveChecks.setType(expr, Types.booleanType);
			}
			TrAnExpr.initForClause();

			Hashtable map = null;
			if (dmd.isConstructor()) {
				map = new Hashtable();
				map.put(GC.thisvar.decl, GC.resultvar);
				//map.put(GC.thisvar.decl, temporary(GC.resultvar.id.toString(), scall,
				// scall)
			}
			Expr pred = TrAnExpr.trSpecExpr(expr, map, null);

			//Expr pred = TrAnExpr.trSpecExpr(expr);
			if (TrAnExpr.extraSpecs) {
				preAssumptions.append(TrAnExpr.trSpecExprAuxAssumptions);
				preAssumptions.append(TrAnExpr.trSpecExprAuxConditions);
				axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
				TrAnExpr.closeForClause();
			}

			// [GKS]
			if(Main.options().idc) { // add is-defined conditions
				//Expr e = expr2IsDefExpr(expr);
				if(!escjava.AnnotationHandler.isTrue(expr)) {
					// 	if (false // DISABLE THIS FOR NOW 
					// 	      && expr instanceof LabelExpr && ((LabelExpr)expr).expr instanceof InstanceOfExpr) 
					// [FIXME] This is a temporary solution.
					// Filter out LabelExpr nodes with their first node being InstanceOfExpr because
					// those are the precondition of client methods that we do not want to handle.
					Expr e2idc=null;

					// [GK] Do not prune the leftmost conjunction because I realized that
					// that there are cases where the precondition is not always in 
					// the form of a conjunction.
					if (true) {
						//if (dmd.isStaticMethod() || dmd.isConstructor())
						e2idc=expr;
					} else if (dmd.isInstanceMethod()) {
						Assert.notFalse(expr.getTag() == TagConstants.AND,
								EscPrettyPrint.inst.toString(expr));
						e2idc=DefGCmd.reapLeftmostConjunct(expr);
					} else {
						Assert.notFalse(false,EscPrettyPrint.inst.toString(expr));
					}

					// Note: the expression argument to the CHECK-DEFINEDNESS
					// command is the requires clause assertion expression
					// untranslated -- i.e. not translated into the equivalent
					// GC expression form.  The translation is not later when
					// the CHECK-DEFINEDNESS command is desugared.
					if(Main.options().debug) {
						System.err.println("GK-Trace-PRE: " + 
								EscPrettyPrint.inst.toString(e2idc));
						System.err.println("\ti.e.: "+ e2idc);
					}

					Condition cond = GC.condition(TagConstants.CHKEXPRDEFINEDNESS, 
							e2idc,//TrAnExpr.trSpecExpr(e, map, null),
							e2idc.getStartLoc());
					pre.addElement(cond);
				}
			}
			// [GKE]
			Condition cond = GC.condition(TagConstants.CHKPRECONDITION, pred, loc);

			pre.addElement(cond);
		}

		addAxioms(axsToAdd, preAssumptions);

		return pre;
	}

	/** Returns a boolean Expr that characterizes when <code>expr</code> is
	 * defined. E.g. when called on an expression like "x/y" it would yield
	 * the Expr (label Zero:n.m y != 0).
	 */
	private static /*@ non_null */ Expr expr2IsDefExpr(/*@ non_null */ Expr expr) {
		// System.err.println("\texpr2IsDefExpr(" + expr + ")");
		Expr result = null;
		if(expr instanceof LabelExpr) {
			LabelExpr labelExpr =(LabelExpr)expr;
			result = expr2IsDefExpr(labelExpr.expr);
			/*
	    result = LabelExpr.make(labelExpr.getStartLoc(),
				    labelExpr.getEndLoc(), 
				    labelExpr.positive,
				    labelExpr.label,
				    result);
			 */
		} else if(expr instanceof ParenExpr){
			ParenExpr parenExpr=(ParenExpr)expr;
			result = expr2IsDefExpr(parenExpr.expr);
		} else if(expr instanceof LiteralExpr ||
				expr instanceof VariableAccess ||
				expr instanceof ThisExpr
		) {
			result = escjava.AnnotationHandler.T;
		} else if(expr instanceof InstanceOfExpr) {
			InstanceOfExpr e = (InstanceOfExpr)expr;
			result = expr2IsDefExpr(e);
		} else if(expr instanceof BinaryExpr) {
			BinaryExpr bExpr = (BinaryExpr)expr;
			Expr leftIsDefExpr = expr2IsDefExpr(bExpr.left);
			Expr rightIsDefExpr = expr2IsDefExpr(bExpr.right);
			Type t = javafe.tc.FlowInsensitiveChecks.getType(bExpr);

			switch(expr.getTag()) {
			case TagConstants.DIV:
				LiteralExpr zero = (LiteralExpr)escjava.tc.FlowInsensitiveChecks
				.setType(LiteralExpr.make(TagConstants.INTLIT, 
						new Integer(0), 
						bExpr.getStartLoc()), 
						t); // Types.intType);
						Expr ne = BinaryExpr.make(TagConstants.NE, 
								bExpr.right, 
								zero, 
								bExpr.getStartLoc());
						javafe.tc.FlowInsensitiveChecks.setType(ne, Types.booleanType);
						result = LabelExpr.make(bExpr.locOp, 
								bExpr.getEndLoc(), 
								false,
								escjava.translate.
								GC.makeFullLabel("Zero",
										bExpr.locOp,
										Location.NULL),
										ne);
						javafe.tc.FlowInsensitiveChecks.setType(result, Types.booleanType);
						break;

			case TagConstants.AND:
				Expr impPart = escjava.AnnotationHandler.implies(bExpr.left,
						rightIsDefExpr);
				result = escjava.AnnotationHandler.and(leftIsDefExpr, impPart);
				break;

			case TagConstants.OR:
				Expr notLeft = escjava.AnnotationHandler.not(bExpr.left);
				result = escjava.AnnotationHandler.implies(notLeft,rightIsDefExpr);
				break;

			default:
				result = escjava.AnnotationHandler.and(leftIsDefExpr, rightIsDefExpr);
			break;
			}
		} else if(expr instanceof CondExpr) {
			CondExpr cExpr = (CondExpr)expr;
			Expr testIsDefExpr = expr2IsDefExpr(cExpr.test);
			Expr thnIsDefExpr = expr2IsDefExpr(cExpr.thn);
			Expr elsIsDefExpr = expr2IsDefExpr(cExpr.els);

			Expr impThen = escjava.AnnotationHandler.implies(cExpr.test, 
					thnIsDefExpr);
			Expr notTest = escjava.AnnotationHandler.not(cExpr.test);
			javafe.tc.FlowInsensitiveChecks.setType(notTest, Types.booleanType);
			Expr impElse = escjava.AnnotationHandler.implies(notTest,
					elsIsDefExpr);
			result = escjava.AnnotationHandler.
			and(testIsDefExpr,
					escjava.AnnotationHandler.and(impThen, impElse));
		} else {
			if(true) throw new NullPointerException(expr.toString());
			switch(expr.getTag()) {
			default:
				if(true) throw new NullPointerException(expr.toString());
			break;
			}
		}
		// System.err.println("\texpr2IsDefExpr =" + result);
		return result == null ? escjava.AnnotationHandler.T : result;
	}

	/** Computes the postconditions, according to section 7.2.2 of ESCJ 16. */
	//@ ensures \result != null;
	private static /*@ non_null */ ConditionVec trMethodDeclPostcondition(
			/*@ non_null */DerivedMethodDecl dmd,
			/*@ non_null */Hashtable wt,
			/*@ non_null */ExprVec postAssumptions) {
		ConditionVec post = ConditionVec.make();

		//System.out.println("STARTING WITH " + postAssumptions.size());
		// type correctness of targets (including "alloc", if "alloc" is a target)
		Enumeration wtEnum = wt.keys();
		while (wtEnum.hasMoreElements()) {
			GenericVarDecl vd = (GenericVarDecl)wtEnum.nextElement();
			Expr e = TrAnExpr.targetTypeCorrect(vd, wt);
			Condition cond = GC.freeCondition(e, Location.NULL);
			post.addElement(cond);
		}

		if (dmd.isConstructor()) {
			// Free: RES != null && !isAllocated(RES, wt[[alloc]])
			Expr nonnull = GC.nary(TagConstants.REFNE, GC.resultvar, GC.nulllit);
			Expr allocated = GC.nary(TagConstants.ISALLOCATED, GC.resultvar,
					(VariableAccess)wt.get(GC.allocvar.decl));
			Expr notAllocated = GC.not(allocated);
			Condition cond = GC.freeCondition(GC.and(nonnull, notAllocated),
					Location.NULL);
			post.addElement(cond);
		}

		if (!Types.isVoidType(dmd.returnType)) {
			// Free: TypeCorrectAs[[ RES, T ]]
			addFreeTypeCorrectAs(GC.resultvar.decl, dmd.returnType, post);

			if (Utils.isPure(dmd.original) && (
					dmd.original instanceof ConstructorDecl ||
					!Utils.isAllocates(dmd.original))) {
				Expr e = TrAnExpr.resultEqualsCall(GC.resultvar.decl, dmd.original, wt);
				Condition cond = GC.freeCondition(e, Location.NULL);
				post.addElement(cond);
			}
		}

		TypeSig T = TypeSig.getSig(dmd.getContainingClass());
		if (dmd.isConstructor() && !T.isTopLevelType()) {
			// Free: RES.this$0 == this$0arg
			Expr this0 = TrAnExpr.makeVarAccess(Inner.getEnclosingInstanceField(T),
					Location.NULL);
			FormalParaDecl this0arg = Inner
			.getEnclosingInstanceArg((ConstructorDecl)dmd.original);
			Expr thisSet = GC.nary(TagConstants.REFEQ,
					GC.select(this0, GC.resultvar), TrAnExpr.makeVarAccess(this0arg,
							Location.NULL));
			Condition cond = GC.freeCondition(thisSet, Location.NULL);
			post.addElement(cond);
		}

		if (dmd.throwsSet.size() == 0) {
			// UnexpectedException: EC == ecReturn
			Expr pred = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return);
			Condition cond = GC.condition(TagConstants.CHKUNEXPECTEDEXCEPTION, pred,
					Location.NULL);
			if (NoWarn.getChkStatus(TagConstants.CHKUNEXPECTEDEXCEPTION)
					== TagConstants.CHK_AS_ASSERT) post.addElement(cond);
			cond = GC.condition(TagConstants.CHKUNEXPECTEDEXCEPTION2, pred,
					Location.NULL);
			if (NoWarn.getChkStatus(TagConstants.CHKUNEXPECTEDEXCEPTION2)
					== TagConstants.CHK_AS_ASSERT) post.addElement(cond);
		} else {
			// Free: EC == ecThrow ==>
			//          XRES != null && typeof(XRES) <: Throwable &&
			//          isAllocated(XRES, alloc)
			Expr antecedent = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_throw);
			ExprVec ev = ExprVec.make();
			// XRES != null
			Expr p = GC.nary(TagConstants.REFNE, GC.xresultvar, GC.nulllit);
			ev.addElement(p);
			// typeof(XRES) <: Throwable
			p = GC.nary(TagConstants.TYPELE, GC.nary(TagConstants.TYPEOF,
					GC.xresultvar), GC.typeExpr(Types.javaLangThrowable()));
			ev.addElement(p);
			// isAllocated(XRES, alloc)
			p = GC.nary(TagConstants.ISALLOCATED, GC.xresultvar, GC.allocvar);
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
			int num = dmd.throwsSet.size();
			Object originalnum = Utils.exceptionDecoration.get(dmd.original);
			if (originalnum != null) num = ((Integer)originalnum).intValue(); 
			for (int i = 0; i < num; i++) {
				TypeName x = dmd.throwsSet.elementAt(i);
				p = GC.nary(TagConstants.TYPELE, typeofXRES, GC.typeExpr(x));
				ev.addElement(p);
			}
			Expr pp = GC.or(normal, GC.and(except, GC.or(ev)));

			if (!Main.options().strictExceptions) {
				for (int i = num; i < dmd.throwsSet.size(); i++) {
					TypeName x = dmd.throwsSet.elementAt(i);
					p = GC.nary(TagConstants.TYPELE, typeofXRES, GC.typeExpr(x));
					ev.addElement(p);
				}
			}

			Expr eOutcomes;
			eOutcomes = GC.or(ev);

			p = GC.or(normal, GC.and(except, eOutcomes));

			// Note: This is commented out because we sometimes fabricate a 
			// throws clause where there was none before. - DRCok
			//Assert.notFalse(dmd.original.locThrowsKeyword != Location.NULL);
			cond = GC
			.condition(TagConstants.CHKUNEXPECTEDEXCEPTION, p, Location.NULL);
			if (NoWarn.getChkStatus(TagConstants.CHKUNEXPECTEDEXCEPTION)
					== TagConstants.CHK_AS_ASSERT) post.addElement(cond);
			cond = GC
			.condition(TagConstants.CHKUNEXPECTEDEXCEPTION2, pp, Location.NULL);
			if (NoWarn.getChkStatus(TagConstants.CHKUNEXPECTEDEXCEPTION2)
					== TagConstants.CHK_AS_ASSERT) post.addElement(cond);
		}

		// constructors ensure that the new object's owner field is null
		if (dmd.isConstructor()) {
			// get java.lang.Object
			TypeSig obj = Types.javaLangObject();

			FieldDecl owner = null; // make the compiler happy
			boolean found = true;
			boolean save = escjava.tc.FlowInsensitiveChecks.inAnnotation;
			try {
				escjava.tc.FlowInsensitiveChecks.inAnnotation = true;
				owner = Types.lookupField(obj, Identifier.intern("owner"), obj);
			} catch (javafe.tc.LookupException e) {
				found = false;
			} finally {
				escjava.tc.FlowInsensitiveChecks.inAnnotation = save;
			}
			// if we couldn't find the owner ghost field, there's nothing to do
			if (found) {
				VariableAccess ownerVA = TrAnExpr.makeVarAccess(owner, Location.NULL);
				Expr everything;
				Expr ownerNull = GC.nary(TagConstants.REFEQ, GC.select(ownerVA,
						GC.resultvar), GC.nulllit);
				// if there are no exceptions thrown, it is sufficient to do
				// RES.owner == null
				if (dmd.throwsSet.size() == 0)
					everything = ownerNull;
				// else we need to do (EC == ECReturn) ==> (RES.owner == null)
				else {
					Expr ret = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return);
					everything = GC.implies(ret, ownerNull);
				}
				Condition cond = GC.condition(TagConstants.CHKOWNERNULL, everything,
						Location.NULL);
				post.addElement(cond);
			}
		}

		// Finally, add declared postconditions
		// First normal postconditions
		try {
			// EC == ecReturn
			Expr ante = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return);

			Hashtable map;
			if (dmd.isConstructor()) {
				map = new Hashtable();
				map.put(GC.thisvar.decl, GC.resultvar);
			} else {
				map = null;
			}
			java.util.Set axsToAdd = new java.util.HashSet();
			ArrayList conds = new ArrayList();
			for (int i = 0; i < dmd.ensures.size(); i++) {
				try {
					ExprModifierPragma prag = dmd.ensures.elementAt(i);
					//[GKS]
					   if (Main.options().idc) {
					  	 if(Main.options().debug) {
					  		 System.err.println("GK-Trace-NPC: " + 
					  				 EscPrettyPrint.inst.toString(prag.expr));
					  		 System.err.println("\ti.e.: "+prag.expr);
					  	 }

					  	 Condition cond=GC.condition(TagConstants.CHKEXPRDEFNORMPOST,
					  			 prag.expr,prag.expr.getStartLoc());
					  	 conds.add(cond);
					   }
					   //[GKE]
					   TrAnExpr.initForClause();
					   //System.out.println("TRANEXPR");
					   //escjava.AnnotationHandler.printSpec(prag);
					   Expr pred = TrAnExpr.trSpecExpr(prag.expr, map, wt);
					   if (TrAnExpr.extraSpecs) {
					  	 postAssumptions.append(TrAnExpr.trSpecExprAuxAssumptions);
					  	 postAssumptions.append(TrAnExpr.trSpecExprAuxConditions);
					  	 axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
					  	 TrAnExpr.closeForClause();
					   }
					   pred = GC.implies(ante, pred);
					   int tag = prag.errorTag == 0 ? TagConstants.CHKPOSTCONDITION
					  		 : prag.errorTag;
					   Condition cond = GC.condition(tag, pred, prag.getStartLoc());
					   conds.add(cond);
				} catch (NotImplementedException e) {
					TrAnExpr.closeForClause();
					// If something not implemented occurs, a message has already
					// been issued (unless turned off by an option). We catch it
					// at this point and go on to the next ensures clause, only
					// skipping the clause containing the construct that is not
					// implemented.
				}
			}
			/*
      for (int i=0; i<postAssumptions.size(); i++) {
        escjava.ast.EscPrettyPrint.inst.print(System.out,0,postAssumptions.elementAt(i));
        System.out.println("");
      }
      System.out.println("ENDING WITH " + postAssumptions.size() + " " + axsToAdd.size());
			 */

			addAxioms(axsToAdd, postAssumptions);
			Iterator jj = conds.iterator();
			while (jj.hasNext()) {
				post.addElement((Condition)jj.next());
			}
		} finally {
			TrAnExpr.closeForClause();
		}
		//System.out.println("TRMETHODECLPOST T WITH " + postAssumptions.size());

		//      System.out.println("WT"); Enumeration ee = wt.keys(); while
		//      (ee.hasMoreElements()) { Object o = ee.nextElement();
		//      System.out.println("MAP: " + o + " -->> " + wt.get(o)); }

		// Then exceptional postconditions
		{
			// EC == ecThrow
			Expr ante0 = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_throw);
			// typeof(XRES)
			Expr typeofXRES = GC.nary(TagConstants.TYPEOF, GC.xresultvar);

			java.util.Set axsToAdd = new java.util.HashSet();
			ArrayList conds = new ArrayList();
			for (int i = 0; i < dmd.exsures.size(); i++) {
				try {
					// Pragma has the form: exsures (T v) E
					VarExprModifierPragma prag = dmd.exsures.elementAt(i);
					//[GKS]
					if (Main.options().idc) {
						if(Main.options().debug) {
							System.err.println("GK-Trace-EPC: " + 
									EscPrettyPrint.inst.toString(prag.expr));
							System.err.println("\ti.e.: "+prag.expr);
						}
						Condition cond=GC.condition(TagConstants.CHKEXPRDEFEXCEPOST,
								prag.expr,prag.expr.getStartLoc());
						conds.add(cond);
					}
					//[GKE]
					TrAnExpr.initForClause();
					// TrSpecExpr[[ E, {v-->XRES}, wt ]]
					Hashtable map = new Hashtable(); 
					Expr pred;
					if (prag.arg != null) {
						map.put(prag.arg, GC.xresultvar);
						pred = TrAnExpr.trSpecExpr(prag.expr, map, wt);
						// typeof(XRES) <: T
						Expr ante1 = GC.nary(TagConstants.TYPELE, typeofXRES, GC
								.typeExpr(prag.arg.type));
						//     EC == ecThrow && typeof(XRES) <: T
						// ==> TrSpecExpr[[ E, {v-->XRES}, wt ]]
						pred = GC.implies(GC.and(ante0, ante1), pred);
					} else {
						pred = TrAnExpr.trSpecExpr(prag.expr, map, wt);
						pred = GC.implies(ante0, pred);
					}
					if (TrAnExpr.extraSpecs) {
						postAssumptions.append(TrAnExpr.trSpecExprAuxAssumptions);
						postAssumptions.append(TrAnExpr.trSpecExprAuxConditions);
						axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
						TrAnExpr.closeForClause();
					}
					//int tag = prag.errorTag == 0 ? TagConstants.CHKPOSTCONDITION :
					// prag.errorTag;
					int tag = TagConstants.CHKPOSTCONDITION;
					Condition cond = GC.condition(tag, pred, prag.getStartLoc());
					conds.add(cond);
				} catch (NotImplementedException e) {
					TrAnExpr.closeForClause();
				}
			}
			addAxioms(axsToAdd, postAssumptions);
			Iterator jj = conds.iterator();
			while (jj.hasNext()) {
				post.addElement((Condition)jj.next());
			}
		}

		//System.out.println("TRMETHODECLPOST Z WITH " + postAssumptions.size());
		// Then any initially clauses (for constructors, if not a helper)

		boolean isHelper = Utils.findModifierPragma(dmd.original.pmodifiers,
				TagConstants.HELPER) != null;

		if (dmd.isConstructor() && !isHelper) {
			Hashtable map = new Hashtable();
			map.put(GC.thisvar.decl, GC.resultvar);
			TypeDeclElemVec pmods = dmd.getContainingClass().elems;
			java.util.Set axsToAdd = new java.util.HashSet();
			for (int i = 0; i < pmods.size(); ++i) {
				TypeDeclElem p = pmods.elementAt(i);
				if (!(p instanceof TypeDeclElemPragma)) continue;
				if (((TypeDeclElemPragma)p).getTag() != TagConstants.INITIALLY)
					continue;
				ExprDeclPragma prag = (ExprDeclPragma)p;
				try {
					TrAnExpr.initForClause();
					Expr pred = TrAnExpr.trSpecExpr(prag.expr, map, wt);
					if (TrAnExpr.extraSpecs) {
						postAssumptions.append(TrAnExpr.trSpecExprAuxAssumptions);
						postAssumptions.append(TrAnExpr.trSpecExprAuxConditions);
						axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
						TrAnExpr.closeForClause();
					}
					int tag = TagConstants.CHKINITIALLY;
					Condition cond = GC.condition(tag, pred, prag.getStartLoc());
					post.addElement(cond);
				} catch (NotImplementedException e) {
					TrAnExpr.closeForClause();
				}
			}
			addAxioms(axsToAdd, postAssumptions);
		}

		if (dmd.isConstructor() || isHelper) return post;
		// Then any constraint clauses (for methods)

		TypeDecl tdecl = dmd.getContainingClass();
		Set s = new javafe.util.Set();
		if (tdecl instanceof InterfaceDecl)
			s.add(tdecl);
		else {
			ClassDecl cdecl = (ClassDecl)tdecl;
			while (true) {
				post = addConstraintClauses(post, cdecl, wt, postAssumptions);
				addSuperInterfaces(cdecl, s);
				if (cdecl.superClass == null) break;
				cdecl = (ClassDecl)TypeSig.getSig(cdecl.superClass).getTypeDecl();
			}
		}
		Enumeration en = s.elements();
		while (en.hasMoreElements()) {
			InterfaceDecl ifd = (InterfaceDecl)en.nextElement();
			post = addConstraintClauses(post, ifd, wt, postAssumptions);
		}
		//System.out.println("TRMETHODECLPOST ENDING WITH " + postAssumptions.size());
		return post;
	}

	/**
	 * axsToAdd holds a Set of RepHelper - we need to add to the assumptions any
	 * axioms pertinent to the RepHelper.
	 */
	static public void addAxioms(/*@ non_null @*/ java.util.Set axsToAdd, /*@ non_null */ ExprVec assumptions) {
		java.util.Set axsDone = new java.util.HashSet();
		while (!axsToAdd.isEmpty()) {
			RepHelper o = (RepHelper)axsToAdd.iterator().next();
			axsToAdd.remove(o);
			if (!axsDone.add(o)) continue;
			Expr e = TrAnExpr.getEquivalentAxioms(o, null);
			assumptions.addElement(e);
			axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
			TrAnExpr.trSpecAuxAxiomsNeeded.clear();
		}
	}

	// FIXME - need to include inherited constraint clauses
	static public /*@ non_null */ ConditionVec addConstraintClauses(/*@ non_null */ ConditionVec post,
			/*@ non_null @*/ TypeDecl decl, 
			/*@ nullable */ Hashtable wt, 
			/*@ non_null */ ExprVec postAssumptions) {
		TypeDeclElemVec pmods = decl.elems;
		java.util.Set axsToAdd = new java.util.HashSet();
		for (int i = 0; i < pmods.size(); ++i) {
			TypeDeclElem p = pmods.elementAt(i);
			if (!(p instanceof TypeDeclElemPragma)) continue;
			if (((TypeDeclElemPragma)p).getTag() != TagConstants.CONSTRAINT)
				continue;
			ExprDeclPragma prag = (ExprDeclPragma)p;
			try {
				TrAnExpr.initForClause();
				Expr pred = TrAnExpr.trSpecExpr(prag.expr, null, wt);
				if (TrAnExpr.extraSpecs) {
					postAssumptions.append(TrAnExpr.trSpecExprAuxAssumptions);
					postAssumptions.append(TrAnExpr.trSpecExprAuxConditions);
					axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
					TrAnExpr.closeForClause();
				}
				int tag = TagConstants.CHKCONSTRAINT;
				Condition cond = GC.condition(tag, pred, prag.getStartLoc());
				post.addElement(cond);
			} catch (NotImplementedException e) {
				TrAnExpr.closeForClause();
			}
		}

		addAxioms(axsToAdd, postAssumptions);
		return post;
	}

	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------
	//-------------------------------------------------------------------------

	/** Implements ExtendSpecForCall, section 7.3 of ESCJ 16. */

	//@ ensures \result != null;
	private static /*@ non_null */ Spec extendSpecForCall(/*@ non_null */Spec spec,
			/*@ non_null */FindContributors scope, 
			/*@ nullable */ Set predictedSynTargs, /*@ nullable */ RoutineDecl callingRoutine) 
	{
		//System.out.println("EXTEND START " + spec.postAssumptions.size());
		// FIXME - I'm not sure that \old variables not in the modifies list get
		// translated here
		// I think those translations are in scope but not in spec.
		// spec.preVarMap contains the modified variables for the current routine
		// but it is the preVarMap in the initialState generated from scope that has
		// the
		// relevant mappings of variables mentioned in \old expressions

		// The set of variables modified by *this* GC call:
		Set modifiedVars = new Set(spec.preVarMap.keys());

		ParamAndGlobalVarInfo vars = null;

		boolean isConstructor = spec.dmd.isConstructor();
		InvariantInfo ii = mergeInvariants(collectInvariants(scope, spec.preVarMap));
		// FIXME - Possibly causes bloated VCs -- I haven't found an example
		// yet that needs this, but it seems it ought.
		//HashSet axs = collectInvariantsAxsToAdd;
		//ExprVec assumptions = addNewAxs(axs,null);
		//spec.preAssumptions.append(assumptions);
		//spec.postAssumptions.append(assumptions);

		for (; ii != null; ii = ii.next) {

			int tag = ii.prag.getTag();
			boolean includeInPre = true;
			boolean includeInPost = tag != TagConstants.AXIOM;

			/*
			 * Does ii mention a variable that this GC call will modify?
			 */
			Set invFV = Substitute.freeVars(ii.J);
			boolean mentionsModifiedVars = Main.options().useAllInvPostCall
			|| invFV.containsAny(modifiedVars) || spec.modifiesEverything;

			/*
			 * Does ii mention a variable that the body that is making the GC call
			 * ever modifies?
			 */
			boolean falsifiable = true;
			if (predictedSynTargs != null || spec.modifiesEverything) {
				Assert.notFalse(!Main.options().useAllInvPreBody);
				falsifiable = invFV.containsAny(predictedSynTargs);
			}

			if (ii.isStatic) {
				// static invariant

				// PRECONDITION for static invariant
				Condition cond = GC.condition(TagConstants.CHKOBJECTINVARIANT, ii.J,
						ii.prag.getStartLoc());
				if (falsifiable && includeInPre) spec.pre.addElement(cond);

				// POSTCONDITION for static invariant

				if (mentionsModifiedVars) {
					// The free variables of "J" overlap with "synTargs", so add "J"
					cond = GC.freeCondition(ii.J, ii.prag.getStartLoc());
					if (includeInPost) spec.post.addElement(cond);
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

					/*
					 * Without any optimizations, we would generate one precondition here,
					 * 
					 * p == null || !is(p, trType[[ U ]]) || TrSpecExpr[[ J, {this-->p}, {} ]]
					 * 
					 * for each parameter or static field "p", where U is the type of this
					 * in invariant J.
					 * 
					 * 
					 * Optimizations:
					 *  - First, generate no preconditions for any p such that we can
					 * prove statically that p cannot have type U at runtime.
					 *  - Second, combine all the remaining preconditions into 1
					 * universally quantified precondition:
					 * 
					 * (FORALL s :: (s == p0 || s == p1 || ...) ==> s == null || !is(s,
					 * trType[[ U ]] || TrSpecExpr[[ J, {this-->p}, {} ]] )
					 * 
					 * where "p0", "p1", ... are the parameters and static fields. If the
					 * list "p0", "p1", ... is empty, don't generate a precondition.
					 *  - Third, if all of the p_i's are "non_null", drop the disjunct "s ==
					 * null".
					 *  - Fourth, if all of the p_i's can be proved to be statically of
					 * type U, drop the disjunct "!is(s, trType[[ U ]]".
					 */

					// Build equalities & collect info on which disjuncts to include:
					boolean allAreNonnull = true;
					boolean allAreOfTypeU = true;
					ExprVec alternatives = ExprVec.make();

					for (ParamAndGlobalVarInfo info = vars; info != null; info = info.next) {
						if (!Types.isSubclassOf(info.U, ii.U)) {
							// p may not always/never hold an object of type U (ii.U)
							if (!Types.isSubclassOf(ii.U, info.U))
								// p can never hold an object of type U (ii.U)
								continue;
							allAreOfTypeU = false;
						}

						Expr eq = GC.nary(TagConstants.REFEQ, ii.s, TrAnExpr.makeVarAccess(
								info.vdecl, Location.NULL));
						alternatives.addElement(eq);
						//if (! info.isNonnull) // FIXME
						allAreNonnull = false;
					}

					/*
					 * -noOutCalls changes this to check *in addition* all non-null
					 * allocated objects of dynamic type U *except* objectToBeConstructed:
					 */
					if (Main.options().noOutCalls) {
						// isAllocated(ii.s, alloc [in pre-state])
						Expr isAllocated = GC.nary(TagConstants.ISALLOCATED, ii.s,
								GC.allocvar);
						Expr notEq = GC.nary(TagConstants.REFNE, ii.s, GC.objectTBCvar);

						alternatives.addElement(GC.and(isAllocated, notEq));
						allAreNonnull = false;
						allAreOfTypeU = false;
					}

					// build precondition if any alternatives:
					if (alternatives.size() != 0) {
						Expr ante = GC.or(alternatives);
						Expr cons = ii.J;

						ExprVec disjuncts = ExprVec.make();
						if (!allAreNonnull)
							disjuncts.addElement(GC.nary(TagConstants.REFEQ, ii.s,
									GC.nulllit));
						if (!allAreOfTypeU)
							disjuncts.addElement(GC.not(GC.nary(TagConstants.IS, ii.s,
									TrAnExpr.trType(ii.U))));
						if (disjuncts.size() != 0) {
							disjuncts.addElement(cons);
							cons = GC.or(disjuncts);
						}

						Expr quant = GC.forall(ii.sdecl, GC.implies(ante, cons));
						Condition cond = GC.condition(TagConstants.CHKOBJECTINVARIANT,
								quant, ii.prag.getStartLoc());
						if (includeInPre) spec.pre.addElement(cond);
					}
				}

				// POSTCONDITION for instance invariant

				if (mentionsModifiedVars && includeInPost) {
					// TypeCorrectNoallocAs[[ s, U ]] && s != null
					ExprVec ev = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
							null, true, null, false);
					ExprVec nopats = ev.copy();

					Expr p = TrAnExpr.trSpecExpr(ii.prag.expr, TrAnExpr.union(
							spec.preVarMap, ii.map), null);
					if (spec.modifiesEverything)
						collectFields(p, spec.postconditionLocations);
					if (includeInPre) ev.addElement(p);

					Expr ante = GC.and(ev);
					Expr impl = GC.implies(ante, ii.J);

					spec.post.addElement(GC.freeCondition(GC.forall(ii.sdecl, impl,
							nopats), ii.prag.getStartLoc()));
				}
			}
		}

		// FIXME - there are helper constructors for which we need to do this block, but
		// we don't do it for the this call or super call at the beginning of the constructor body
		// so for now we won't do it for any constructors
		if ((callingRoutine instanceof ConstructorDecl) && !Helper.isHelper(spec.dmd.original) && !isConstructor) {
			RoutineDecl calledRoutine = spec.dmd.original;
			TypeDecl td = calledRoutine.parent;
			TypeDeclElemVec elems = td.elems;
			for (int i=0; i<elems.size(); ++i) {
				TypeDeclElem tde = elems.elementAt(i);
				if (tde instanceof FieldDecl) {
					FieldDecl fd = (FieldDecl)tde;
					SimpleModifierPragma p = NonNullPragma(fd);
					if (Modifiers.isStatic(fd.modifiers)) {
						// We don't check static fields since those are required to be initialized at the end of static init blocks
						continue;
					}
					if (Modifiers.isStatic(calledRoutine.modifiers) &&
							!Modifiers.isStatic(fd.modifiers)) {
						// Non-static fields are not relevant to static methods
						continue;
					}
					//System.out.println("FIELD " + (fd).id + " NONNULL? " + (p!=null));
					if (p!=null) {
						VariableAccess v = TrAnExpr.makeVarAccess(fd,p.getStartLoc());
						Expr e = v;
						if (!Modifiers.isStatic(fd.modifiers)) e = GC.select(e, GC.thisvar);
						e = GC.nary(TagConstants.REFNE, e, GC.nulllit);
						e = LabelExpr.make(p.getStartLoc(), 
								p.getStartLoc(), // FIXME - annotation locatino or call location
								false,
								escjava.translate.
								GC.makeLabel("Pre",
										p.getStartLoc(),Location.NULL),
										e); 
						Condition cond = GC.condition(TagConstants.CHKPRECONDITION,
								e, p.getStartLoc());
						spec.pre.addElement(cond);
					}
				}
			}
		}
		//System.out.println("EXTEND END " + spec.postAssumptions.size());
		return spec;
	}

	/** Implements ExtendSpecForBody, section 7.4 of ESCJ 16. */

	//@ ensures \result != null;
	private static /*@ non_null */ Spec extendSpecForBody(/*@ non_null */Spec spec,
			/*@ non_null */FindContributors scope,
			/*@ non_null */Set synTargs) 
	{

		TypeDecl td = spec.dmd.getContainingClass();
		boolean isConstructor = spec.dmd.isConstructor();

		// Add NonNullInit checks
		if (isConstructor && !spec.dmd.isConstructorThatCallsSibling()) {
			// first check fields in first-inherited interfaces
			ClassDecl cd = (ClassDecl)td;
			Enumeration inheritedInterfaces = getFirstInheritedInterfaces(cd);
			while (inheritedInterfaces.hasMoreElements()) {
				TypeDecl tdSuperInterface = (TypeDecl)inheritedInterfaces.nextElement();
				nonNullInitChecks(tdSuperInterface, spec.post);
			}
			// then check fields in the current class
			nonNullInitChecks(td, spec.post);
		}

		InvariantInfo ii = mergeInvariants(collectInvariants(scope, spec.preVarMap));
		// FIXME - Possibly causing bloated VCs
		HashSet axsToAdd = collectInvariantsAxsToAdd;
		ExprVec assumptions = addNewAxs(axsToAdd, null);
		spec.preAssumptions.append(assumptions);
		spec.postAssumptions.append(assumptions);

		for (; ii != null; ii = ii.next) {
			int tag = ii.prag.getTag();
			addInvariantBody(ii, spec, synTargs);
		}

		ExprVec axioms = collectAxioms(scope);

		for (int i = 0; i < axioms.size(); i++) {
			spec.pre.addElement(GC.freeCondition(axioms.elementAt(i), Location.NULL));
		}

		// FIXME - possibly causing bloated VCs
		for (int i = 0; i < axioms.size(); i++) {
			spec.postAssumptions.addElement(axioms.elementAt(i));
		}

		return spec;
	}

	/**
	 * Add to <code>post</code> all NonNullInit checks for non_null instance
	 * fields and instance ghost fields declared in <code>td</code>.
	 */
	private static void nonNullInitChecks(/*@ non_null */TypeDecl td,
			/*@ non_null */ConditionVec post) {
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

					Expr left = GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return);

					VariableAccess f = TrAnExpr.makeVarAccess(fd, Location.NULL);
					Expr right = GC.nary(TagConstants.REFNE, GC.select(f, GC.resultvar),
							GC.nulllit);
					Expr pred = GC.implies(left, right);
					Condition cond = GC.condition(TagConstants.CHKNONNULLINIT, pred, smp
							.getStartLoc());
					post.addElement(cond);
				}
			}
		}
	}

	//@ ensures \result != null && \result.elementType == \type(InterfaceDecl);
	static public /*@ non_null */ Enumeration getFirstInheritedInterfaces(
			/*@non_null */ClassDecl cd) {
		Set interfaces = new Set();
		addSuperInterfaces(cd, interfaces);
		if (interfaces.size() != 0 && cd.superClass != null) {
			TypeDecl tdSuper = TypeSig.getSig(cd.superClass).getTypeDecl();
			Set superClassInterfaces = new Set();
			addSuperInterfaces(tdSuper, superClassInterfaces);
			Enumeration superClassesInterfaces = superClassInterfaces.elements();
			while (superClassesInterfaces.hasMoreElements()) {
				interfaces.remove(superClassesInterfaces.nextElement());
			}
		}
		return interfaces.elements();
	}

	//@ requires set.elementType == \type(InterfaceDecl);
	private static void addSuperInterfaces(/*@ non_null */TypeDecl td,
			/*@ non_null */Set set) {
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

	/**
	 * Extend <code>spec</code>, in a way appropriate for checking the body of
	 * a method or constructor, to account for invariant <code>ii.J</code>
	 * declared in class <code>ii.U</code>. The body to be checked has
	 * syntactic targets <code>synTargs</code>.
	 */

	private static void addInvariantBody(/*@ non_null */InvariantInfo ii,
			/*@ non_null */Spec spec,
			/*@ non_null */Set synTargs) {

		Set invFV = Substitute.freeVars(ii.J);

		/*
		 * Include invariant in post only if intersection of vars of invariant and
		 * vars modified in the method body is nonempty.
		 */
		// Also include it if we are dealing with a constructor, since then
		// the invariant might never have been established.
		boolean isConstructor = spec.dmd.isConstructor();
		boolean includeInPre = true;
		boolean includeInPostOrig = true;
		boolean includeInPost = includeInPostOrig
		&& (Main.options().useAllInvPostBody || invFV.containsAny(synTargs));

		if (ii.isStatic) {
			// static invariant

			Condition cond = GC.freeCondition(ii.J, ii.prag.getStartLoc());

			if (includeInPre) spec.pre.addElement(cond);

			if (includeInPost) {
				cond = GC.condition(TagConstants.CHKOBJECTINVARIANT, ii.J, ii.prag
						.getStartLoc());
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
				//   * U is a subtype of T, and
				//      either U is not T or the constructor does not call a sibling
				// or
				//   * U is an interface, and
				//        + m calls sibling constructor and U is not a
				//           superinterface of T
				//        or
				//        + m does not call sibling constructor and U is not a
				//           superinterface of a proper superclass of T
				//   (FORALL s :: TypeCorrectNoallocAs[[ s, U ]] && s != null &&
				//                s != objectToBeConstructed
				//            ==> J)
				//
				// In either case, NOPATS is the same as the antecedent.
				ExprVec ante = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
						null, true, null, false);
				if (spec.dmd.isConstructor()) {
					TypeSig tU = ii.U; // Type that contains the invariant ii
					TypeSig tT = TypeSig.getSig(spec.dmd.getContainingClass());
					boolean includeAntecedent = false;
					//if (Types.isSubclassOf(tU, tT) || Types.isSubclassOf(tT, tU)) {
					if (Types.isSubclassOf(tU, tT)) {
						if (!Types.isSameType(tU, tT)
								|| !spec.dmd.isConstructorThatCallsSibling()) {
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
				if (includeInPre) spec.pre.addElement(cond);
			}

			// Do the postcondition

			// Include the invariant as a checked postcondition if its free variables
			// overlap with the syntactic targets of the body (as indicated by the
			// current value of "includeInPost"), or if the routine is a constructor
			// (that does not call a sibling) of some class "T" and the invariant is
			// declared in "T" or in one of "T"'s first-inherited interfaces.
			if (!includeInPost && spec.dmd.isConstructor()
					&& !spec.dmd.isConstructorThatCallsSibling()) {
				TypeSig tU = ii.U;
				ClassDecl cd = (ClassDecl)spec.dmd.getContainingClass();
				TypeSig tT = TypeSig.getSig(cd);
				if (Types.isSubclassOf(tT, tU)
						&& (cd.superClass == null || !Types.isSubclassOf(TypeSig
								.getSig(cd.superClass), tU))) {
					// "U" is "T" or a first-inherited interface of "T"
					includeInPost = true;
				}
			}

			if (includeInPost && includeInPostOrig) {
				// TypeCorrectAs[[ s, U ]] && s != null
				ExprVec ante = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
						null, true, null, true);

				if (spec.dmd.isConstructor()) {
					TypeSig tU = ii.U;
					TypeSig tT = TypeSig.getSig(spec.dmd.getContainingClass());
					if (!Types.isSubclassOf(tT, tU)) {
						// "m" is a constructor, and "U" is not a superclass or
						// superinterface of "T"
						// Add to antecedent the conjunct: s != this
						ante.addElement(GC.nary(TagConstants.REFNE, ii.s, GC.thisvar));
					} else if (Types.isSameType(tU, tT) && spec.dmd.throwsSet.size() != 0) {
						// "m" is a constructor, and "U" is "T", and throws set is nonempty
						// Add to antecedent the conjunct: (EC == ecReturn || s != this)
						ante.addElement(GC.or(GC.nary(TagConstants.ANYEQ, GC.ecvar,
								GC.ec_return), GC.nary(TagConstants.REFNE, ii.s, GC.thisvar)));
					}
				}
				Expr body = GC.implies(GC.and(ante), ii.J);
				Expr quant = GC.forall(ii.sdecl, body);
				Condition cond = GC.condition(TagConstants.CHKOBJECTINVARIANT, quant,
						ii.prag.getStartLoc());
				spec.post.addElement(cond);
				if (Info.on) {
					Info.out("[addInvariantBody: Including body-post-invariant at "
							+ Location.toString(ii.prag.getStartLoc()) + "]");
				}
			} else {
				if (Info.on) {
					Info.out("[addInvariantBody: Skipping body-post-invariant at "
							+ Location.toString(ii.prag.getStartLoc()) + "]");
				}
			}
		}
	}

	/** Collects the axioms in <code>scope</code>. */

	//@ ensures \result != null;
	private static /*@ non_null */ ExprVec collectAxioms(/*@ non_null */FindContributors scope) {

		ExprVec r = ExprVec.make();

		TrAnExpr.initForClause();
		for (Enumeration typeSigs = scope.typeSigs(); typeSigs.hasMoreElements();) {

			TypeDecl td = ((javafe.tc.TypeSig)typeSigs.nextElement()).getTypeDecl();

			for (int i = 0; i < td.elems.size(); i++) {
				TypeDeclElem tde = td.elems.elementAt(i);
				if (tde.getTag() == TagConstants.AXIOM) {
					ExprDeclPragma axiom = (ExprDeclPragma)tde;
					if (!Main.options().filterInvariants
							|| exprIsVisible(scope.originType, axiom.expr)) {
						r.addElement(TrAnExpr.trSpecExpr(axiom.expr, null, null));
					}
				}
			}

			TypeDeclElemVec tv = (TypeDeclElemVec)Utils.representsDecoration.get(td);
			for (int i = 0; i < tv.size(); ++i) {
				TypeDeclElem tde = tv.elementAt(i);
				NamedExprDeclPragma p = (NamedExprDeclPragma)tde;
				FieldDecl fd = ((FieldAccess)p.target).decl;
				Expr e = TrAnExpr.getRepresentsAxiom(p, null);
				r.addElement(e);
			}
		}

		TrAnExpr.closeForClause();
		return r;
	}

	/**
	 * Gets the represents clauses for a model field fd as seen from a type
	 * declaration td; fd may be declared in td or in a supertype of td.
	 * If td is null, then all represents clauses are returned, in any loaded class.
	 */
	static public /*@ nullable */ TypeDeclElemVec getRepresentsClauses(/*@ nullable */ TypeDecl td, /*@ non_null */ FieldDecl fd) {
		if (td == null) return (TypeDeclElemVec)Utils.representsDecoration.get(fd);
		TypeDeclElemVec mpv = (TypeDeclElemVec)Utils.representsDecoration.get(td);
		TypeDeclElemVec n = TypeDeclElemVec.make(mpv.size());
		for (int i = 0; i < mpv.size(); ++i) {
			TypeDeclElem m = mpv.elementAt(i);
			if (!(m instanceof NamedExprDeclPragma)) continue;
			NamedExprDeclPragma nem = (NamedExprDeclPragma)m;
			if (((FieldAccess)nem.target).decl == fd) n.addElement(m);
		}
		return n;
	}

	/** Collects the invariants in <code>scope</code>. */

	private static /*@ nullable */ HashSet collectInvariantsAxsToAdd;

	private static /*@ nullable */ InvariantInfo collectInvariants(
			/*@ non_null */FindContributors scope, /*@ nullable */ Hashtable premap) {
		collectInvariantsAxsToAdd = null;
		InvariantInfo ii = null;
		InvariantInfo iiPrev = null;

		Enumeration invariants = scope.invariants();
		try {
			TrAnExpr.initForClause();
			while (invariants.hasMoreElements()) {
				ExprDeclPragma ep = (ExprDeclPragma)invariants.nextElement();
				Expr J = ep.expr;

				boolean Jvisible = !Main.options().filterInvariants
				|| exprIsVisible(scope.originType, J);

				// System.out.print( (Jvisible?"Visible":"Invisible")+": ");
				// javafe.ast.PrettyPrint.inst.print(System.out, 0, J );
				// System.out.println();

				if (Jvisible) {
					//System.out.println("COLLECTING INVARIANT " +
					// Location.toString(ep.getStartLoc()));

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
					// in "J" itself. (These yield different results in the
					// unusual case that "J" mentioned "this" in a subexpression
					// "E.g", where "g" is a static field.)
					//  First, build the map "{this-->s}" for a new "s".

					LocalVarDecl sdecl = UniqName.newBoundThis();
					VariableAccess s = TrAnExpr.makeVarAccess(sdecl, Location.NULL);
					Hashtable map = new Hashtable();
					map.put(GC.thisvar.decl, s);

					int cReplacementsBefore = TrAnExpr.getReplacementCount();

					/*
					 * Unlike any body we may be translating, for these translations
					 * this's type is that of the type that contains the invariant J.
					 */
					Type savedType = GC.thisvar.decl.type;
					GC.thisvar.decl.type = TypeSig.getSig(ep.getParent());

					invinfo.J = TrAnExpr.trSpecExpr(J, map, null);
					if (TrAnExpr.trSpecExprAuxConditions.size() != 0) {
						// Use andx here, because the op needs to be and in preconditions
						// and
						// implies in postconditions
						invinfo.J = GC.andx(GC.nary(TagConstants.BOOLAND,
								TrAnExpr.trSpecExprAuxConditions), invinfo.J);
						TrAnExpr.trSpecExprAuxConditions = ExprVec.make();
					}
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
					//System.out.println("COLLECTING INVARIANT-END " +
					// Location.toString(ep.getStartLoc()));
				}
			}
			// FIXME - Possibly causing bloated VCs
			collectInvariantsAxsToAdd = new java.util.HashSet();
			collectInvariantsAxsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
			java.util.Set axsToAdd = new java.util.HashSet();
			//axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
			java.util.Set axsDone = new java.util.HashSet();
			/*
			 * while (false && ! axsToAdd.isEmpty()) { // FIXME - keep this off ???
			 * RepHelper o = (RepHelper)axsToAdd.iterator().next();
			 * axsToAdd.remove(o); if (!axsDone.add(o)) continue; Expr e =
			 * TrAnExpr.getEquivalentAxioms(o,null); //axsToAdd.addAll(
			 * TrAnExpr.trSpecAuxAxiomsNeeded); // Add a new node at the end of "ii"
			 * InvariantInfo invinfo = new InvariantInfo(); invinfo.J = e;
			 * invinfo.prag = ExprDeclPragma.make(TagConstants.AXIOM, e, 0,
			 * Location.NULL); // FIXME invinfo.U = TypeSig.getSig(ep.parent); if
			 * (iiPrev == null) ii = invinfo; else iiPrev.next = invinfo; iiPrev =
			 * invinfo; if (true ) { //|| cReplacementsBefore ==
			 * TrAnExpr.getReplacementCount()) // FIXME // static invariant
			 * invinfo.isStatic = true; invinfo.sdecl = null; invinfo.s = null;
			 * invinfo.map = null; } else { invinfo.isStatic = false; // FIXME
			 * invinfo.sdecl = sdecl; // FIXME invinfo.s = s; // FIXME invinfo.map =
			 * map; } }
			 */
		} finally {
			TrAnExpr.closeForClause();
		}

		return ii;
	}

	/**
	 * Collects the parameters of <code>spec.args</code> and the static fields
	 * in <code>scope</code>, whose types are class types.
	 */

	private static /*@ nullable */ ParamAndGlobalVarInfo collectParamsAndGlobals(
			/*@ non_null */Spec spec,
			/*@ non_null */FindContributors scope) {
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
				isNonnull = false; // FIXME
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
		Enumeration fields = scope.fields();
		while (fields.hasMoreElements()) {
			FieldDecl fd = (FieldDecl)fields.nextElement();

			TypeSig classSig = Types.toClassTypeSig(fd.type);

			if (classSig != null && Modifiers.isStatic(fd.modifiers)) {
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

	//@ ensures \result != null;
	private static /*@ non_null */ ExprVec addNewAxs(/*@ non_null @*/ HashSet axsToAdd, /*@ nullable */ ExprVec assumptions) {
		if (assumptions == null) assumptions = ExprVec.make();
		java.util.Set axsDone = new java.util.HashSet();
		while (!axsToAdd.isEmpty()) {
			RepHelper o = (RepHelper)axsToAdd.iterator().next();
			axsToAdd.remove(o);
			if (!axsDone.add(o)) continue;
			Expr e = TrAnExpr.getEquivalentAxioms(o, null);
			assumptions.addElement(e);
			axsToAdd.addAll(TrAnExpr.trSpecAuxAxiomsNeeded);
			TrAnExpr.trSpecAuxAxiomsNeeded.clear();
		}
		return assumptions;

	}

	/** Shaves a GC designator. */

	private static /*@ nullable */ VariableAccess shave(/*@ non_null */Expr e) {
		switch (e.getTag()) {
		case TagConstants.VARIABLEACCESS:
			return (VariableAccess)e;

		case TagConstants.SELECT:
			return shave(((NaryExpr)e).exprs.elementAt(0));

		default:
			Assert.fail("Unexpected case: " + " "
					+ TagConstants.toString(e.getTag()) + " " + e + " "
					+ Location.toString(e.getStartLoc()));
		return null;
		}
	}

	/**
	 * Creates and returns a new map that is <code>map</code> restricted to the
	 * domain <code>e</code>. Assumes that every element in <code>e</code> is
	 * in the domain of <code>map</code>.
	 */

	//+@ requires map.keyType == \type(GenericVarDecl);
	//+@ requires map.elementType == \type(VariableAccess);
	//+@ requires e.elementType == \type(GenericVarDecl);
	//@ ensures \result != null;
	static /*@ nullable */ Hashtable restrict(/*@ non_null @*/ Hashtable map, 
			/*@ non_null @*/ Enumeration e) {
		Hashtable r = new Hashtable();
		while (e.hasMoreElements()) {
			GenericVarDecl vd = (GenericVarDecl)e.nextElement();
			VariableAccess variant = (VariableAccess)map.get(vd);
			Assert.notNull(variant);
			r.put(vd, variant);
		}
		return r;
	}

	/**
	 * Converts a GenericVarDecl enumeration to a mapping from those variables to
	 * new Variableaccesses.
	 */

	//@ requires e.elementType == \type(GenericVarDecl);
	//@ ensures \result != null;
	static /*@ nullable */ Hashtable makeSubst(/*@ non_null */Enumeration e,
			/*@ non_null */String postfix) {
		Hashtable r = new Hashtable();
		while (e.hasMoreElements()) {
			GenericVarDecl vd = (GenericVarDecl)e.nextElement();
			VariableAccess variant = createVarVariant(vd, postfix);
			r.put(vd, variant);
		}
		return r;
	}

	//@ ensures \result != null;
	static /*@ nullable */ Hashtable makeSubst(/*@ non_null */FormalParaDeclVec args,
			/*@ non_null */String postfix) {
		Hashtable r = new Hashtable();
		for (int i = 0; i < args.size(); i++) {
			GenericVarDecl vd = args.elementAt(i);
			VariableAccess variant = createVarVariant(vd, postfix);
			r.put(vd, variant);
		}
		return r;
	}

	/**
	 * * Given a GenericVarDecl named "x@old", returns a VariableAccess to a *
	 * fresh LocalVarDecl named "x@ <postfix>". * * This handles ESCJ 23b case 13.
	 */
	//@ ensures \result != null;
	static /*@ nullable */ VariableAccess createVarVariant(/*@ non_null */GenericVarDecl vd,
			/*@ non_null */String postfix) 
	{
		String name = vd.id.toString();
		if (name.indexOf('@') != -1) name = name.substring(0, name.indexOf('@'));

		Identifier id = Identifier.intern(name + "@" + postfix);
		LocalVarDecl ld = LocalVarDecl.make(vd.modifiers, vd.pmodifiers, id,
				vd.type, vd.locId, null, Location.NULL);
		ld.source = vd;
		VariableAccess va = VariableAccess.make(id, vd.locId, ld);

		return va;
	}

	/**
	 * Adds to <code>cv</code> a condition stating that <code>vd</code> has
	 * type <code>type</code>.
	 */

	private static void addFreeTypeCorrectAs(/*@ non_null */ GenericVarDecl vd, /*@ non_null */ Type type,
			/*@ non_null @*/ ConditionVec cv) {
		Expr e = TrAnExpr.typeCorrectAs(vd, type);
		Condition cond = GC.freeCondition(e, Location.NULL);
		cv.addElement(cond);
	}

	/**
	 * Returns a command that first does an <code>assume</code> for each
	 * precondition in <code>spec</code>, then does <code>body</code>, then
	 * checks the postconditions of <code>spec</code>, and finally checks the
	 * modifies constraints implied by <code>spec</code>.
	 */

	public static /*@ non_null */ GuardedCmd surroundBodyBySpec(/*@ non_null */ GuardedCmd body, 
			/*@ non_null @*/ Spec spec,
			/*@ nullable */ FindContributors scope, 
			/*@ nullable */ Set syntargets, 
			/*@ nullable */ Hashtable premap,
			int locEndCurlyBrace,
			boolean nobody) 
	{
		StackVector code = new StackVector();
		code.push();
		addAssumptions(spec.preAssumptions, code);
		assumeConditions(spec.pre, code);
		code.addElement(body);
		addAssumptions(spec.postAssumptions, code);
		checkConditions(spec.post, locEndCurlyBrace, code, nobody);
		checkModifiesConstraints(spec, scope, syntargets, premap, locEndCurlyBrace,
				code);

		return GC.seq(GuardedCmdVec.popFromStackVector(code));
	}

	/**
	 * Appends <code>code</code> with an <code>assume C</code> command for
	 * every condition <code>C</code> in <code>cv</code>.
	 */

	public static void addAssumptions(/*@ non_null @*/ ExprVec ev,
			/*@ non_null @*/ StackVector code) 
	{
		for (int i = 0; i < ev.size(); i++) {
			Expr e = ev.elementAt(i);
			code.addElement(GC.assume(e));
			//code.addElement(GC.check(e.getStartLoc(),TagConstants.CHKCONSISTENT,e,e.getStartLoc(),null));
		}
	}

	public static void assumeConditions(/*@ non_null @*/ ConditionVec cv, 
			/*@ non_null @*/ StackVector code)
	{
		for (int i = 0; i < cv.size(); i++) {
			Condition cond = cv.elementAt(i);
			//[GKS]
			if(Main.options().idc && cond.label == TagConstants.CHKEXPRDEFINEDNESS) {
				DefGCmd oDefGCs = new DefGCmd();
				if(Main.options().debug) {
					System.err.println("\tAbout to trAndGen:" +
							EscPrettyPrint.inst.toString(cond.pred));
					System.err.println("\tI.e.:" + cond.pred);
				}
				oDefGCs.trAndGen(cond.pred);
				GuardedCmd gc=oDefGCs.popFromCode();
				code.addElement(gc);
				//[GKE]
			} else {
				code.addElement(GC.assumeAnnotation(cond.locPragmaDecl, cond.pred));
			}
		}
	}

	/**
	 * Appends <code>code</code> with an <code>check loc: C</code> command for
	 * every condition <code>C</code> in <code>cv</code>.
	 */

	private static void checkConditions(/*@ non_null @*/ ConditionVec cv, 
			int loc, 
			/*@ non_null */ StackVector code,
			boolean nobody)
	{
		// [GKS]
		Vector idcGCs=new Vector();
		if (Main.options().idc) {
			// First pass through the ConditionVec so that we build the IDC GCs.
			// We expect postconditions to be in A /\ P1 /\ ... /\ Pk ==> Q form or
			// to be just in the Q form.
			// For every postcondition:
			//   Break the implication and store the antecedent and consequents.
			//   All the antecedents should be identical for the same specification 
			//     block - if not throw an exception.
			//   Conjoin all Qs per antecedent and pass that for IDC checking.
			//   Iterate the GC gerenated by the IDC checking of the conjoin 
			//   postconditions and append the antecedent to all ASSERT commands.
			// Lets see how this will work out :>
			LinkedHashMap mapAnte2AntePost=new LinkedHashMap();
			for (int i=0;i<cv.size();i++) {
				Condition cond = cv.elementAt(i);
				if (Main.options().idc &&
						(cond.label == TagConstants.CHKEXPRDEFNORMPOST ||
								cond.label == TagConstants.CHKEXPRDEFEXCEPOST)) {
					if (Main.options().debug) {
						System.err.println("Processing-Post: " +
								EscPrettyPrint.inst.toString(cond.pred));
						//System.err.println("\tI.e.:" + cond.pred);
					}
					DefGCmd.processPostcondition(cond.label, mapAnte2AntePost, cond.pred);
				}
			}
			java.util.Set s=mapAnte2AntePost.entrySet();
			Iterator is=s.iterator();
			while (is.hasNext()) {
				Map.Entry me = (Map.Entry)is.next();
				Object [] antePost=(Object[])me.getValue();
				String k=(String)me.getKey();
				Expr ante=(Expr)antePost[0];
				Expr conjPosts=(Expr)antePost[1];
				DefGCmd oDefGCs = new DefGCmd();
				// Pass the conjoint postconditions for IDC.
				oDefGCs.trAndGen(conjPosts);
				// Obtain the GCs.
				GuardedCmd idcGC=oDefGCs.popFromCode();
				// Translate the antecedent into a GCExpr.
				Expr gcAnte=TrAnExpr.trSpecExpr(ante, oDefGCs.minHMap4Tr(), null);
				if (k.startsWith(String.valueOf(TagConstants.CHKEXPRDEFNORMPOST)))
					gcAnte=GC.and(GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return),
							gcAnte);
				else if (k.startsWith(String.valueOf(TagConstants.CHKEXPRDEFEXCEPOST)))
					gcAnte=GC.and(GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_throw),
							gcAnte);
				else
					Assert.fail("Unknown key prefix: " + k);
				// Add the antedents in the proper 
				oDefGCs.morfAsserts(idcGC,gcAnte);
				idcGCs.addElement(idcGC);
			}
		}
		//[GKE]
		// Flag to denote whether the IDC GCs were added.
		boolean idcGCAdded=false;
		for (int i = 0; i < cv.size(); i++) {
			Condition cond = cv.elementAt(i);

			//[GKS]
			if (Main.options().idc &&
					(cond.label == TagConstants.CHKEXPRDEFNORMPOST ||
							cond.label == TagConstants.CHKEXPRDEFEXCEPOST)) {
				if (!idcGCAdded) {
					if (Main.options().debug) {
						System.err.println("IDCGCs.SIZE: "+idcGCs.size());
					}
					for (int j=0;j<idcGCs.size();j++) {
						GuardedCmd idcGC=(GuardedCmd)idcGCs.elementAt(j);
						code.addElement(idcGC);
					}
					idcGCAdded=true;
				}
				continue;
			}
			// Add postconditions in the usual way.
			if (!nobody)
			{
				if (cond.label == TagConstants.CHKUNEXPECTEDEXCEPTION2) continue;
				Translate.setop(cond.pred);
				// if the condition is an object invariant, send its guarded command
				// translation as auxiliary info to GC.check
				if (cond.label == TagConstants.CHKOBJECTINVARIANT)
					code.addElement(GC.check(loc, cond, cond.pred));
				else
					code.addElement(GC.check(loc, cond));
			}
		}
	}

	private static void checkModifiesConstraints(/*@ nullable */ Spec spec,
			/*@ nullable */ FindContributors scope, /*@ nullable */ Set syntargets, /*@ nullable */ Hashtable premap, int loc,
			/*@ nullable */ StackVector code) {
		// TBW
	}

	private static /*@ nullable */ InvariantInfo mergeInvariants(/*@ nullable */ InvariantInfo ii) {
		if (!Main.options().mergeInv) return ii;

		// Combined static/dynamic invariants, indexed by TypeSIg
		Hashtable staticInvs = new Hashtable();
		Hashtable dynInvs = new Hashtable();

		while (ii != null) {

			Hashtable table = ii.isStatic ? staticInvs : dynInvs;

			InvariantInfo old = (InvariantInfo)table.get(ii.U);

			if (old == null) {

				table.put(ii.U, ii);

			} else {

				// Add ii to old
				old.J = GC.and(old.J, ii.isStatic ? ii.J : GC.subst(ii.sdecl, old.s,
						ii.J));
			}

			ii = ii.next;
		}

		// Now pull out merged invariants from tables
		Hashtable[] tables = {staticInvs, dynInvs};
		ii = null;

		for (int i = 0; i < 2; i++) {
			Hashtable table = tables[i];

			for (Enumeration e = table.elements(); e.hasMoreElements();) {
				InvariantInfo t = (InvariantInfo)e.nextElement();
				t.next = ii;
				ii = t;
			}
		}

		return ii;
	}

	static boolean exprIsVisible(/*@ non_null @*/ TypeSig originType, 
			/*@ non_null @*/ Expr e) {

		switch (e.getTag()) {

		case TagConstants.FIELDACCESS: {
			FieldAccess fa = (FieldAccess)e;
			FieldDecl decl = fa.decl;

			if (fa.od instanceof ExprObjectDesignator
					&& !exprIsVisible(originType, ((ExprObjectDesignator)fa.od).expr))
				return false;

			if (decl.parent == null) {
				// for array.length "field", there is no parent
				return true;
			} else if (Utils.findModifierPragma(decl, TagConstants.SPEC_PUBLIC) != null) {
				return true;
			} else {
				TypeSig sigDecl = TypeSig.getSig(decl.parent);
				return TypeCheck.inst.canAccess(originType, sigDecl, decl.modifiers,
						decl.pmodifiers);
			}
		}

		default: {
			// Recurse over all children
			for (int i = 0; i < e.childCount(); i++) {
				Object o = e.childAt(i);
				if (o instanceof Expr) {
					if (!exprIsVisible(originType, (Expr)o)) return false;
				}
			}

			return true;
		}
		}
	}

	static public void collectFields(/*@ non_null @*/ Expr e, /*@ non_null @*/ java.util.Set s) {
		// FIXME - have to avoid collecting bound variables of quantifiers
		switch (e.getTag()) {
		case TagConstants.PRE:
			return;
		case TagConstants.VARIABLEACCESS:
			VariableAccess va = (VariableAccess)e;
			if (va.decl instanceof FormalParaDecl) {
				//System.out.println("PARA " + ((VariableAccess)e).decl );
				return;
			}
			if (va.id.toString().endsWith("@pre")) return;
			//System.out.println("COLLECTED-VA " + e);
			s.add(e);
			break;
		default:
		}

		// Recurse over all children
		for (int i = 0; i < e.childCount(); i++) {
			Object o = e.childAt(i);
			if (o instanceof Expr) collectFields((Expr)o, s);
		}

	}

	/*****************************************************************************
	 * * Handling NON_NULL: * *
	 ****************************************************************************/

	/**
	 * Decorates <code>GenericVarDecl</code>'s to point to NonNullPragmas
	 * (SimpleModifierPragma's).
	 */
	//@ invariant nonnullDecoration != null;
	// FIXME @ invariant nonnullDecoration.decorationType ==
	// FIXME @ \type(SimpleModifierPragma);
	/*@ spec_public */
	private static /*@ non_null */ ASTDecoration nonnullDecoration = new ASTDecoration(
	"nonnullDecoration");

	/**
	 * Mark v as non_null because of non_null pragma nonnull.
	 * Precondition: nonnull is a NON_NULL pragma. (Used to implement inheritence of
	 * non_null's.)
	 */
	private static void setNonNullPragma(/*@ non_null */ GenericVarDecl v,
			/*@ non_null */ SimpleModifierPragma nonnull) {
		nonnullDecoration.set(v, nonnull);
	}

	/**
	 * Has a particular declaration been declared non_null? If so, 
	 * return the non_null pragma responsible. Otherwise, return null.
	 * <ul>
	 * <li>Precondition: if the declaration is a formal parameter, then the
	 * spec of it's routine has already been computed.
	 * <li>WARNING: this is the only authorized way to determine this
	 * information. Do <em>not</em> attempt to search for NON_NULL modifiers 
	 * directly.
	 * </ul>
	 * (This is needed to handle inherited NON_NULL's * properly.)
	 */
	public static /*@nullable*/ SimpleModifierPragma NonNullPragma(/*@ non_null */ GenericVarDecl v) {
		// In JML, non_null pragmas do not inherit!
		return (SimpleModifierPragma)Utils.findModifierPragma(v,
				TagConstants.NON_NULL);
	}

	/**
	 * Returns non-null if the formal parameter is declared non-null in some
	 * overridden declaration of the method.
	 */
	public static /*@ nullable */ SimpleModifierPragma superNonNullPragma(/*@ non_null */ GenericVarDecl v) {
		// First check for a mark:
		SimpleModifierPragma mark = (SimpleModifierPragma)nonnullDecoration.get(v);
		return mark;
	}

}

/**
 * * This class is used by <code>collectInvariants</code> and its callers, *
 * <code>extendSpecForCall</code> and <code>extendSpecForBody</code>.
 */

class InvariantInfo {

	/*@ nullable */ ExprDeclPragma prag;

	/*@ nullable */ TypeSig U; // "TypeSig" of class or interface that contains "prag"

	boolean isStatic; // "true" if pragma declares a static invariant

	/*@ nullable */ LocalVarDecl sdecl; // "null" if "isStatic"; otherwise, a dummy variable

	/*@ nullable */ VariableAccess s; // "null" if "isStatic"; otherwise, a variable access

	// of "sdecl"
	/*@ nullable */ Hashtable map; // "{this-->s}"

	/*@ nullable */ Expr J; // translated expression, with substitution

	// "{this-->s}" if not "isStatic"
	/*@ nullable */ InvariantInfo next; // for linking "InvariantInfo" nodes
}

/**
 * This class is used by <code>collectParamsAndGlobalVars</code> and its *
 * caller, <code>extendSpecForCall</code>.
 */

class ParamAndGlobalVarInfo {

	/*@ nullable */ GenericVarDecl vdecl;

	/*@ nullable */ TypeSig U;

	boolean isNonnull;

	/*@ nullable */ ParamAndGlobalVarInfo next;
}

