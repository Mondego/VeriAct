/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.backpred;

import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.util.LinkedList;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.ast.Modifiers;
import escjava.ast.ExprDeclPragmaVec;		// compiler bug workaround

import javafe.tc.*;
import javafe.util.*;

import escjava.translate.GC;
import escjava.translate.GetSpec;
import escjava.translate.Inner;
import escjava.translate.Translate;
import escjava.translate.Helper;

/**
 * This class is responsible for determining the contributors to a
 * given TypeSig or RoutineDecl.
 *
 * <p> The contributors are divided into a set of TypeSigs, a set of
 * Invariants, and a set of Fields.
 */

public class FindContributors
{
    // Public interface

    /**
     * Generate the contributors for a given TypeSig.  This may result
     * in errors being reported and TypeSigs being type checked. <p>
     *
     * The originType is taken to be that Typesig.<p>
     *
     * Precondition: T must have been typechecked.<p>
     *
     * Note: in the process of locating contributors,
     * FindContributors may type check additional types leading
     * possibly to type errors.  If such error(s) occur, a fatal
     * error is reproted.
     */
    public FindContributors(/*@ non_null @*/ TypeSig T) {
	originType = T;
	addType(T);

	walk(T.getTypeDecl());
	fieldToPossible = null;		// conserve space

	if (Info.on) {
	    Info.out("[[contributors: "
		+ contributorTypes.size() + " types "
		+ contributorInvariants.size() + " invariants "
		+ contributorFields.size() + " fields ]]");
	}
    }

    /**
     * Generate the contributors for a given RoutineDecl.  This may
     * result in errors being reported and TypeSigs being type checked.<p>
     *
     * The originType is taken to be the type declaring that RoutineDecl.<p>
     *
     * Precondition: the type declaring rd must have been typechecked.<p>
     *
     * Note: in the process of locating contributors,
     * FindContributors may type check additional types leading
     * possibly to type errors.  If such error(s) occur, a fatal
     * error is reproted.
     */
    public FindContributors(/*@ non_null @*/ RoutineDecl rd) {
	originType = TypeSig.getSig(rd.parent);
	addType(originType);

	walk(rd);
	fieldToPossible = null;		// conserve space

	if (Info.on) {
	    Info.out("[[local contributors: "
		+ contributorTypes.size() + " types "
		+ contributorInvariants.size() + " invariants "
		+ contributorFields.size() + " fields ]]");
	}
    }


    /**
     * Our origin type; used to determine visibility and accessibility
     * when needed.
     */
    public /*@ non_null */ TypeSig originType;


    /**
     * Enumerate the TypeSig contributors
     */
    public /*@ non_null */ Enumeration typeSigs() {
	return contributorTypes.elements();
    }

    /**
     * Enumerate the invariant contributors
     */
    public /*@ non_null */ Enumeration invariants() {
	return contributorInvariants.elements();
    }

    /**
     * Enumerate the field contributors
     */
    public /*@ non_null */ Enumeration fields() {
	return contributorFields.elements();
    }


    // Closure code

    /** The set of routines visited so far.
     */

    private /*@ non_null @*/ Set visitedRoutines = new Set();

    /**
     * The set of TypeSigs we've determined to be contributors so far. <p>
     *
     * Invariant: This set is closed under taking supertypes
     */
    private /*@ non_null @*/ Set contributorTypes = new Set();

    /**
     * The set of invariants (elementType ExprDeclPragmas) we've
     * determined to be contributors so far. <p>
     *
     * Closure Property: walk(-) has been called on each of
     * these invariants.
     */
    private /*@ non_null @*/ Set contributorInvariants = new Set();

    /**
     * The set of fields (elementType FieldDecl) we've determined to be
     * contributors so far. <p>
     *
     * Closure Property:
     *
     *    all invariants J that are declared in types in
     *    contributorTypes and "mention the field" f for f a field in
     *    contributorFields are members of contributorInvariants
     */
    private /*@ non_null @*/ Set contributorFields = new Set();

    public int preFieldMode = 0;
    public /*@ non_null @*/ Set preFields = new Set();

    /**
     * A mapping from fields (FieldDecls) to possible invariant
     * contributors (ExprDeclPragmaVec). <p>
     *
     * (An invariant is a possible contributor if it is declared in a
     * type of contributorTypes.)
     *
     * Invariant:
     *
     *	   if (f,J) in fieldToPossible then the invariant J "mentions
     *    the field" f and J is a possible invariant contributor
     *
     * Closure property:
     *
     *    if J is possible invariant contributor,
     *    J not in contributorInvariants, and J "mentions the field" f,
     *    then (f,J) is in fieldToPossible.
     *
     */
    // can be null
    /*@ nullable */ Hashtable fieldToPossible = new Hashtable();



    /**
     * Add the TypeSigs mentioned explicitly in a given Type to
     * contributorTypes, maintaining all the closure properties. <p>
     *
     * Precondition: T has been resolved.<p>
     */
    //@ requires T != null;
    private void addType(/*@ non_null */ Type T) {
	// TypeName case:
	if (T instanceof TypeName) {
	    T = TypeSig.getSig((TypeName)T);
	    Assert.precondition(T != null);
	}

	// ArrayType case:
	if (T instanceof ArrayType) {
	    addType(((ArrayType)T).elemType);
	    return;
	}

	// PrimitiveType case:
	if (T instanceof PrimitiveType)
	    return;

	/*
	 * Remaining case is TypeSig:
	 */
	TypeSig sig = (TypeSig)T;
	if (contributorTypes.contains(sig))
	    return;

	// make sure sig is typechecked:
	typecheck(sig);

	// Close over taking supertypes:
	TypeDecl td = sig.getTypeDecl();
	if (td instanceof ClassDecl) {
	    ClassDecl cd = (ClassDecl)td;
	    if (cd.superClass != null)
		addType(cd.superClass);
	}
	for (int i=0; i<td.superInterfaces.size(); i++)
	    addType(td.superInterfaces.elementAt(i));

	contributorTypes.add(sig);

	// Add sig's invariants as possible contributor invariants:
	for (int i = 0; i<td.elems.size(); i++) {
	    TypeDeclElem tde = td.elems.elementAt(i);

	    if (tde.getTag() == TagConstants.INVARIANT)
		addPossibleInvariant((ExprDeclPragma)tde);
	}


	/*
	 * Hack: for 1.1, add an inner classes' this$0 pointer when
	 * that class is added as a type, rather than trying to keep
	 * track of everywhere it might be introduced.
	 */
	if (!sig.isTopLevelType()) {
	    FieldDecl thisptr = Inner.getEnclosingInstanceField(sig);

	    backedgeToGenericVarDecl(thisptr, null, true, new LinkedList());
	}
    }


    /**
     * Add a given field to contributorFields, maintaining all the
     * closure properties. <p>
     */
    //@ requires fd != null;
    private void addField(/*@ non_null */ FieldDecl fd) {
	if (contributorFields.contains(fd))
	    return;

	contributorFields.add(fd);

	/*
         * Turn possible invariant contributors that mention fd into
	 * real ones.  We use fieldToPossible to do this quickly.
	 */
	ExprDeclPragmaVec newInvariants =
	    (ExprDeclPragmaVec)fieldToPossible.get(fd);
	if (newInvariants==null)
	    return;

	// Note: newInvariants is not modified here because we added fd to
	// contributorFields
	for (int i=0; i<newInvariants.size(); i++)
	    addInvariant(newInvariants.elementAt(i));
    }

    /**
     * Add the mapping (fd, J) to fieldToPossible.
     *
     * Precondition: J is a possible invariant contributor, J "mentions
     * the field" fd.
     */
    private void addPossibleMentions(/*@ non_null */ FieldDecl fd, /*@ non_null @*/ ExprDeclPragma J) {
	ExprDeclPragmaVec range = (ExprDeclPragmaVec)fieldToPossible.get(fd);
	if (range==null) {
	    range = ExprDeclPragmaVec.make();
	    fieldToPossible.put(fd, range);
	}

	range.addElement(J);
    }


    /**
     * Add a possible invariant contributor to either fieldToPossible
     * or contributorInvariants as approperiate, maintaining all
     * closure properties. <p>
     *
     * Precondition: J has been type checked.
     */
    //@ requires J != null;
    private void addPossibleInvariant(/*@ non_null */ ExprDeclPragma J) {
	FieldDeclVec fieldsMentioned = fieldsInvariantMentions(J);

	for (int i=0; i<fieldsMentioned.size(); i++) {
	    FieldDecl fd = fieldsMentioned.elementAt(i);

	    if (contributorFields.contains(fd)) {
		// Ah!  J is actually a real invariant contributor:
		addInvariant(J);
		return;
	    }

	    // Add (f,J) to fieldToPossible:
	    addPossibleMentions(fd, J);
	}
    }

    /**
     * Add a given invariant to contributorInvarints, maintaining all
     * closure properties. <p>
     *
     * Precondition: J has been type checked.
     */
    //@ requires J != null;
    private void addInvariant(/*@ non_null */ ExprDeclPragma J) {
	if (contributorInvariants.contains(J))
	    return;

	contributorInvariants.add(J);

	walk(J);
    }


    // Walking ASTNodes

    /**
     * Walks a given ASTNode, adding all the types it "mentions" via
     * addType and adding all the fields it "mentions" via
     * addField. <p>
     *
     * Precondition: N has been typechecked.
     *
     * WARNING: N is assumed to have no free local or parameter
     * varables.
     */
    private void walk(/*@ nullable */ ASTNode N) {
	walk(N, null, true, new LinkedList());
    }


    /**
     * Returns the set of fields that a given invariant mentions. <p>
     *
     * Precondition: J must be an invariant, J must be typechecked.
     */
    private /*@ non_null */ FieldDeclVec fieldsInvariantMentions(/*@ nullable */ ExprDeclPragma J) {
	// We may cache this function later...

	FieldDeclVec result = FieldDeclVec.make();
	walk(J, result, false, new LinkedList());
	return result;
    }


    /**
     * Walks a given ASTNode, finding all the types it "mentions" and
     * all the fields it "mentions". <p>
     *
     * If fields is null, the fields mentioned are added via addField;
     * otherwise, they are added to fields directly.<p>
     *
     * If addTypes is true, the types mentioned are added via addType.<p>
     *
     * Precondition: N has been typechecked.
     */
    private void walk(/*@ nullable */ ASTNode N, /*@ nullable */ FieldDeclVec fields, boolean addTypes,
				/*@ non_null */ LinkedList visited) {
	/*
	 * Leaf nodes:
	 */
	if (N==null)
	    return;
	if (N instanceof Type) {
	    if (addTypes)
		addType((Type)N);
	    return;
	}

	if (N instanceof LiteralExpr || N instanceof ClassLiteral) {
	    Expr lit = (Expr)N;
	    if (addTypes) {
	        addType(TypeCheck.inst.getType(lit));
	    }
	}
	
	/*
	 * Handle relevant backedges:
	 */
	if (N instanceof VariableAccess) {
	    backedgeToGenericVarDecl(((VariableAccess)N).decl,
				     fields, addTypes, visited);
	}
	if (N instanceof FieldAccess) {
	    if (preFieldMode > 0) {
		preFields.add(N);
	    }
	    backedgeToGenericVarDecl(((FieldAccess)N).decl,
				     fields, addTypes, visited);
	}

	if (N instanceof ConstructorInvocation) {
	    ConstructorInvocation ci = (ConstructorInvocation) N;
	    visited.addFirst(ci.decl);
	    int inline = Translate.inlineDecoration.get(ci) != null ? 2 :
                         isNonRecursiveHelperInvocation(ci.decl) ? 1 : 0;
	    backedgeToRoutineDecl(ci.decl, fields, addTypes, inline, visited);
	}
	if (N instanceof NewInstanceExpr) {
	    NewInstanceExpr ni = (NewInstanceExpr) N;
	    int inline = Translate.inlineDecoration.get(ni) != null ? 2 :
                         isNonRecursiveHelperInvocation(ni.decl) ? 1 : 0;
	    backedgeToRoutineDecl(ni.decl,fields, addTypes, inline, visited);
	}
	if (N instanceof MethodInvocation) {
	    MethodInvocation mi = (MethodInvocation) N;
	    int inline = Translate.inlineDecoration.get(mi) != null ? 2 :
                         isNonRecursiveHelperInvocation(mi.decl) ? 1 : 0;

	    backedgeToRoutineDecl(mi.decl, fields, addTypes, inline, visited);
	}
// FIXME - add the type from quantified expression? set comprehension expr? new array expr?
	/*
	 * Add references not explicitly in Java code or from backedges:
	 */
	if (N instanceof RoutineDecl) {
	    // Get references in our derived spec:
	    backedgeToRoutineDecl((RoutineDecl)N, fields, addTypes, 0, visited);

	    // Add implicit references if N is a ConstructorDecl:
	    if (N instanceof ConstructorDecl)
		addImplicitConstructorRefs((ConstructorDecl)N,
					   fields, addTypes, visited);
	}

	if (N.getTag() == TagConstants.PRE) {
	    ++preFieldMode;
	}


	
	/*
	 * Recurse to subnodes (this automatically ignores backedges):
	 *
	 * We intentionally skip TypeDecls so that we stay in the same type.
	 */
	try {
	    int size = N.childCount();
	    for (int i=0; i<size; i++) {
		Object child = N.childAt(i);
		if (child instanceof ASTNode && !(child instanceof TypeDecl))
		    walk((ASTNode)child, fields, addTypes, visited);
	    }
	} finally {
	    if (N.getTag() == TagConstants.PRE) {
		--preFieldMode;
	    }
	}
    }

  /** Returns <code>true</code> if and only if <code>r</code> is a helper
    * routine that has not been visited by this <code>FindContributor</code>
    * object.
    */

  private boolean isNonRecursiveHelperInvocation(/*@ non_null */ RoutineDecl r) {
    return Helper.isHelper(r) && !visitedRoutines.contains(r);
  }

    /**
     * Add implicit references from a ConstructorDecl that do not
     * appear in Java code or via backedges as per walk(,,).
     *
     * Precondition: the type declaring cd has been typechecked.
     */
    private void addImplicitConstructorRefs(/*@ non_null @*/ ConstructorDecl cd,
					    /*@ nullable */ FieldDeclVec fields,
					    boolean addTypes,
					    /*@ non_null */ LinkedList visited) {
	/*
	 * Walk the initialization code derived from the same class as
	 * the constructor:
	 */
	TypeDecl td = cd.parent;
	walkInstanceInitialier(td, fields, addTypes, visited);

	/*
	 * For all superinterfaces that the constructor's type
	 * implements that are not already implemented by its
	 * superclass(es), walk the initialization code derived from
	 * them:
	 */
	Enumeration FII = GetSpec.getFirstInheritedInterfaces((ClassDecl)td);
	while (FII.hasMoreElements()) {
	    walkInstanceInitialier((TypeDecl)FII.nextElement(), fields,
				   addTypes, visited);
	}
    }


    /**
     * Walk the implicit instance initializer code for a given
     * TypeDecl, excluding any field initializations of
     * superinterface fields. <p>
     *
     * E.g., f_1 = 0; ...; f_3 = <initializer exp>; ... ; <instance
     * initializer block>...
     * 
     * This is the code that constructors of that type implicitly
     * start with after their super/sibling call modulo the
     * initialization of superinterface fields.
     *
     * See addImplicitConstructorRefs for the full version w/o the
     * exclusion.
     *
     *
     * Addition: This now also pulls in all invariants in the given
     *           TypeDecl, regardless of what they mention.  This is
     *           to avoid user surprise; see also Vanilla.java in
     *           test58.
     *
     *
     * Precondition: the TypeDecl has been typechecked.
     */
    private void walkInstanceInitialier(/*@ non_null @*/ TypeDecl td,
					/*@ nullable */ FieldDeclVec fields,
					boolean addTypes,
					/*@ non_null */ LinkedList visited) {
	for (int i = 0; i < td.elems.size(); i++) {
	    TypeDeclElem tde = td.elems.elementAt(i);

	    if (tde.getTag() == TagConstants.INVARIANT)
		addInvariant((ExprDeclPragma)tde);

	    if (tde instanceof ModelDeclPragma)
		tde = ((ModelDeclPragma)tde).decl;
	    if (tde instanceof GhostDeclPragma)
		tde = ((GhostDeclPragma)tde).decl;
	    if (tde.getTag() == TagConstants.FIELDDECL
		  && !Modifiers.isStatic(((FieldDecl)tde).modifiers)) {
		FieldDecl fd = (FieldDecl)tde;

		// walk "fd := (fd.init==null ? <zero-equivalent> : fd.init)":
		backedgeToGenericVarDecl(fd, fields, addTypes, visited);
		walk(fd.init, fields, addTypes, visited);

	    } else if (tde.getTag() == TagConstants.INITBLOCK
		  && !Modifiers.isStatic(((InitBlock)tde).modifiers)) {
		// walk any instance initializer blocks found:
		walk((InitBlock)tde, fields, addTypes, visited);

	    }
	}
    }


    /**
     * Calculate the fields and types "mentioned" by a backedge to a
     * GenericVarDecl and then add them as per walk(,,).
     */
    private void backedgeToGenericVarDecl(/*@ non_null */ GenericVarDecl decl,
				          /*@ nullable */ FieldDeclVec fields,
				          boolean addTypes,
					  /*@ non_null */ LinkedList visited) {
	// The length field of arraytypes is never considered "mentioned":
	if (decl==javafe.tc.Types.lengthFieldDecl)
	    return;

	if (decl instanceof FieldDecl) {
	    FieldDecl fd = (FieldDecl)decl;
	    typecheck(TypeSig.getSig(fd.parent));


	    // The range and domain types of fd are "mentioned":
	    addType(fd.type);
	    if (fd.parent != null)	// "length" field has no parent...
		addType(TypeSig.getSig(fd.parent));

	    /*
             * Exit if have already processed this field; need to do
	     * this to avoid recursion in case the field's modifiers
	     * mention it.
	     */
	    if (fields==null) {
		if (contributorFields.contains(fd))
		    return;
	    } else if (fields.contains(fd))
		return;


	    // The field fd is "mentioned":
	    if (fields != null)
		fields.addElement(fd);
	    else
		addField(fd);

	    /*
	     * We need to walk the "spec" part of fd as well to handle
	     * readable_if and the like:
	     */
	    if (fd.pmodifiers != null) {
		for (int i=0; i<fd.pmodifiers.size(); i++)
		    walk(fd.pmodifiers.elementAt(i), fields, addTypes, visited);
	    }
	}

	/*
	 * Ignore local & parameter variable Decls here, as they
	 * should already have been walked over.
	 */    
    }

    /**
     * Calculate the fields and types "mentioned" by a backedge to a
     * RoutineDecl and then add them as per walk(,,). <p>
     *
     * <code>inlined</code> is one of: 0 (not an inlined routine),
     * 1 (an inlined helper routine), or 2 (a routine inlined for
     * a reason other than being a helper).  (Why this complication,
     * why not just use a boolean field "inlined"?  By distinguishing
     * cases 1 and 2, one can write a nice run-time assert inside the
     * implementation of this method.)
     */
    //@ requires inlined == 0 || inlined == 1 || inlined == 2;
    private void backedgeToRoutineDecl(/*@ nullable */ RoutineDecl rd,
				       /*@ nullable */ FieldDeclVec fields,
				       boolean addTypes, 
				       int inlined, /*@ non_null */ LinkedList visited) {
	if (rd == null) return; // FIXME - this happens with some NewInstanceExpr
	// FIXME - remove references to visited
	//if (visited.contains(rd)) return;
	//visited.addFirst(rd);
	if (inlined==0 && visitedRoutines.contains(rd)) return;
        //Assert.notFalse(inlined != 1 || !visitedRoutines.contains(rd));
        visitedRoutines.add(rd);

	TypeSig thisType = TypeSig.getSig(rd.parent);
	typecheck(thisType);

	/*
	 * Temporarily set GC.thisvar's type to thisType; we need to
	 * do this because GetSpec.getCombinedMethodDecl() returns
	 * GC.thisvar as the first argument of instance methods.
	 *
	 * If we don't do this, we may add the wrong type when we add
	 * the types of the routine's arguments below.
	 */
	Type savedType = GC.thisvar.decl.type;
	GC.thisvar.decl.type = thisType;

	// Get the derived spec for rd:
	DerivedMethodDecl dmd = GetSpec.getCombinedMethodDecl(rd);
	dmd = GetSpec.filterMethodDecl(dmd, this);


	/*
	 * The types in the called routine's Java signature are
	 * "mentioned" (including the type of Constructors): */
	if (addTypes) {
	    // Add args here mostly for safely reasons
	    for (int i=0; i<dmd.args.size(); i++)
		addType(dmd.args.elementAt(i).type);
	    for (int i=0; i<dmd.throwsSet.size(); i++)
		addType(dmd.throwsSet.elementAt(i));
	    addType(dmd.returnType);
	}

	// We also need to walk the routine's spec:
	for (int i=0; i<dmd.requires.size(); i++)
	    walk(dmd.requires.elementAt(i), fields, addTypes, visited);
	if (inlined == 0) {
	  // Add modifies here mostly for safely reasons
	  for (int i=0; i<dmd.modifies.size(); i++)
	    walk(dmd.modifies.elementAt(i), fields, addTypes, visited);
	}
	for (int i=0; i<dmd.ensures.size(); i++)
	    walk(dmd.ensures.elementAt(i), fields, addTypes, visited);
	for (int i=0; i<dmd.exsures.size(); i++)
	    walk(dmd.exsures.elementAt(i), fields, addTypes, visited);
	
	if (inlined != 0) {
	    walk(rd.body, fields, addTypes, visited);
	}

	GC.thisvar.decl.type = savedType;
	//visited.removeFirst();
    }


    // Utility routines

    /**
     * Make sure a given TypeSig has been type checked, type checking
     * it if necessary. <p>
     *
     * Throws a fatal error if a type error occurs while checking sig.
     */
    void typecheck(/*@ non_null @*/ TypeSig sig) {
	int errorCount = ErrorSet.errors;

	sig.typecheck();
	if (errorCount == ErrorSet.errors)
	    return;

	ErrorSet.fatal("A type error has occurred at an unexpected point;"
		       + " unable to continue processing");
    }
}
