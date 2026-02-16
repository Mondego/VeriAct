/* Copyright Hewlett-Packard, 2002 */

package escjava.backpred;


import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.ast.FieldDeclVec;		// compiler bug workaround
import escjava.ast.ExprDeclPragmaVec;		// compiler bug workaround

import javafe.tc.*;
import javafe.util.*;

import escjava.translate.GC;
import escjava.translate.GetSpec;
import escjava.translate.Inner;
import escjava.translate.Translate;
import escjava.translate.Helper;



/**
 ** This class is responsible for determining the contributors to a
 ** given TypeSig or RoutineDecl. <p>
 **
 ** The contributors are divided into a set of TypeSigs, a set of
 ** Invariants, and a set of Fields.
 **/

public class FindContributors {

    /***************************************************
     *                                                 *
     * Public interface:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Generate the contributors for a given TypeSig.  This may result
     ** in errors being reported and TypeSigs being type checked. <p>
     **
     ** The originType is taken to be that Typesig.<p>
     **
     ** Precondition: T must have been typechecked.<p>
     **
     ** Note: in the process of locating contributors,
     ** FindContributors may type check additional types leading
     ** possibly to type errors.  If such error(s) occur, a fatal
     ** error is reproted.
     **/
    public FindContributors(TypeSig T) {
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
     ** Generate the contributors for a given RoutineDecl.  This may
     ** result in errors being reported and TypeSigs being type checked.<p>
     **
     ** The originType is taken to be the type declaring that RoutineDecl.<p>
     **
     ** Precondition: the type declaring rd must have been typechecked.<p>
     **
     ** Note: in the process of locating contributors,
     ** FindContributors may type check additional types leading
     ** possibly to type errors.  If such error(s) occur, a fatal
     ** error is reproted.
     **/
    public FindContributors(RoutineDecl rd) {
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
     ** Our origin type; used to determine visibility and accessibility
     ** when needed.
     **/
    public TypeSig originType;


    /**
     ** Enumerate the TypeSig contributors
     **/
    public Enumeration typeSigs() {
	return contributorTypes.elements();
    }

    /**
     ** Enumerate the invariant contributors
     **/
    public Enumeration invariants() {
	Enumeration e = contributorInvariants.elements();
	Set invs = new Set();

	while (e.hasMoreElements()) {
	    TypeDeclElemPragma td = (TypeDeclElemPragma) e.nextElement();
	    if (td.getTag() == TagConstants.INVARIANT) {
		invs.add(td);
	    }
	}
	return invs.elements();
    }

    /**
     ** Enumerate the global invariant contributors
     **/
    public Enumeration globalInvariants() {
	Enumeration e = contributorInvariants.elements();
	Set globalInvs = new Set();

	while (e.hasMoreElements()) {
	    TypeDeclElemPragma td = (TypeDeclElemPragma) e.nextElement();
	    /*
	      // to check why some global invs dont get in
		ExprDeclPragma edp = (ExprDeclPragma) td;
		System.out.println("++++++++++ GI1 BEGINS +++++++++++++");
		System.out.println(PrettyPrint.inst.toString(edp.expr));
		System.out.println("++++++++++ GI1 ENDS +++++++++++++");
	    */
	    if (td.getTag() == TagConstants.GLOBAL_INVARIANT) {
		globalInvs.add(td);
	    }
	}
	return globalInvs.elements();
    }

    /**
     ** Enumerate the oti contributors
     **/
    public Enumeration otis() {
	Enumeration e = contributorInvariants.elements();
	Set otis = new Set();

	while (e.hasMoreElements()) {
	    TypeDeclElemPragma td = (TypeDeclElemPragma) e.nextElement();
	    if (td.getTag() == TagConstants.OTI) {
		otis.add(td);
	    }
	}
	return otis.elements();
    }

    /**
     ** Enumerate the ti contributors
     **/
    public Enumeration tis() {
	Enumeration e = contributorInvariants.elements();
	Set tis = new Set();

	while (e.hasMoreElements()) {
	    TypeDeclElemPragma td = (TypeDeclElemPragma) e.nextElement();
	    if (td.getTag() == TagConstants.TI) {
		tis.add(td);
	    }
	}
	return tis.elements();
    }

    /**
     ** Enumerate the global invariants and the (object)
     ** invariants
     **/
    public Enumeration globalAndObjInvariants() {
	Enumeration e = contributorInvariants.elements();
	Set globAndObjInvs = new Set();

	while (e.hasMoreElements()) {
	    TypeDeclElemPragma td = (TypeDeclElemPragma) e.nextElement();
	    if (td.getTag() == TagConstants.GLOBAL_INVARIANT
		|| td.getTag() == TagConstants.INVARIANT) {
		globAndObjInvs.add(td);
	    }
	}
	return globAndObjInvs.elements();
    }


    /**
     ** Enumerate the field contributors
     **/
    public Enumeration fields() {
	return contributorFields.elements();
    }


    /***************************************************
     *                                                 *
     * Closure code:				       *
     *                                                 *
     ***************************************************/

    /** The set of routines visited so far.
     **/

    private Set visitedRoutines = new Set();

    /**
     ** The set of TypeSigs we've determined to be contributors so far. <p>
     **
     ** Invariant: This set is closed under taking supertypes
     **/
    private Set contributorTypes = new Set();

    /**
     ** The set of invariants (elementType ExprDeclPragmas) we've
     ** determined to be contributors so far. <p>
     **
     ** Closure Property: walk(-) has been called on each of
     ** these invariants.
     **/
    /** 
     ** Global invariants, otis and tis are also stored here
     **/
    private Set contributorInvariants = new Set();

    /**
     ** The set of fields (elementType FieldDecl) we've determined to be
     ** contributors so far. <p>
     **
     ** Closure Property:
     **
     **    all invariants J that are declared in types in
     **    contributorTypes and "mention the field" f for f a field in
     **    contributorFields are members of contributorInvariants
     **/
    private Set contributorFields = new Set();



    /**
     ** A mapping from fields (FieldDecls) to possible invariant
     ** contributors (ExprDeclPragmaVec). <p>
     **
     ** (An invariant is a possible contributor if it is declared in a
     ** type of contributorTypes.)
     **
     ** Invariant:
     **
     **	   if (f,J) in fieldToPossible then the invariant J "mentions
     **    the field" f and J is a possible invariant contributor
     **
     ** Closure property:
     **
     **    if J is possible invariant contributor,
     **    J not in contributorInvariants, and J "mentions the field" f,
     **    then (f,J) is in fieldToPossible.
     **
     **/
    Hashtable fieldToPossible = new Hashtable();



    /**
     ** Add the TypeSigs mentioned explicitly in a given Type to
     ** contributorTypes, maintaining all the closure properties. <p>
     **
     ** Precondition: T has been resolved.<p>
     **/
    //@ requires T!=null
    private void addType(Type T) {
	// TypeName case:
	if (T instanceof TypeName) {
	    T = TypeSig.getSig((TypeName)T);
	    Assert.precondition(T!=null);
	}

	// ArrayType case:
	if (T instanceof ArrayType) {
	    addType(((ArrayType)T).elemType);
	    return;
	}

	// MapType case:
	if (T instanceof MapType) {
	    addType(((MapType)T).elemType);
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
	    int tag = tde.getTag();

	    if (tag == TagConstants.INVARIANT ||
		tag == TagConstants.GLOBAL_INVARIANT ||
		tag == TagConstants.OTI ||
		tag == TagConstants.TI)
		addPossibleInvariant((ExprDeclPragma)tde);
	}


	/*
	 * Hack: for 1.1, add an inner classes' this$0 pointer when
	 * that class is added as a type, rather than trying to keep
	 * track of everywhere it might be introduced.
	 */
	if (!sig.isTopLevelType()) {
	    FieldDecl thisptr = Inner.getEnclosingInstanceField(sig);

	    backedgeToGenericVarDecl(thisptr, null, true);
	}
    }


    /**
     ** Add a given field to contributorFields, maintaining all the
     ** closure properties. <p>
     **/
    //@ requires fd!=null
    private void addField(FieldDecl fd) {
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
     ** Add the mapping (fd, J) to fieldToPossible.
     **
     ** Precondition: J is a possible invariant contributor, J "mentions
     ** the field" fd.
     **/
    private void addPossibleMentions(FieldDecl fd, ExprDeclPragma J) {
	ExprDeclPragmaVec range = (ExprDeclPragmaVec)fieldToPossible.get(fd);
	if (range==null) {
	    range = ExprDeclPragmaVec.make();
	    fieldToPossible.put(fd, range);
	}

	range.addElement(J);
    }


    /**
     ** Add a possible invariant contributor to either fieldToPossible
     ** or contributorInvariants as approperiate, maintaining all
     ** closure properties. <p>
     **
     ** Precondition: J has been type checked.
     **/
    //@ requires J!=null
    private void addPossibleInvariant(ExprDeclPragma J) {
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
     ** Add a given invariant to contributorInvarints, maintaining all
     ** closure properties. <p>
     **
     ** Precondition: J has been type checked.
     **/
    // OTIs, TIs and global_invariants are also added into
    // contributorInvariants
    //@ requires J!=null
    private void addInvariant(ExprDeclPragma J) {
	if (contributorInvariants.contains(J))
	    return;
	
	contributorInvariants.add(J);

	walk(J);
    }


    /***************************************************
     *                                                 *
     * Walking ASTNodes:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Walks a given ASTNode, adding all the types it "mentions" via
     ** addType and adding all the fields it "mentions" via
     ** addField. <p>
     **
     ** Precondition: N has been typechecked.
     **
     ** WARNING: N is assumed to have no free local or parameter
     ** varables.
     **/
    private void walk(ASTNode N) {
	walk(N, null, true);
    }


    /**
     ** Returns the set of fields that a given invariant mentions. <p>
     **
     ** Precondition: J must be an invariant, J must be typechecked.
     **/
    private FieldDeclVec fieldsInvariantMentions(ExprDeclPragma J) {
	// We may cache this function later...

	FieldDeclVec result = FieldDeclVec.make();
	walk(J, result, false);
	return result;
    }


    /**
     ** Walks a given ASTNode, finding all the types it "mentions" and
     ** all the fields it "mentions". <p>
     **
     ** If fields is null, the fields mentioned are added via addField;
     ** otherwise, they are added to fields directly.<p>
     **
     ** If addTypes is true, the types mentioned are added via addType.<p>
     **
     ** Precondition: N has been typechecked.
     **/
    private void walk(ASTNode N, FieldDeclVec fields, boolean addTypes) {
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
	if (N instanceof VariableAccess)
	    backedgeToGenericVarDecl(((VariableAccess)N).decl,
				     fields, addTypes);
	if (N instanceof FieldAccess)
	    backedgeToGenericVarDecl(((FieldAccess)N).decl,
				     fields, addTypes);

	if (N instanceof ConstructorInvocation) {
	    ConstructorInvocation ci = (ConstructorInvocation) N;
	    int inline = Translate.inlineDecoration.get(ci) != null ? 2 :
                         isNonRecursiveHelperInvocation(ci.decl) ? 1 : 0;
	    backedgeToRoutineDecl(ci.decl, fields, addTypes, inline);
	}
	if (N instanceof NewInstanceExpr) {
	    NewInstanceExpr ni = (NewInstanceExpr) N;
	    int inline = Translate.inlineDecoration.get(ni) != null ? 2 :
                         isNonRecursiveHelperInvocation(ni.decl) ? 1 : 0;
	    backedgeToRoutineDecl(ni.decl,fields, addTypes, inline);
	}
	if (N instanceof MethodInvocation) {
	    MethodInvocation mi = (MethodInvocation) N;
	    int inline = Translate.inlineDecoration.get(mi) != null ? 2 :
                         isNonRecursiveHelperInvocation(mi.decl) ? 1 : 0;
	    backedgeToRoutineDecl(mi.decl, fields, addTypes, inline);
	}

	/*
	 * Add references not explicitly in Java code or from backedges:
	 */
	if (N instanceof RoutineDecl) {
	    // Get references in our derived spec:
	    backedgeToRoutineDecl((RoutineDecl)N, fields, addTypes, 0);

	    // Add implicit references if N is a ConstructorDecl:
	    if (N instanceof ConstructorDecl)
		addImplicitConstructorRefs((ConstructorDecl)N,
					   fields, addTypes);
	}

	
	/*
	 * Recurse to subnodes (this automatically ignores backedges):
	 *
	 * We intentionally skip TypeDecls so that we stay in the same type.
	 */
	int size = N.childCount();
	for (int i=0; i<size; i++) {
	    Object child = N.childAt(i);
	    if (child instanceof ASTNode && !(child instanceof TypeDecl))
		walk((ASTNode)child, fields, addTypes);
	}
    }

  /** Returns <code>true</code> if and only if <code>r</code> is a helper
    * routine that has not been visited by this <code>FindContributor</code>
    * object.
    **/

  private boolean isNonRecursiveHelperInvocation(/*@ non_null */ RoutineDecl r) {
    return Helper.isHelper(r) && !visitedRoutines.contains(r);
  }

    /**
     ** Add implicit references from a ConstructorDecl that do not
     ** appear in Java code or via backedges as per walk(,,).
     **
     ** Precondition: the type declaring cd has been typechecked.
     **/
    private void addImplicitConstructorRefs(ConstructorDecl cd,
					    FieldDeclVec fields,
					    boolean addTypes) {
	/*
	 * Walk the initialization code derived from the same class as
	 * the constructor:
	 */
	TypeDecl td = cd.parent;
	walkInstanceInitialier(td, fields, addTypes);

	/*
	 * For all superinterfaces that the constructor's type
	 * implements that are not already implemented by its
	 * superclass(es), walk the initialization code derived from
	 * them:
	 */
	Enumeration FII = GetSpec.getFirstInheritedInterfaces((ClassDecl)td);
	while (FII.hasMoreElements()) {
	    walkInstanceInitialier((TypeDecl)FII.nextElement(), fields,
				   addTypes);
	}
    }


    /**
     ** Walk the implicit instance initializer code for a given
     ** TypeDecl, excluding any field initializations of
     ** superinterface fields. <p>
     **
     ** E.g., f_1 = 0; ...; f_3 = <initializer exp>; ... ; <instance
     ** initializer block>...
     ** 
     ** This is the code that constructors of that type implicitly
     ** start with after their super/sibling call modulo the
     ** initialization of superinterface fields.
     **
     ** See addImplicitConstructorRefs for the full version w/o the
     ** exclusion.
     **
     **
     ** Addition: This now also pulls in all invariants in the given
     **           TypeDecl, regardless of what they mention.  This is
     **           to avoid user surprise; see also Vanilla.java in
     **           test58.
     **
     **
     ** Precondition: the TypeDecl has been typechecked.
     **/
    private void walkInstanceInitialier(TypeDecl td,
					FieldDeclVec fields,
					boolean addTypes) {
	for (int i = 0; i < td.elems.size(); i++) {
	    TypeDeclElem tde = td.elems.elementAt(i);
	    int tag = tde.getTag();

	    if (tag == TagConstants.INVARIANT || 
		tag == TagConstants.GLOBAL_INVARIANT ||
		tag == TagConstants.OTI ||
		tag == TagConstants.TI) {
		addInvariant((ExprDeclPragma)tde);
	    }

	    if (tde instanceof GhostDeclPragma)
		tde = ((GhostDeclPragma)tde).decl;
	    if (tde.getTag() == TagConstants.FIELDDECL
		  && !Modifiers.isStatic(((FieldDecl)tde).modifiers)) {
		FieldDecl fd = (FieldDecl)tde;

		// walk "fd := (fd.init==null ? <zero-equivalent> : fd.init)":
		backedgeToGenericVarDecl(fd, fields, addTypes);
		walk(fd.init, fields, addTypes);

	    } else if (tde.getTag() == TagConstants.INITBLOCK
		  && !Modifiers.isStatic(((InitBlock)tde).modifiers)) {
		// walk any instance initializer blocks found:
		walk((InitBlock)tde, fields, addTypes);

	    }
	}
    }


    /**
     ** Calculate the fields and types "mentioned" by a backedge to a
     ** GenericVarDecl and then add them as per walk(,,).
     **/
    private void backedgeToGenericVarDecl(GenericVarDecl decl,
				          FieldDeclVec fields,
				          boolean addTypes) {
	// The length field of arraytypes is never considered "mentioned":
	if (decl==javafe.tc.Types.lengthFieldDecl)
	    return;

	if (decl instanceof FieldDecl) {
	    FieldDecl fd = (FieldDecl)decl;
	    typecheck(TypeSig.getSig(fd.parent));

	    // The range and domain types of fd are "mentioned":
	    addType(fd.type);
	    if (fd.parent!=null)	// "length" field has no parent...
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
	    if (fields!=null)
		fields.addElement(fd);
	    else
		addField(fd);

	    /*
	     * We need to walk the "spec" part of fd as well to handle
	     * defined_if and the like:
	     */
	    if (fd.pmodifiers!=null) {
		for (int i=0; i<fd.pmodifiers.size(); i++)
		    walk(fd.pmodifiers.elementAt(i), fields, addTypes);
	    }
	}

	/*
	 * Ignore local & parameter variable Decls here, as they
	 * should already have been walked over.
	 */    
    }

    /**
     ** Calculate the fields and types "mentioned" by a backedge to a
     ** RoutineDecl and then add them as per walk(,,). <p>
     **
     ** <code>inlined</code> is one of: 0 (not an inlined routine),
     ** 1 (an inlined helper routine), or 2 (a routine inlined for
     ** a reason other than being a helper).  (Why this complication,
     ** why not just use a boolean field "inlined"?  By distinguishing
     ** cases 1 and 2, one can write a nice run-time assert inside the
     ** implementation of this method.)
     **/
    //@ requires inlined == 0 || inlined == 1 || inlined == 2;
    private void backedgeToRoutineDecl(RoutineDecl rd,
				       FieldDeclVec fields,
				       boolean addTypes, 
				       int inlined) {
        Assert.notFalse(inlined != 1 || !visitedRoutines.contains(rd));
        visitedRoutines.add(rd);

	TypeSig thisType = TypeSig.getSig(rd.parent);
	typecheck(thisType);

	// Bug fix insert begins 7/27/01
	Type savedType = GC.thisvar.decl.type;
	GC.thisvar.decl.type = thisType;
	// Bug fix insert ends 7/27/01

	// Get the derived spec for rd:
	DerivedMethodDecl dmd = GetSpec.getCombinedMethodDecl(rd);
	dmd = GetSpec.filterMethodDecl(dmd, this);


	/*
	 * The types in the called routine's Java signature are
	 * "mentioned" (including the type of Constructors):
	 */
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
	    walk(dmd.requires.elementAt(i), fields, addTypes);
	if (inlined == 0) {
	  // Add modifies here mostly for safely reasons
	  for (int i=0; i<dmd.modifies.size(); i++)
	    walk(dmd.modifies.elementAt(i), fields, addTypes);
	}
	for (int i=0; i<dmd.ensures.size(); i++)
	    walk(dmd.ensures.elementAt(i), fields, addTypes);
	for (int i=0; i<dmd.exsures.size(); i++)
	    walk(dmd.exsures.elementAt(i), fields, addTypes);
	if (dmd.performs != null) {
	    walk(dmd.performs, fields, addTypes);
	}

	if (inlined != 0) {
	    walk(rd.body, fields, addTypes);
	}

	// Bug fix insert begins 7/27/01
	GC.thisvar.decl.type = savedType;
	// Bug fix insert ends 7/27/01

    }


    /***************************************************
     *                                                 *
     * Utility routines:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Make sure a given TypeSig has been type checked, type checking
     ** it if necessary. <p>
     **
     ** Throws a fatal error if a type error occurs while checking sig.
     **/
    void typecheck(TypeSig sig) {
	int errorCount = ErrorSet.errors;

	sig.typecheck();
	if (errorCount == ErrorSet.errors)
	    return;

	ErrorSet.fatal("A type error has occured at an unexpected point;"
		       + " unable to continue processing");
    }
}
