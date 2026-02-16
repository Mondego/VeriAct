/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;


import javafe.ast.*;
import javafe.util.*;


/**
 * EnvForCUs are used to create an Env for a CompilationUnit.
 */

public class EnvForCU extends Env {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * Our CompilationUnit.
     */
    protected /*@ non_null @*/ CompilationUnit CU;


    /**
     * Create an environment for a CompilationUnit.
     */
    public EnvForCU(/*@ non_null @*/ CompilationUnit CU) {
	this.CU = CU;
    }


    /***************************************************
     *                                                 *
     * Current/enclosing instances I:		       *
     *                                                 *
     **************************************************/


    /**
     * Is there a current instance in scope? <p>
     *
     * E.g., is "this" (or "<enclosing class>.this") legal here? <p>
     *
     * This is also refered to as "are we in a static context?".  The
     * legality of super also depends on this result. <p>
     *
     * The legality of C.this, C != <enclosing class> is different; see 
     * canAccessInstance(-).
     */
    public boolean isStaticContext() { return true; }


    /**
     * Return the intermost class enclosing the code that is checked
     * in this environment. <p>
     *
     * May return null if there is no enclosing class (aka, for
     * environments for CompilationUnits). <p>
     *
     * If isStaticContext() returns true, then this is the type of "this".
     */
    public TypeSig getEnclosingClass() { return null; }


    /**
     * If there is an enclosing instance in scope, then return the
     * (exact) type of the innermost such instance. <p>
     *
     * Note: this is considered a current instance, not an enclosing
     * instance, even inside its methods.
     */
    public TypeSig getEnclosingInstance() { return null; }


    /**
     * Returns a new Env that acts the same as us, except that its
     * current instance (if any) is not accessible. <p>
     *
     * Note: this routine is somewhat inefficient and should be
     * avoided unless an unknown environment needs to be coerced in
     * this way. <p>
     */
    public /*@non_null*/Env asStaticContext() { return this; }


    /***************************************************
     *                                                 *
     * Simple names:				       *
     *                                                 *
     **************************************************/

    /**
     * Locate the lexically innermost field or local variable
     * declaration. <p>
     *
     * Let d be the lexically innermost field or local variable
     * declaration (including formals) of id (if any such declaration
     * exists).  Then this routine returns: <p>
     *
     *    d (a LocalVarDecl or FormalParaDecl) if d is a local
     *                                            variable declaration
     *
     *    the class C that lexically encloses us and contains the
     *    (inherited) field d if d is a field declaration
     *
     *    null if d does not exist
     *
     * Note: inherited fields are considered to lexically enclose the
     * code of their subclasses.  We give the class containing the
     * field instead of the field itself to postpone dealing with
     * multiple fields named id visible in the same class.<p>
     *
     * In the field case, id disambiguates to C[.this].id.<p>
     */
    public ASTNode locateFieldOrLocal(/*@non_null*/Identifier id) {
	// CompilationUnits have no fields or local variable declarations:
	return null;
    }

    public boolean isDuplicate(/*@non_null*/Identifier id) {
	return false;
    }

    /**
     * Attempt to lookup a simple TypeName in this environment to get
     * the TypeSig it denotes.  Returns null if no such type
     * exists.<p>
     *
     * This routine does not check that the resulting type (if any)
     * is actually accessable. <p>
     *
     * If id is ambiguous, then if loc != Location.NULL then a fatal
     * error is reported at that location via ErrorSet else one of
     * its possible meanings is returned.<p>
     */
    public TypeSig lookupSimpleTypeName(TypeSig caller, /*@non_null*/Identifier id, int loc) {
	/*
	 * First, look at types declared in CU:
	 */
	for (int i = 0; i < CU.elems.size(); i++) {
	    TypeDecl d = CU.elems.elementAt(i);
	    if (id == d.id) {
		if (caller == null) return TypeSig.getSig(d);
		if (TypeCheck.inst.canAccess(caller,TypeSig.getSig(d),d.modifiers,d.pmodifiers)) return TypeSig.getSig(d);
	    }
	}


	/*
	 * Next, look in single-type imports of CU
	 */
	for(int i = 0; i < CU.imports.size(); i++) {
	    try {
		Name imp =
		    ((SingleTypeImportDecl)CU.imports  //@ nowarn Cast; //caught
		          .elementAt(i)).typeName.name; 
		int sz = imp.size();
		String[] P = imp.toStrings(sz-1);
		Identifier T = imp.identifierAt(sz-1);

		if (id == T) {
		    TypeSig r = lookupWithoutInheritence(caller, P, T.toString());
		    if (r != null) return r;
		}
	    } catch (ClassCastException e) { }
	}


	/*
	 * Next, look in package of CU:
	 */
	{
	    String[] P = new String[0];
	    if (CU.pkgName != null)
		P = CU.pkgName.toStrings();
	    TypeSig r = lookupWithoutInheritence(caller, P, id.toString());
	    if (r != null) return r;
	}


	/*
	 * Next, look in on-demand imports of CU:
	 */
	{ 
			// FIXME - caller argument?
	    TypeSig r = OutsideEnv.lookup(Types.javaLangPackage(),
					  id.toString());
	    for(int i = 0; i < CU.imports.size(); i++) {
		try {
		    OnDemandImportDecl imp =
		      (OnDemandImportDecl)CU.imports.elementAt(i); //@ nowarn Cast; //caught
		    TypeSig r2 = lookupWithoutInheritence(caller,
					   imp.pkgName.toStrings(),
					   id.toString());
		    if (r2 != null)
			if (r != null && r != r2) {
			    if (loc != Location.NULL)
				ErrorSet.fatal(loc, 
					    "Ambiguous import-on-demand of \""
					    + id.toString() + "\". Choices are: " + r.getExternalName() + " and " + r2.getExternalName());
			    else
				return r;
			} else r = r2;
		} catch (ClassCastException e) { }
	    }
	    if (r != null) return r;
	}
	
	return null;
    }


    /**
     * Attempt to lookup the type N.I without using inheritence in
     * the outside environment.  (N.I may divide into package and
     * class parts arbitrarily.) <p>
     *
     * This routine does not check that the resulting type (if any)
     * is actually accessable, unless caller is non-null. <p>
     */
    //@ requires \nonnullelements(N);
    public static TypeSig lookupWithoutInheritence(TypeSig caller, String[] N,
						   /*@ non_null @*/ String I) {
	TypeSig soFar = null;
	int prefix = 0;

	/*
	 * Find the smallest prefix that denotes an package-member
	 * type P.T:
	 */
	while (++prefix<=N.length+1) {
	    // Let P.T = first prefix ids in N.I:
	    String[] P = new String[prefix-1];
	    for (int i=0; i<prefix-1; i++)
		P[i] = N[i];
	    String T = (prefix<=N.length) ? N[prefix-1] : I;

	    soFar = OutsideEnv.lookup(P, T);
	    if (soFar != null)
		break;
	}
	if (soFar==null)
	    return null;
      
	// Remaining part of N.I must be a $C1$C2...$Cn expression:
	while (++prefix<=N.length+1) {
	    String T = (prefix<=N.length) ? N[prefix-1] : I;
	    soFar = soFar.lookupLocalType(caller,Identifier.intern(T));
	    if (soFar==null)
		return null;
	}
	
	return soFar;
    }


    /**
     * Locate the lexically innermost method named id. <p>
     *
     * Returns the TypeSig for the innermost lexically enclosing type
     * that has a method named id or null if no such type exists.<p>
     *
     * Note: inherited methods are considered to lexically enclose
     * the code of their subclasses.<p>
     *
     * id disambiguates to C[.this].id.<p>
     */
    public TypeSig locateMethod(/*@non_null*/Identifier id) {
	// We bind no methods:
	return null;
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /**
     * Display information about us to System.out.  This function is
     * intended only for debugging use.
     */
    public void display() {
	System.out.println("[[ environment for compilation unit from file \""
	    + Location.toFileName(CU.loc) + "\" ]]");
    }
}
