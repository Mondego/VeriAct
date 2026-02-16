/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.tc;


import java.util.Hashtable;

import javafe.ast.*;
import javafe.util.*;
import javafe.tc.*;
import javafe.tc.TypeSig;


/**
 * EnvForGhostLocals are used to extend an existing Env with one new local
 * binding, either a local variable definition or a formal parameter. <p>
 *
 * See EnvForLocalType for how to extend an existing Env with one new
 * local type binding.<p>
 */

public class EnvForGhostLocals extends Env implements/*privately*/ Cloneable {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * Our parent environment
     */
    //@ invariant !(parent instanceof EnvForCU);
    protected /*@ non_null @*/ Env parent;

    /**
     * The new local binding.
     */
    //@ invariant decl.id != null;
    protected /*@ non_null @*/ GenericVarDecl decl;


    /**
     * Create a environment from an existing one by adding a new
     * local binding. <p>
     *
     * We report an error to ErrorSet if the new local binding is a
     * redefinition of a local binding not hidden by a field.<p>
     */
    //@ requires decl.id != null;
    //@ requires !(parent instanceof EnvForCU);
    public EnvForGhostLocals(/*@ non_null @*/ Env parent,
			/*@ non_null @*/ GenericVarDecl decl) {
	this.parent = parent;
	this.decl = decl;

	TypeSig declaringClass = parent.getEnclosingClass();
	Assert.notNull(declaringClass);
	whereDecoration.set(decl, declaringClass);

	/*
	 * Check for duplication:
	 *
	 * (Note that result returns a TypeSig if decl.id refers to a
	 * field.)
	 */
	ASTNode result = parent.locateFieldOrLocal(decl.id);
	if (result instanceof GenericVarDecl)
	    // && whereDeclared((GenericVarDecl)result)==declaringClass)
	    ErrorSet.error(decl.locId, "Variable '" + decl.id +
		   "' is already defined here.", result.getStartLoc());
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
     * The legality of C.this, C!=<enclosing class> is different; see 
     * canAccessInstance(-).
     */
    public boolean isStaticContext() { return parent.isStaticContext(); }


    /**
     * Return the intermost class enclosing the code that is checked
     * in this environment. <p>
     *
     * May return null if there is no enclosing class (aka, for
     * environments for CompilationUnits). <p>
     *
     * If isStaticContext() returns true, then this is the type of "this".
     */
    public /*@ nullable */ TypeSig getEnclosingClass() {
	return parent.getEnclosingClass();
    }


    /**
     * If there is an enclosing instance in scope, then return the
     * (exact) type of the innermost such instance. <p>
     *
     * Note: this is considered a current instance, not an enclosing
     * instance, even inside its methods.
     */
    public /*@ nullable */ TypeSig getEnclosingInstance() {
	return parent.getEnclosingInstance();
    }


    /**
     * Returns a new Env that acts the same as us, except that its
     * current instance (if any) is not accessible. <p>
     *
     * Note: this routine is somewhat inefficient and should be
     * avoided unless an unknown environment needs to be coerced in
     * this way. <p>
     */
    public /*@ non_null @*/ Env asStaticContext() {
	EnvForGhostLocals n;
	try {
	    n = (EnvForGhostLocals)this.clone();
	} catch (CloneNotSupportedException e) {
	    Assert.fail("clone did not obey its spec!");
	    return null;  // keep compiler happy
	}
	n.parent = parent.asStaticContext();
	return n;
    }


    /***************************************************
     *                                                 *
     * Simple names:				       *
     *                                                 *
     **************************************************/

    /**
     * Attempt to lookup a simple TypeName in this environment to get
     * the TypeSig it denotes.  Returns null if no such type
     * exists.<p>
     *
     * This routine does not check that the resulting type (if any)
     * is actually accessable, if caller is null. <p>
     *
     * If id is ambiguous, then if loc != Location.NULL then a fatal
     * error is reported at that location via ErrorSet else one of
     * its possible meanings is returned.<p>
     */
    public /*@ nullable */ TypeSig lookupSimpleTypeName(/*@ nullable */ TypeSig caller, /*@non_null*/Identifier id, int loc) {
	// We bind no type variables ourshelves:
	return parent.lookupSimpleTypeName(caller, id, loc);
    }


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
    public /*@ nullable */ ASTNode locateFieldOrLocal(/*@non_null*/Identifier id) {
	// Only return my declaration if it matches and we are in an
	// annotation environment
	if (id == decl.id && FlowInsensitiveChecks.inAnnotation)
	    return decl;
	else
	    return parent.locateFieldOrLocal(id);
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
    public /*@ nullable */ TypeSig locateMethod(/*@non_null*/Identifier id) {
	// we bind no methods ourshelves:
	return parent.locateMethod(id);
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
	parent.display();
	System.out.println("[[ extended with local binding of "
	    + decl.id.toString() + " bound to:");
	PrettyPrint.inst.print(System.out, decl);
	System.out.println("]]");
    }
}
