/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;

/**
 * EnvForTypeSigs are used to extend an existing Env with the
 * bindings of a TypeSig. <p>
 *
 * Each TypeSig has two different internal environments, depending on
 * whether or not its instance members are considered to be
 * accessible.  (Static members are always visible.)  The creator
 * specifies which of the two environments are desired.
 */

public class EnvForTypeSig extends Env {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * Our parent environment
     */
    protected /*@ non_null @*/ Env parent;

    /**
     * The TypeSig providing new bindings
     */
    protected /*@ non_null @*/ TypeSig peervar;

    /**
     * Are peervar's instance members accessible? <p>
     *
     * WARNING: never modify this variable after creation -- Env's
     * need to be immutible.
     */
    protected boolean staticContext;


    /**
     * Create an environment from an existing one by adding a new
     * TypeSigs bindings. <p>
     *
     * The instance members of the TypeSig are considered accessible
     * iff staticContext is true.
     */
    public EnvForTypeSig(/*@ non_null @*/ Env parent,
			 /*@ non_null @*/ TypeSig peervar,
			 boolean staticContext) {
	this.parent = parent;
	this.peervar = peervar;
	this.staticContext = staticContext;
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
    public boolean isStaticContext() {
	return staticContext;
    }


    /**
     * Return the intermost class enclosing the code that is checked
     * in this environment. <p>
     *
     * May return null if there is no enclosing class (aka, for
     * environments for CompilationUnits). <p>
     *
     * If isStaticContext() returns true, then this is the type of "this".
     */
    public TypeSig getEnclosingClass() {
	return peervar;
    }


    /**
     * If there is an enclosing instance in scope, then return the
     * (exact) type of the innermost such instance. <p>
     *
     * Note: this is considered a current instance, not an enclosing
     * instance, even inside its methods.
     */
    public TypeSig getEnclosingInstance() {
	/*
	 * For now, the rule is: we have enclosing instances iff we
	 * are not in a static context: !!!!
	 */
	Env outsideEnv = peervar.getEnclosingEnv();
	if (outsideEnv.isStaticContext())
	    return null;

	// Our first enclosing instance is our enclosing type:
	return peervar.enclosingType;
    }


    /**
     * Returns a new Env that acts the same as us, except that its
     * current instance (if any) is not accessible. <p>
     *
     * Note: this routine is somewhat inefficient and should be
     * avoided unless an unknown environment needs to be coerced in
     * this way. <p>
     */
    public /*@non_null*/Env asStaticContext() {
	return new EnvForTypeSig(parent, peervar, true);
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
     * is actually accessable. <p>
     *
     * If id is ambiguous, then if loc != Location.NULL then a fatal
     * error is reported at that location via ErrorSet else one of
     * its possible meanings is returned.<p>
     */
    public TypeSig lookupSimpleTypeName(TypeSig caller, /*@non_null*/Identifier id, int loc) {
	// Check for a definition in peervar:
	TypeSig result = peervar.lookupType(caller, id, loc);
	if (result != null) return result;

	// Otherwise, look to enclosing scopes...
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
    public ASTNode locateFieldOrLocal(/*@non_null*/Identifier id) {
	if (hasField(id))
	    return peervar;

	return parent.locateFieldOrLocal(id);
    }

    public boolean isDuplicate(/*@non_null*/Identifier id) {
	if (hasField(id)) return true;
	return false;
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
	// Check for a definition in peervar:
	if (hasMethod(id))
	    return peervar;

	// Otherwise, look to enclosing scopes...
	return parent.locateMethod(id);
    }


    /** This is to allow overriding by subclasses */

    protected boolean hasField(Identifier id) {
	return peervar.hasField(id);
    }

    public FieldDeclVec getFields(boolean allFields) {
	return peervar.getFields(allFields);
    }

    protected boolean hasMethod(Identifier id) {
	MethodDeclVec methods = peervar.getMethods();

	for (int i = 0; i < methods.size(); i++) {
	    MethodDecl md = methods.elementAt(i);
	    if (md.id == id)
		return true;
	}

	return false;
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /**
     * Display information about an Env to System.out.  This function
     * is intended only for debugging use.
     */
    public void display() {
	parent.display();
	System.out.println("[[ extended with the "
			   + (staticContext ? "static" : "complete")
			   + " bindings of type "
			   + peervar.getExternalName() + " ]]");
    }
}
