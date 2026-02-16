/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;
import javafe.util.*;


/**
 * Env's are the environments used during typechecking to keep track
 * of what types, local variables, fields, and current/enclosing
 * instances are in scope.
 */

public abstract class Env {

    ///////////////////////////////////////////////////////////////////////
    //                                                                   //
    // Methods implemented by each instance of Env                       //
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////


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
    abstract public boolean isStaticContext();


    /**
     * Return the innermost class enclosing the code that is checked
     * in this environment.<p>
     *
     * May return null if there is no enclosing class (see postcondition).<p>
     *
     * If isStaticContext() returns true, then this is the type of "this".
     */
    //@ ensures (this instanceof EnvForCU) == (\result==null);
    abstract public TypeSig getEnclosingClass();


    /**
     * If there is an enclosing instance in scope, then return the
     * (exact) type of the innermost such instance. <p>
     *
     * Note: this is considered a current instance, not an enclosing
     * instance, even inside its methods.
     */
    abstract public TypeSig getEnclosingInstance();


    /**
     * Returns a new Env that acts the same as us, except that its
     * current instance (if any) is not accessible. <p>
     *
     * Note: this routine is somewhat inefficient and should be
     * avoided unless an unknown environment needs to be coerced in
     * this way. <p>
     */
    //@ ensures (this instanceof EnvForCU) == (\result instanceof EnvForCU);
    abstract public /*@non_null*/Env asStaticContext();


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
     * is actually accessible, if caller is null. <p>
     *
     * If id is ambiguous, then if loc != Location.NULL then a fatal
     * error is reported at that location via ErrorSet else one of
     * its possible meanings is returned.<p>
     */
    abstract public TypeSig lookupSimpleTypeName(
				/*@ nullable */ TypeSig caller, /*@ non_null @*/ Identifier id,
						 int loc);


    /**
     * Locate the lexically innermost field or local variable
     * declaration with a given name. <p>
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
     *
     * This routine does not check that a resulting field
     * is actually accessible. <p>
     */
    /*@ ensures \result==null || (\result instanceof GenericVarDecl)
                || (\result instanceof TypeSig); */
    /*@ ensures \result instanceof GenericVarDecl ==>
                       ((GenericVarDecl)\result).id == id; */
    //@ ensures (this instanceof EnvForCU) ==> \result==null;
    abstract public ASTNode locateFieldOrLocal(/*@ non_null @*/ Identifier id);

    public boolean isDuplicate(/*@ non_null */ Identifier id) {
	return locateFieldOrLocal(id) instanceof GenericVarDecl;
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
     *
     * This routine does not check that a resulting method
     * is actually accessible. <p>
     */
    //@ ensures (this instanceof EnvForCU) ==> \result==null;
    abstract public TypeSig locateMethod(/*@ non_null @*/ Identifier id);


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /**
     * Display information about us to System.out.  This function is
     * intended only for debugging use.
     */
    abstract public void display();


    ///////////////////////////////////////////////////////////////////////
    //                                                                   //
    // Derived Methods				                         //
    //                                                                   //
    ///////////////////////////////////////////////////////////////////////


    /***************************************************
     *                                                 *
     * Type variable names:			       *
     *                                                 *
     **************************************************/

    /**
     * Attempts to find the canonical prefix of a given name that
     * denotes a TypeName in this environment. <p>
     *
     * A canonical prefix is composed of a base type name (either the
     * leftmost identifer or a fully-quantified outside type name
     * (P.I) depending), extended by some number of type member
     * accesses (.C1.C2...).
     *
     * If ignoreFields is not set, then we stop extending the base
     * type name as soon as we encounter an access that can refer to
     * a field.  If it is set, then we stop extending only when we
     * reach the end of the name or an access that cannot refer to a
     * type member. <p>
     *
     * If we encounter an ambiguous prefix, we report a fatal error
     * at loc via ErrorSet.<p>
     *
     * Otherwise, we return the TypeSig that the found prefix denotes
     * (null if the prefix is of length 0) and sets prefixSize to the
     * prefix's size. <p>
     *
     * This routine does not check that the resulting type (if any)
     * is actually accessible, unless caller is not null. <p>
     */
    //@ modifies prefixSize;
    //@ ensures \result==null ==> 0==prefixSize;
    //@ ensures \result != null ==> 0<prefixSize && prefixSize <= n.length;
    public TypeSig findTypeNamePrefix(TypeSig caller,
				      /*@ non_null @*/ Name n,
				      boolean ignoreFields) {
	// Check for an unqualified name first:
	TypeSig sig = lookupSimpleTypeName(caller,n.identifierAt(0),
					   n.getStartLoc());
	prefixSize = 1;
	if (sig==null) {
	    // Nope; must be a qualified name with a non-empty package prefix:
	    while (++prefixSize<=n.size()) {
		// Lookup n[0]..n[prefixSize-2] . n[prefixSize-1]:
		sig = OutsideEnv.lookup(n.prefix(prefixSize-1).toStrings(),
				n.identifierAt(prefixSize-1).toString());
		if (sig != null)
		    break;      // Stop at smallest qualified name
	    }
	    if (prefixSize>n.size()) {
		prefixSize = 0;
		return null;
	    }
	}

	// Here we have that n[0]..n[prefixSize-1] denotes the type sig

	// Try and extend it as much as possible via nested types
	while (prefixSize<n.size()) {
	    Identifier id = n.identifierAt(prefixSize);
	    int idLoc = n.locIdAt(prefixSize);

	    // Stop if next access refers to a field & !ignoreFields:
	    if (!ignoreFields && sig.hasField(id))
		break;

	    // Stop if next access cannot refer to a type member:
	    TypeSig next = sig.lookupType(caller, id, idLoc);
	    if (next==null)
		break;

	    sig = next;
	    prefixSize++;
	}

	return sig;
    }
    //* Holds the second return value of findTypeNamePrefix
    public int prefixSize;


    /**
     * Attempt to lookup a TypeName using this environment. <p>
     *
     * If it encounters an ambiguous prefix, a fatal error is
     * reported via ErrorSet.<p>
     *
     * Otherwise, returns the TypeSig that n denotes or null if n
     * does not denote a type.<p>
     *
     * This routine does not check that the resulting type (if any)
     * is actually accessible, unless caller is not null. <p>
     */
    public TypeSig lookupTypeName(TypeSig caller, /*@ non_null @*/ Name n) {
	TypeSig sig = findTypeNamePrefix(caller, n, true);
	if (prefixSize != n.size())
	    return null;

	return sig;
    }


    /**
     * This processes the annotations on a type name
     */
    //@ ensures \result != null;
    public TypeSig processTypeNameAnnotations(/*@ non_null @*/ TypeName n, 
					      /*@ non_null @*/ TypeSig sig) {
      	return PrepTypeDeclaration.inst.processTypeNameAnnotations(n,sig,this);
    }
    

    /**
     * Attempt to resolve a TypeName using this environment. <p>
     *
     * If an error occurs (including no such type), reports it to
     * ErrorSet via a fatal error.<p>
     *
     * Otherwise, returns the TypeSig that n denotes.  This TypeSig
     * may also later be obtained by using TypeSig.getSig on n.<p>
     *
     * This routine does not check that the resulting type (if any)
     * is actually accessible, unless caller is not null. <p>
     */
    //@ ensures \result != null;
    public TypeSig resolveTypeName(TypeSig caller, /*@ non_null @*/ TypeName tn) {
	Name n = tn.name;

	TypeSig sig = TypeSig.getRawSig(tn);  // FIXME - use caller ?
	if (sig != null)
	    return sig;

	sig = lookupTypeName(caller, n);
	if (sig==null) {
	    ErrorSet.fatal(tn.name.locIdAt(0),
			   "Can't find type named \""
			   + tn.name.printName() + "\"");
	}

	sig = processTypeNameAnnotations(tn, sig);  // FIXME - use caller?
	TypeSig.setSig(tn, sig);
	return sig;
    }

    /**
     * decoration holding the type environment in which a type is resolved.
     */
    //@ invariant typeEnv != null;
    //@ invariant typeEnv.decorationType == \type(Env);
    static public ASTDecoration typeEnv = 
	new ASTDecoration("environment");

    /**
     * Attempt to resolve a Type using this environment. <p>
     *
     * If an error occurs, reports it to ErrorSet via a fatal error.<p>
     *
     * This routine does not check that (immediate) types (if any)
     * are actually accessible, if caller is null. <p>
     */
    public void resolveType(TypeSig caller, /*@ non_null @*/ Type t) {
      	typeEnv.set(t,this);
	switch(t.getTag()) {
	  case TagConstants.ARRAYTYPE:
	    resolveType(caller, ((ArrayType)t).elemType);
	    break;

	  case TagConstants.TYPENAME:
	    resolveTypeName(caller, (TypeName)t);
	    break;

          // No need to resolve primitive types or TypeSigs...
	}
    }


    /***************************************************
     *                                                 *
     * Expr names:				       *
     *                                                 *
     **************************************************/

    /**
     * Attempt to disambiguate an Expr Name.  Either returns the
     * disambiguated Name as an Expr or null if it does not denote
     * anything. <p>
     *
     * If non-null, the result will always be a field access of some
     * kind.
     *
     * If a prefix of n is ambiguous because of multiple
     * import-on-demand declarations, a fatal error will result.
     * Nothing is reported if n does not name anything.<p>
     *
     *
     * If n is a reference to a field f in lexically enclosing class
     * C, then the result will be of the form "[C.]this.n" if C's
     * instance fields are accessible and "C.n" otherwise.<p>
     *
     * (At this point we haven't decided which field f refers to so
     * we don't know if it is an instance field or not.)
     */
    //@ ensures !(\result instanceof AmbiguousVariableAccess);
    public Expr disambiguateExprName(/*@ non_null @*/ Name n) {
	/*
	 * Find the smallest prefix of n, n[0]..n[prefix-1], such that
	 * it denotes an Expr, call it left:
	 */
	Expr left = null;
	Identifier leftmost = n.identifierAt(0);
	int leftmostLoc = n.getStartLoc();

	ASTNode result = locateFieldOrLocal(leftmost);
	int prefix = 1;
	if (result instanceof GenericVarDecl) {
	    // leftmost is a local variable:
	    left = VariableAccess.make(leftmost, leftmostLoc,
				       (GenericVarDecl)result);
	} else if (result instanceof TypeSig) {
	    // leftmost is a field in class result:
	    ObjectDesignator od = getObjectDesignator((TypeSig)result,
						      leftmostLoc);
	    left = FieldAccess.make(od, leftmost, leftmostLoc);
	} else {
	    // n *must* start with a typename then:
	    TypeSig sig = findTypeNamePrefix(null, n, false); // FIXME - a caller for access checking?
	    if (sig==null || prefixSize==n.size())
		return null;

	    prefix = prefixSize+1;
	    TypeName tn = TypeName.make(n.prefix(prefixSize));
	    TypeSig.setSig(tn, sig);
	    int leftmostDot = leftmostLoc;
	    if (prefixSize>0)
		leftmostDot = n.locDotAfter(prefixSize-1);
	    ObjectDesignator od =
		TypeObjectDesignator.make(leftmostDot, tn);
	    left = FieldAccess.make(od, n.identifierAt(prefixSize),
				    n.locIdAt(prefixSize));
	}


	/*
	 * Extend the prefix to the full n by using field dereferences:
	 */
	for (; prefix<n.size(); prefix++) {
	    left = FieldAccess.make(
			    ExprObjectDesignator.make(n.locDotAfter(prefix-1),
						      left),
			    n.identifierAt(prefix),
			    n.locIdAt(prefix));
	}

	return left;
    }

    public Object disambiguateTypeOrFieldName(/*@ non_null */ Name n) {
	/*
	 * Find the smallest prefix of n, n[0]..n[prefix-1], such that
	 * it denotes an Expr, call it left:
	 */
	Expr left = null;
	Identifier leftmost = n.identifierAt(0);
	int leftmostLoc = n.getStartLoc();

	ASTNode result = locateFieldOrLocal(leftmost);
	int prefix = 1;
	if (result instanceof GenericVarDecl) {
	    // leftmost is a local variable:
	    left = VariableAccess.make(leftmost, leftmostLoc,
				       (GenericVarDecl)result);
	} else if (result instanceof TypeSig) {
	    // leftmost is a field in class result:
	    ObjectDesignator od = getObjectDesignator((TypeSig)result,
						      leftmostLoc);
	    left = FieldAccess.make(od, leftmost, leftmostLoc);
	} else {
	    // n *must* start with a typename then:
	    TypeSig sig = findTypeNamePrefix(null, n, false); // FIXME - a caller for access checking?
	    if (sig==null) return null;
	    if (prefixSize==n.size()) return sig;

	    prefix = prefixSize+1;
	    TypeName tn = TypeName.make(n.prefix(prefixSize));
	    TypeSig.setSig(tn, sig);
	    int leftmostDot = leftmostLoc;
	    if (prefixSize>0)
		leftmostDot = n.locDotAfter(prefixSize-1);
	    ObjectDesignator od =
		TypeObjectDesignator.make(leftmostDot, tn);
	    left = FieldAccess.make(od, n.identifierAt(prefixSize),
				    n.locIdAt(prefixSize));
	}


	/*
	 * Extend the prefix to the full n by using field dereferences:
	 */
	for (; prefix<n.size(); prefix++) {
	    left = FieldAccess.make(
			    ExprObjectDesignator.make(n.locDotAfter(prefix-1),
						      left),
			    n.identifierAt(prefix),
			    n.locIdAt(prefix));
	}

	return left;
    }


    /***************************************************
     *                                                 *
     * Routine names:				       *
     *                                                 *
     **************************************************/

    /**
     * Attempt to disambiguate an AmbiguousMethodInvocation.  Either
     * returns the disambiguated method invocation as an Expr or
     * reports a fatal error to ErrorSet if it does not denote
     * anything.<p>
     *
     * The result will always be a method invocation.<p>
     *
     * If a prefix of n is ambiguous because of multiple
     * import-on-demand declarations, a fatal error will result.
     *
     *
     * If n is a reference to a method m in lexically enclosing class
     * C, then the result will be of the form "[C.]this.m" if C's
     * instance methods are accessible and "C.m" otherwise.<p>
     *
     * (At this point we haven't decided which method m refers to so
     * we don't know if it is an instance method or not.)
     */
    public MethodInvocation disambiguateMethodName(
				/*@ non_null @*/ AmbiguousMethodInvocation inv) {
	ObjectDesignator where;     // Where the method comes from
	
	Name n = inv.name;
	int size = n.size();
	int nStart = n.getStartLoc();


	/*
	 * Handle the simple name case first:
	 */
	if (n.size()==1) {
	    Identifier id = n.identifierAt(0);
	    TypeSig container = locateMethod(id);
	    if (container==null)
		ErrorSet.fatal(nStart,
			       "No method named " + id + " is in scope here.");
	    
	    where = getObjectDesignator(container, nStart);
	} else {
	    /*
	     * Not a simple name; try preceeding name as an ExprName first:
	     */
	    Name butRight = n.prefix(size-1);
	    Expr e = disambiguateExprName(butRight);
	    if (e != null)
		where = ExprObjectDesignator.make(n.locDotAfter(size-2), e);
	    else {
		/*
		 * then try preceeding name as a TypeName:
		 */
		TypeSig t = lookupTypeName(null,butRight); // FIXME - a caller for access checking?
		if (t != null) {
		    TypeName tn = TypeName.make(butRight);
		    TypeSig.setSig(tn, t);
		    where = TypeObjectDesignator.make(n.locDotAfter(size-2),
						      tn);
		} else {
		    /*
		     * Give up!
		     */
		    ErrorSet.fatal(nStart,
			"Can't disambiguate method name " + n.printName());
		    where = null;     // keep compiler happy
		}
	    }
	}


	// Return <where>.rightmost(...):
	Identifier rightmost = n.identifierAt(size-1);
	int rightmostLoc = n.locIdAt(size-1);
	return MethodInvocation.make(where, rightmost, null, rightmostLoc,
				     inv.locOpenParen, inv.args);
    }


    /***************************************************
     *                                                 *
     * Current/enclosing instances II:		       *
     *                                                 *
     **************************************************/

    /**
     * Returns the innermost current or enclosing instance, or null
     * if none exists.
     */
    public TypeSig getInnermostInstance() {
	if (!isStaticContext())
	    return getEnclosingClass();
	else
	    return getEnclosingInstance();
    }


    /**
     * Are C's instance variables accessible? <p>
     *
     * If C is getEnclosingClass(), then this is equivalent to
     * isStaticContext().
     */
    public boolean canAccessInstance(/*@ non_null @*/ TypeSig C) {
	/*
	 * C's instance variables are accessible iff C is one of our
	 * current or enclosing instances:
	 */
	for (TypeSig instance = getInnermostInstance();
	     instance != null;
	     instance = instance.getEnv(true).getEnclosingInstance()) {
	    if (instance==C)
		return true;
	}

	return false;
    }


    /**
     * Attempt to locate a current or enclosing instance that has
     * type T. <p>
     *
     * If such exist, return an inferred "<actual class>.this" Expr
     * for the innermost such one; otherwise, return null.  The
     * location fields of the Expr will be set to loc.<p>
     *
     * Note: The returned instance may have be of a subtype of T.<p>
     */
    //@ requires loc != Location.NULL;
    public Expr lookupEnclosingInstance(/*@ non_null @*/ TypeSig T,
					int loc) {
	TypeSig instance;

	// Find innermost satisfactory instance if it exists:
	for (instance = getInnermostInstance();
	     instance != null;
	     instance = instance.getEnv(true).getEnclosingInstance()) {
	    if (instance.isSubtypeOf(T))
		break;
	}

	if (instance==null)
	    return null;
       
	return getInferredThisExpr(instance, loc);
    }


    /***************************************************
     *                                                 *
     * Finding where something is declared:	       *
     *                                                 *
     **************************************************/

    /**
     * Decorates LocalVarDecl and FormalParaDecl nodes to point to
     * the TypeSig of the type they are declared in. <p>
     *
     * Set by the EnvForLocals constructor.<p>
     */
    //@ invariant whereDecoration.decorationType == \type(TypeSig);
    protected static final ASTDecoration whereDecoration
	= new ASTDecoration("whereDecoration");


    /**
     * What type is a GenericVarDecl declared in? <p>
     *
     * Precondition: decl's type has been "parsed"; an Env containing
     * decl has been constructed.<p>
     */
    //@ requires (decl instanceof FieldDecl) ==> ((FieldDecl)decl).hasParent;
    //@ ensures \result != null;
    public static TypeSig whereDeclared(/*@ non_null @*/ GenericVarDecl decl) {
	TypeSig result;

	if (decl instanceof FieldDecl) {
	    TypeDecl parent = ((FieldDecl)decl).getParent();
	    Assert.notNull(parent);
	    result = TypeSig.getSig(parent);
	} else {
	    // LocalVarDecl or FormalParaDecl left here:
	    result = (TypeSig)whereDecoration.get(decl);
	}

	Assert.notNull(result);          //@ nowarn Pre;
	return result;
    }


    /***************************************************
     *                                                 *
     * Expr-construction utility functions:	       *
     *                                                 *
     **************************************************/

    /**
     * Return an inferred ThisExpr for "[C.]this", using location loc. <p>
     *
     * The "C." part is omitted if C is the type of this (e.g.,
     * getEnclosingClass()).
     */
    //@ requires loc != Location.NULL;
    //@ ensures \result != null;
    public final ThisExpr getInferredThisExpr(/*@ non_null @*/ TypeSig C,
					      int loc) {
	ThisExpr newThis = ThisExpr.make((C == getEnclosingClass())
                                         ? null : C, loc);
	newThis.inferred = true;

	return newThis;
    }


    /**
     * Return an inferred ObjectDesignator for use in a reference to
     * a possibly-instance member of class C from here. <p>
     *
     *
     * If C's instance variables are not accessible from this point
     * (see canAccessInstance(-)), then returns "C.". <p>
     *
     * Otherwise returns an inferred "[C.]this.".
     * (cf. getInferredThisExpr(-))
     *
     * loc is used as the location for the this. and C. parts.
     */
    //@ requires loc != Location.NULL;
    //@ ensures \result != null;
    public final ObjectDesignator getObjectDesignator(/*@ non_null @*/ TypeSig C,
						      int loc) {
	if (!canAccessInstance(C))
	    return TypeObjectDesignator.make(loc, C);
	else
	    return ExprObjectDesignator.make(loc,
					     getInferredThisExpr(C, loc));
    }
}
