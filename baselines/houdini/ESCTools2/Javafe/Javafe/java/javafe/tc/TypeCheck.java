/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import java.util.Enumeration;
import javafe.ast.*;
import javafe.util.Set;
import javafe.util.Assert;
import javafe.util.Info;

import javafe.tc.TagConstants;	// resolve ambiguity

/**
 * The <code>TypeCheck</code> class contains methods to disambiguate, resolve,
 * and check type declarations.  (Before methods in this class can be called, the
 * {@link TypeSig} class must be initialized.)
 * 
 * <h3> Overview of checking, resolution, and disambiguation </h3>
 * 
 * <p> This description is out of date, particularly with respect to the names
 *     of classes. </P>
 *
 * <p> <em>Checking</em> involves performing the static checks specified by the Java
 * language specification. </p>
 * 
 * <p> <em>Resolution</em> involves connecting symbolic references in the parse tree
 * to objects representing declarations of the referred-to entities.  The parser
 * generates a number of nodes -- instances of {@link Identifier}, {@link FieldAccess},
 * and {@link MethodDecl} -- containing identifiers found in the input plus a
 * <code>decl</code> field which is initially <code>null</code>.  Resolution sets
 * these <code>decl</code> fields to point to the declaration referred to by the
 * identifiers.  Similarly, {@link TypeName} nodes have a <code>sig</code> field
 * which is initially <code>null</code> and which must be resolved to an instance of
 * {@link TypeSig}.  For example, the name <code>java.lang.String</code> appearing in
 * a type context would be parsed to a {@link TypeName}; resolution of this node
 * would point its <code>sig</code> field to the {@link TypeSig} object representing
 * Java's standard {@link String} type. </p>
 * 
 * <p> <em>Disambiguation</em> deals with "ambiguous names" in Java (see Section Six
 * of the Java language specification, or <a
 * href="http://src-www.pa.dec.com/~stata/ESCJava/naming.html">this document</a>).
 * These are qualified names of the form <code>I1.I2...In</code> that appear in an
 * expression context.  For such a name, <code>I1</code> could be a local variable or
 * a field of <code>this</code>, or some prefix of the name could be the
 * fully-qualified name of a type, as in <code>java.lang.String.concat</code>. </p>
 * 
 * <p> When it encounters an ambiguous name, the parser generates either an {@link
 * AmbiguousVariableAccess} or {@link AmbiguousMethodInvocation} node depending on the context.  These are leaf
 * nodes.  In these cases, disambiguation involves replacing these nodes with
 * appropriate {@link VariableAccess} or {@link MethodInvocation} nodes; these are non-leaf
 * nodes, and in general the replacement might be fairly deep. </p>
 * 
 * <p> As an example of disambiguation, assume the name <code>x.y</code> is parsed as
 * an {@link AmbiguousVariableAccess}.  Assume further that no local named <code>x</code> is in
 * scope, the current scope is in an instance method for a class that has a field
 * named <code>x</code>.  In this case, disambiguation would replace this {@link
 * AmbiguousVariableAccess} with:
 * 
 * <blockquote>
 * <code>(FieldAccess (FieldAccess this x) y)</code>
 * </blockquote>
 * 
 * that is, an instance of {@link FieldAccess} whose <code>id</code> field was
 * <code>y</code> and whose <code>expr</code> field was another {@link
 * FieldAccess} whose <code>id</code> field was <code>x</code> and whose
 * <code>expr</code> field was an instance of {@link ThisExpr}. </p>
 * 
 * <p> An alternative design for disambiguation and resolution was considered.  In
 * this design, the {@link Name} class, the three subclasses of {@link FieldAccess},
 * and the three subclasses of {@link MethodInvocation} would be replaced with a new
 * expression class that looked something like:
 * 
 * <blockquote>
 * <pre>
 * class DotExpr extends Expr {
 * int tag; // Indicates the kind of dot
 * Expr expr;
 * Identifier id;
 * }
 * </pre>
 * </blockquote>
 * </p>
 * 
 * <p> When confronted with phrases of the form <code>I1.I2...In</code>, the parser
 * would generate trees of DotExpr nodes all with the same tag, this tag
 * indicating that the meaning of the dot was ambiguous.  Disambiguation would
 * involve replacing this ambiguous tag with a tag whose meaning was clear (e.g., a
 * tag that meant "select a type from a package").  Resolution would involve using
 * our generic decoration mechanism to link certain of these nodes with the
 * declarations to which they refer. </p>
 * 
 * <p> The advantages of this approach over the one we selected is that it is more
 * conventional (it's been used, for example, in compilers for the Modula family of
 * languages), it has a simpler class hierarchy, and it does not involve mutating the
 * structure of the parse tree.  The primary advantage of our approach is that we
 * capture many more invariants in the type system, leaving less to go wrong at
 * run-time.  It is mostly for this reason that we selected it.  In addition, our
 * approach takes less space to represent type names, avoids downcasting (which can
 * be costly time-wise), and is more friendly to the "visitor" pattern. </p>
 * 
 * 
 * <h3> Staging the processing of type declarations </h3>
 * 
 * <p> Resolving and checking a type declaration usually involves looking at other
 * declarations to which it refers.  Finding, reading, and processing referred-to
 * types makes resolution and checking fairly complicated.  As a result, we have
 * decomposed it up into smaller steps.  Type declarations move through a number of
 * states as the resolution and checking process procedes.  In addition to making the
 * overall processing of type declarations conceptually more manageable, this
 * decomposition has two other benefits: </p>
 * 
 * <ul>
 * 
 * <li> <em>Handling cycles.</em> As mentioned above, processing one type may involve
 * processing types to which it refers.  However, two types may refer to each other,
 * making it impossible to process any one of them "first."  Decomposing the
 * processing into stages helps us handle such cycles. </li>
 * 
 * <li> <em>Improving performance.</em> Processing one type declaration does not
 * always involve fully processing the declarations to which it refers.  How much
 * processing is required of a referred-to type depends on the manner in which it is
 * referred (e.g., in a method signature versus as a superclass).  Decomposing
 * processing into stages allows us to be lazy in processing referred-to types, that
 * is, allowing us to process them only to the extent that is necessary and no
 * further. </li>
 * 
 * </ul>
 * 
 * <p> To support the lazy handling of type declarations, type declarations are
 * represented using two objects: {@link TypeDecl}s and {@link TypeSig}s.  {@link
 * TypeDecl} objects represents the actual parse tree of a declaration.  {@link
 * TypeSig} objects refer to {@link TypeDecl} objects.  Rather than point directly to
 * {@link TypeDecl}, most references to type declarations point to {@link TypeSig}
 * objects instead.  This extra level of indirection allows us to defer parsing of
 * type declarations until the parse tree is really needed. </p>
 * 
 * <p> Details of the states of type declarations are found with documentation of the
 * {@link TypeSig} class. </p>
 * 
 * @see javafe.tc.TypeSig
 * @see javafe.tc.Env
 * @see javafe.ast.TypeDecl
 * @see javafe.ast.TypeName
 * @see javafe.ast.FieldAccess
 */

public class TypeCheck
{
    /** A (possibly extended) instance of TypeCheck. */

    //@ invariant inst != null;
    public static TypeCheck inst;

    /** Creates a instance of TypeCheck, and sets the <code>inst</code>
     * field to this instance. Only one instance should be created.
     * Also initializes {@link PrepTypeDeclaration}. */

    public TypeCheck() {
        inst = this;
        Info.out("[Init TypeCheck.inst to "+inst+"]");
        new PrepTypeDeclaration();
    }

    /** Called to obtain the algorithm for performing name resolution
     * and type checking.  By default, returns an instance of
     * {@link javafe.tc.FlowInsensitiveChecks}. */

    //@ ensures \result != null;
    public FlowInsensitiveChecks makeFlowInsensitiveChecks() {
        return new FlowInsensitiveChecks();
    }

    /** Moves <code>s</code> into the checked state.  If any of the
     supertypes of <code>s</code> are not prepped, they are prepped
     first. */

    //@ requires s != null;
    public void checkTypeSig(TypeSig s) {
        s.typecheck();
    }

    /** Moves <code>td</code> into the checked state.  If any of the
     supertypes of <code>s</code> are not prepped, they are prepped
     first. */

    //@ requires td != null;
    public void checkTypeDecl(TypeDecl td) {
        TypeSig sig = TypeSig.getSig(td);
        checkTypeSig(sig);
    }

    /** Retrieves the {@link Type} of a {@link VarInit}.  This type is
     * associated with an expression by the typechecking pass. If the
     * expression does not have an associated type, then
     * <code>Assert.fail</code> is called. */

    //@ requires e != null;
    public Type getType(VarInit e) {
        return FlowInsensitiveChecks.getType( e );
    }

    /** Retrieves the {@link Stmt} target of a {@link BranchStmt}.  This
     * {@link Stmt} may be mentioned either explicitly (as in
     * <code>break label;</code>), or implicitly (as in
     * <code>break;</code>) by the {@link BranchStmt}.  The correct
     * {@link Stmt} target is associated with the {@link BranchStmt} by
     * the typechecking pass. This type is associated with an expression
     * by the typechecking pass. If the {@link BranchStmt} does not have
     * an associated {@link Stmt} target, then <code>Assert.fail</code>
     * is called. */

    //@ requires s != null;
    public Stmt getBranchLabel(BranchStmt s) {
        return FlowInsensitiveChecks.getBranchLabel( s );
    }

    /** Retrieves the {@link TypeSig} associated with a particular
     * {@link TypeDecl}. */

    //@ requires d != null;
    //@ ensures \result != null;
    public TypeSig getSig(TypeDecl d) {
        return TypeSig.getSig( d );
    }


    /**
     * Retrieves the {@link TypeSig} associated with a particular
     * {@link TypeName}.
     *
     * Precondition: n has been resolved.
     */
    //@ ensures \result != null;
    public TypeSig getSig(/*@ non_null @*/ TypeName n) {
	return TypeSig.getSig( n );
    }

    
    public TypeSig getRawSig(/*@ non_null @*/ TypeName n) {
	return TypeSig.getRawSig( n );
    }


    /**
     * Construct a <code>String</code> listing the signature of a
     * {@link RoutineDecl}, omitting the return type and throws
     * causes if any. <p>
     *
     * All types are fully qualified if <code>r</code> has
     * been name resolved.<p>
     *
     * Sample output: "(int, javafe.tc.TypeSig, char[])" <p>
     *
     * Precondition: PrettyPrint.inst, and r non-null.<p>
     */
    //@ requires r != null;
    public static String getSignature(RoutineDecl r) {
	StringBuffer s = new StringBuffer("(");

	for (int i=0; i<r.args.size(); i++) {
	    if (i != 0)
		s.append(", ");
	    s.append(Types.printName(r.args.elementAt(i).type));
	}

	s.append(")");
	return s.toString();
    }


    /**
     * Returns the user-readable name for a {@link RoutineDecl}. <p>
     *
     * Either of the form "method <name>(<argument types>)" or the form
     * "constructor <classname>(<argument types>)".<p>
     *
     * All argument types are fully qualified if
     * <code>r</code> has been name resolved.  The method/constructor
     * name is not qualified.<p>
     *
     * Precondition: PrettyPrint.inst, and r non-null.<p>
     */
    //@ requires r.hasParent;
    public String getName(/*@ non_null @*/ RoutineDecl r) {
	String argumentTypes = getSignature(r);

	switch (r.getTag()) {
	    case TagConstants.METHODDECL:
		MethodDecl md = (MethodDecl)r;
		return "method " + md.id + argumentTypes;

	    case TagConstants.CONSTRUCTORDECL:
		ConstructorDecl cd = (ConstructorDecl)r;
		return "constructor " + cd.getParent().id
                    .toString() + argumentTypes;

	    default:
	        /*@ unreachable; */        //@ nowarn Reachable;
		Assert.fail("unreachable point");
		return null;		// keep compiler happy
	}
    }

    /**
     * Returns the user-readable simple name for a {@link RoutineDecl}. <p>
     *
     * Precondition: r non-null.<p>
     */
    //@ requires r.hasParent;
    public String getRoutineName(/*@ non_null @*/ RoutineDecl r) {
	switch (r.getTag()) {
	    case TagConstants.METHODDECL:
		MethodDecl md = (MethodDecl)r;
		return md.id.toString();

	    case TagConstants.CONSTRUCTORDECL:
		ConstructorDecl cd = (ConstructorDecl)r;
		return cd.getParent().id.toString();

	    default:
	        /*@ unreachable; */        //@ nowarn Reachable;
		Assert.fail("unreachable point");
		return null;		// keep compiler happy
	}
    }


    /**
     * Can a member of type target with modifiers
     * modifiers/pmodifiers be accessed by code located in from? <p>
     *
     * Note: pmodifiers may be null. <p>
     */
    public boolean canAccess(/*@ non_null @*/ TypeSig from, 
			     /*@ non_null @*/ TypeSig target,
			     int modifiers,
			     ModifierPragmaVec pmodifiers) {
        if (Modifiers.isPublic(modifiers))
            return true;
        if (Modifiers.isProtected(modifiers) && from.isSubtypeOf(target))
            return true;
        if (!Modifiers.isPrivate(modifiers))  // aka, protected, package
            return from.inSamePackageAs(target);

        /*
         * private case -- have same enclosing class? [1.1]:
         */
        while (from.enclosingType != null)
            from = from.enclosingType;
        while (target.enclosingType != null)
      	    target = target.enclosingType;
        return target==from;
    }

    /** Retrieves the class {@link MethodDecl} that a given class
     * {@link MethodDecl} overrides.  If there is no overridden {@link
     * MethodDecl} in a superclass, then return <code>null</code>. The
     * returned {@link MethodDecl} may be abstract. If multiple class
     * {@link MethodDecl}'s are overridden, it returns the one lowest
     * in the class hierarchy (furthest away from
     * java.lang.Object). This information is generated by the 'Prep'
     * pass. */
 
    //@ requires md.parent instanceof ClassDecl;
    public MethodDecl getOverrides(/*@ non_null @*/ MethodDecl md) {
 
        Set overrides = PrepTypeDeclaration.inst.getOverrides( md );
 
        ClassDecl cd = (ClassDecl)md.parent;
        for(;;) {
            // move cd up to parent, if any
            if( cd.superClass == null )
                return null;
 
            cd = (ClassDecl)( getSig(cd.superClass).getTypeDecl() );//@ nowarn Null,Cast;
 
            // Find MethodDecl at cd that is overridden
            Enumeration overridden_methods = overrides.elements();
            while( overridden_methods.hasMoreElements() ) {
                MethodDecl smd = (MethodDecl)overridden_methods.nextElement();
                if( smd.parent == cd )
                    return smd;
            }
        }
    }
 
    /** Retrieves the set of interface {@link MethodDecl}s that a given
     * class {@link MethodDecl} implements.  This information is
     * generated by the 'Prep' pass. */
 
    //@ requires cd != null && md != null;
    //@ requires md.parent instanceof ClassDecl;
    public Set getImplementsSet(ClassDecl cd, MethodDecl md) {
        Assert.notFalse( md.parent instanceof ClassDecl );
        Set result = new Set();
        //@ assume result.elementType == \type(MethodDecl);
 
        TypeSig sig = getSig( cd );
        sig.prep();
        Set overrides = PrepTypeDeclaration.inst.getOverrides( md );
        Enumeration overridden_methods = overrides.elements();
        while( overridden_methods.hasMoreElements() ) {
            MethodDecl smd = (MethodDecl)overridden_methods.nextElement();
            if( smd.parent instanceof InterfaceDecl 
                && sig.isSubtypeOf( getSig(smd.parent) ) )
                result.add( smd );
        }
        return result;    
    }
 
    /** Retrieves the set of interface {@link MethodDecl}s that a given
     * class {@link MethodDecl} implements.  This information is
     * generated by the 'Prep' pass. */
 
    //@ requires md != null;
    //@ requires md.parent instanceof ClassDecl;
    public Set getAllImplementsSet(MethodDecl md) {
        Assert.notFalse( md.parent instanceof ClassDecl );
        Set result = new Set();
        //@ assume result.elementType == \type(MethodDecl);
 
        Set overrides = PrepTypeDeclaration.inst.getOverrides( md );
        Enumeration overridden_methods = overrides.elements();
        while( overridden_methods.hasMoreElements() ) {
            MethodDecl smd = (MethodDecl)overridden_methods.nextElement();
            if( smd.parent instanceof InterfaceDecl )
                result.add( smd );
        }
        return result;    
    }
 
    /** Retrieves the set of interface {@link MethodDecl}s that a
     * given interface {@link MethodDecl} implements.  This
     * information is generated by the 'Prep' pass. */
 
    //@ requires md != null;
    //@ requires md.parent instanceof InterfaceDecl;
    public Set getImplementsSet(MethodDecl md) {
        Assert.notFalse( md.parent instanceof InterfaceDecl );
        return PrepTypeDeclaration.inst.getOverrides( md );
    }
 
 
 
    /** Retrieves the set of {@link MethodDecl}s that a given
     * {@link MethodDecl} overrides or hides.  This information is
     * generated by the 'Prep' pass. */
 
    //@ requires md != null;
    //@ ensures \result != null;
    //@ ensures \result.elementType == \type(MethodDecl);
    public Set getAllOverrides(MethodDecl md) {
        return PrepTypeDeclaration.inst.getOverrides( md );
    }

} // end of class TypeCheck

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
