/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;
import javafe.util.*;


/**
 * This class is responsible for transitioning TypeSigs to the
 * LINKSRESOLVED state from an earlier state.<p>
 *
 *
 * So transitioning sig involves the following:
 *
 *   (1) Ensuring that sig's CompilationUnit is checked.  (See
 *       CheckCompilationUnit)
 *
 *   (2) Resolving all sig's supertype names (see
 *       Env.resolveTypeName); these are the names mentioned in sig's
 *       TypeDecl's extends and implements clauses (if any).
 *
 *   (3) Handling missing extends clauses for classes appropriately
 *       (E.g., inserting a reference to java.lang.Object, or null if (4))
 *
 *   (4) If sig is java.lang.Object, then we always set its superClass
 *       field to null.  We check to make sure that the user did
 *       not try to make us extend a class other than us.
 *
 *   (5) Transitioning all of sig's direct supertypes (the types
 *       resolved above) to at least the LINKSRESOLVED state.
 *
 *   (6) Detecting loops in the inheritance/enclosing graph that
 *       would otherwise cause (5) to hang.
 *
 * It is a fatal error if we cannot resolve a supertype name or if we
 * detect a cycle.
 *
 *
 * Notes:
 *
 *   - Checking whether sig's direct supertypes are accessible from sig
 *     is now done by the "prepping" stage.  (This is necessary
 *     because access checking requires calling isSubTypeOf.)
 *
 *   - No TypeSig is ever advanced beyond LINKSRESOLVED by this
 *     process.  This fact depends on the control flow elsewhere in the
 *     javafe.tc package.  In particular, Env.resolveTypeName obeys
 *     this requirement.
 *
 *   - Exactly which TypeSigs will be moved to at least the
 *     LINKSRESOLVED stage is tricky to compute in advance because
 *     TypeName resolution may itself require enclosing TypeSigs to be
 *     transitioned to at least the LINKSRESOLVED stage.
 *
 *   - In general, we try to transition to smallest number of
 *     TypeSigs we can to maximize the number of legal programs.
 *     (results in fewer cycles detected.)
 *
 * @see TypeSig
 * @see Env
 * @see CheckCompilationUnit
 */

public class SLResolution {

    /**
     * Transition <code>sig</code> to the supertype-links-resolved
     * state. <p>
     *
     * See the SLResolution type's overall comments for details of
     * what this involves.<p>
     *
     * A fatal error may be reported if we cannot resolve a supertype
     * name, or detect a cycle in the type hierarchy.<p>
     */
    //@ requires sig.state < TypeSig.LINKSRESOLVED;
    //@ modifies sig.state;
    //@ ensures sig.state==TypeSig.LINKSRESOLVED;
    /*package*/ public static void transition(/*@ non_null @*/ TypeSig sig) {
	/*
	 * First, check to see if we've looped:
	 */
	TypeDecl d = sig.getTypeDecl();
	if (sig.state==TypeSig.RESOLVINGLINKS)
	    ErrorSet.fatal(d.locId,
			   "Cycle in inheritance/enclosing hierarchy "
			   + "detected involving type " + sig);

	Info.out("[Superlink resolving " + sig + "]");
	sig.state = TypeSig.RESOLVINGLINKS;


	/*
	 * Do CompilationUnit-level checks first to make sure that
	 * import statements are ok:
	 */
        CheckCompilationUnit.checkCompilationUnit(sig.getCompilationUnit());


	/*
	 * Call handleSuperTypeName on each of our supertype names.
	 *
	 * Fixup a null superclass pointer into a reference to
	 * java.lang.Object unless sig is java.lang.Object.
	 */
	for (int i=0; i<d.superInterfaces.size(); i++) 
	    handleSuperTypeName(sig, d.superInterfaces.elementAt(i));

	if (d.getTag() == TagConstants.CLASSDECL) {
	    ClassDecl cd = (ClassDecl)d;

	    if (cd.superClass==null) {
		cd.superClass = TypeName.make(Name.make(//@ nowarn Pre;//strlen
							"java.lang.Object", 
							cd.locOpenBrace));
		TypeSig.setSig(cd.superClass, Types.javaLangObject());
	    }

	    TypeSig superClass = handleSuperTypeName(sig, cd.superClass);

	    // Handle java.lang.Object specially:
	    if (sig==Types.javaLangObject()) {
		if (superClass != sig)
		    ErrorSet.fatal(d.loc, "Only java.lang.Object may be "
				   + "the superclass of java.lang.Object");

		// Leave TypeDecl in canonical form:
		cd.superClass = null;
	    }
	}

	Info.out("[Superlink resolved " + sig + "]");
	sig.state = TypeSig.LINKSRESOLVED;
    }


    /**
     * Handle a super type name. <p>
     *
     * We resolve the typename (a fatal error results if we cannot)
     * then transition the resulting type to at least the
     * LINKSRESOLVED stage.  We return the TypeSig for the
     * supertype.<p>
     *
     * sig is the TypeSig that superName names a supertype for.<p>
     *
     * Exception: if sig is java.lang.Object, then we do not try and
     * transition its superclass.<p>
     */
    public static TypeSig handleSuperTypeName(
			       /*@ non_null @*/ TypeSig sig,
			       /*@ non_null @*/ TypeName superName) {
	TypeSig supertype = sig.getEnclosingEnv().resolveTypeName(null,superName);

	if (supertype.state < TypeSig.LINKSRESOLVED &&
	    sig != Types.javaLangObject())
	    transition(supertype);

	return supertype;
    }
}
