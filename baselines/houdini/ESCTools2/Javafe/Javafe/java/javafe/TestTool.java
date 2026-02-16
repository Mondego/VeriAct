/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe;


import javafe.ast.*;
import javafe.tc.*;

import javafe.util.*;


/**
 * <code>TestTool</code> is an test class for <code>SrcTool</code> and
 * its superclasses. <p>
 *
 * It just loads in files unless <code>-superclasses</code> is
 * supplied, in which case it chases down each classes' superclass
 * chain.<p>
 *
 * Note: because this class is intended as a test class routine, the 
 * code for chasing superclass pointers is kludgey and does not always
 * work correctly.
 */

public class TestTool extends SrcTool {

    /***************************************************
     *                                                 *
     * Generating a usage message:		       *
     *                                                 *
     **************************************************/

    /**
     * Return the name of this tool.  E.g., "ls" or "cp".<p>
     *
     * Used in usage and error messages.<p>
     */
    public String name() { return "TestTool"; }
    
    public /*@non_null*/javafe.Options makeOptions() { return new Options(); }
    
    //+@ requires options != null;
    public final /*@non_null*/javafe.TestTool.Options options() { //@ nowarn NonNullResult;
    	return (/*+@non_null*/ Options)options; //@ nowarn Cast;
    }

    public class Options extends SrcToolOptions {
	    /**
	     * Print option information to <code>System.err</code>.  Each
	     * printed line should be preceded by two blank spaces. <p>
	     *
	     * Each overriding method should first call
	     * <code>super.showOptions()</code>.<p>
	     */
	    public String showOptions(boolean all) {
	        return super.showOptions(all) + ("  -superclasses") + eol;
	    }
	
	
	    /***************************************************
	     *                                                 *
	     * Option processing:			       *
	     *                                                 *
	     **************************************************/
	
	    /**
	     * Should we chase superclass pointers?  Defaults to no, set by
	     * <code>-superclasses</code>.
	     */
	    public boolean chaseSuperclasses = false;
	
	
	/**
	 * Process next tool option. <p>
	 *
	 * See <code>Tool.processOption</code> for the complete
	 * specification of this routine.<p>
	 *
	 * This routine handles the <code>-superclasses</code> option.<p>
	 */
	public int processOption(/*@non_null*/String option, /*@non_null*/String[/*#@non_null*/] args, int offset) 
                                     throws UsageError {
	    if (option.equals("-superclasses")) {
		chaseSuperclasses = true;
		return offset;
	    }
    
	    // Pass on unrecognized options:
	    return super.processOption(option, args, offset);
	}
	    
    }


    /***************************************************
     *                                                 *
     * Main processing code:			       *
     *                                                 *
     **************************************************/

    /**
     * Start up an instance of this tool using command-line arguments
     * <code>args</code>. <p> 
     *
     * <b>Note</b>: this code needs to be copied verbatim to each
     * subclass of <code>Tool</code> except with the name of the actual
     * subclass inserted after the new operator.<p>
     *
     * (This needs to be done because static methods cannot be
     * inherited.)<p>
     */
    //@ requires \nonnullelements(args);
    //@ diverges true;
    public static void main(/*@non_null*/String[/*#@non_null*/] args) {
		Tool t = new TestTool();
		int result = t.run(args);
		if (result != 0) System.exit(result);
    } //@ nowarn Post;


    /***************************************************
     *                                                 *
     * SrcTool-instance specific processing:	       *
     *                                                 *
     **************************************************/

    /**
     * This method is called on the TypeDecl of each
     * outside type that SrcTool is to process. <p>
     */
    //@ also requires options != null;
    public void handleTD(/*@non_null*/TypeDecl td) {
		Info.out("[processing "
			+ TypeSig.getSig(td).getExternalName());
	
		if (!options().chaseSuperclasses)
		   return;
	
		TypeSig current = TypeSig.getSig(td);
		while (current != null) {
		    Info.out("  At " + current.getExternalName());
	
		    current = getSuperClass(current.getTypeDecl());
		}
	
		Info.out("]");
    }


    /**
     * Attempt to fetch the <code>TypeSig</code> for a given
     * <code>TypeDecl</code>. <p>
     * 
     * Returns null if none exists.<p>
     */
    public /*@nullable*/TypeSig getSuperClass(TypeDecl td) {
		// If  td is not a class, then it has no superclass:
		if (!(td instanceof ClassDecl))
		    return null;
		ClassDecl cd = (ClassDecl)td;
	
		// Get td's superclass name:
		TypeSig root = Types.javaLangObject();
		if (TypeSig.getSig(td) == root)
		    return null;
		if (cd.superClass == null) {
		    Info.out("    extended by java.lang.Object");
		    return root;
		}
	
		Name superClassName = cd.superClass.name;
		Info.out("    extended by " + superClassName.printName());
	
		int sz = superClassName.size();
		String[] P = superClassName.toStrings(sz-1);
		String   T = superClassName.identifierAt(sz-1).toString();
	
		if (sz == 1) {
		    // Simple name -- assume for now in same package:
		    // This is a bit of a hack:
		    TypeSig sig = TypeSig.getSig(td);
		    P = sig.packageName;
		}
	
		TypeSig S = OutsideEnv.lookup(P, T); //@ nowarn Pre;
		if (S == null)
		    ErrorSet.error("unable to load type "
					+ superClassName.printName());
		return S;
    }
}
