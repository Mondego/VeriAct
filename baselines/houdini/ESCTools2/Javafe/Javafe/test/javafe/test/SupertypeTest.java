/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.test;


import java.util.Vector;
import java.util.ArrayList;

import javafe.*;
import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


/**
 ** <code>SupertypeTest</code> is a <code>FrontEndTool</code> that tests
 ** the supertype relation code.
 **/

public class SupertypeTest extends javafe.FrontEndTool {

    /***************************************************
     *                                                 *
     * Generating a usage message:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Return the name of this tool.  E.g., "ls" or "cp".<p>
     **
     ** Used in usage and error messages.<p>
     **/
    public String name() { return "SupertypeTest"; }

    public javafe.Options makeOptions() { return new Options(); }
    
    public class Options extends javafe.Options {
    
	    /**
	     ** Print non-option usage info . <p>
	     **/
	    public String showNonOptions() {
			return ("(<reference type name> <reference type name>)...");
	    }
	
    }
    

    /***************************************************
     *                                                 *
     * Main processing code:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Start up an instance of this tool using command-line arguments
     ** <code>args</code>. <p> 
     **/
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
		Tool t = new SupertypeTest();
		int result = t.run(args);
		if (result != 0) System.exit(result);
    }


    /**
     ** Main processing loop for SupertypeTest.<p>
     **
     ** The remaining arguments are <code>args[offset]</code>,
     ** <code>args[offset+1]</code>, ...<p>
     **/
    public void frontEndToolProcessing(ArrayList argsv) {
		String[] args = (String[])argsv.toArray(new String[0]);
		int left = args.length;	
	
		// Must have an positive even # of type names:
		if (left % 2 != 0 || left==0)
		    usage();
	
		int i = 0;
		// Handle each query:
		//@ loop_invariant left == args.length - i;
		for (; left>1; left-=2, i+=2) {
		    query(args[i], args[i+1]);
		}
    }


    /** Handle a subtyping query **/
    //@ requires left != null && right != null;
    public void query(String left, String right) {
		// Attempt to find the types:
		TypeSig leftType = getType(left);
		TypeSig rightType = getType(right);
	
		// Do nothing if one of the two types not found:
		if (leftType==null || rightType==null)
		    return;
	
		if (leftType.isSubtypeOf(rightType))
		    System.out.println(leftType.getExternalName() + " <: "
			+ rightType.getExternalName());
		else if (rightType.isSubtypeOf(leftType))
		    System.out.println(rightType.getExternalName() + " <: "
			+ leftType.getExternalName());
		else
		    System.out.println(leftType.getExternalName() + " ? "
			+ rightType.getExternalName());
    }
	
	
    /***************************************************
     *                                                 *
     * Utility routines:			                   *
     *                                                 *
     ***************************************************/
	
    /**
     ** Return the TypeSig for fully-qualified package-member type N
     ** or null if no such type exists.<p>
     **
     ** An error will be reported via ErrorSet in the later case.<p>
     **/
    //@ requires N != null;
    public TypeSig getType(String N) {
		TypeSig result;
	
		// Convert N to a list of its components:
		String[] components = javafe.filespace.StringUtil.parseList(N, '.');
		Assert.notFalse(components.length>0);	//@ nowarn Pre;
	
		// Split components into P and T:
		String[] P = new String[components.length-1];
		for (int i=0; i<P.length; i++)
		    P[i] = components[i];
		String T = components[components.length-1];
	
		result = OutsideEnv.lookup(P, T);
		if (result==null)
		    javafe.util.ErrorSet.error(N + ": no such class or interface");
	
		return result;
    }
}
