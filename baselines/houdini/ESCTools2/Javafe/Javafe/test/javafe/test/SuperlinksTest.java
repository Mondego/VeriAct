/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.test;

import java.util.Vector;

import javafe.*;
import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


/**
 ** <code>SuperlinksTest</code> is a <code>SrcTool</code> that tests the
 ** super-link resolution code.
 **/

public class SuperlinksTest extends javafe.SrcTool {

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
    public String name() { return "SuperlinksTest"; }


    /***************************************************
     *                                                 *
     * Main processing code:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Start up an instance of this tool using command-line arguments
     ** <code>args</code>. <p> 
     **
     ** <b>Note</b>: this code needs to be copied verbatim to each
     ** subclass of <code>Tool</code> except with the name of the actual
     ** subclass inserted after the new operator.<p>
     **
     ** (This needs to be done because static methods cannot be
     ** inherited.)<p>
     **/
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
		Tool t = new SuperlinksTest();
		int result = t.run(args);
		if (result != 0) System.exit(result);
    }

    SuperlinksTest() {
      //@ set decls.owner = this;
      //@ set decls.elementType = \type(TypeDecl);
      //@ set decls.containsNull = false;
    }

    /***************************************************
     *                                                 *
     * SrcTool-instance specific processing:	       *
     *                                                 *
     ***************************************************/

    /**
     ** This vector collects all the <code>TypeDecls</code> we have
     ** processed so far. 
     **/
    //@ invariant decls != null;
    //@ invariant decls.elementType == \type(TypeDecl);
    //@ invariant !decls.containsNull;
    //@ invariant decls.owner == this;
    Vector decls = new Vector(10);


    /**
     ** Hook for any work needed after <code>handleCU</code> has been called
     ** on each <code>CompilationUnit</code> to process them.
     **/
    public void postprocess() {
		// Make sure invariants haven't been broken
		for (int k=0; k<decls.size(); k++) {
		    TypeDecl decl = (TypeDecl)decls.elementAt(k);
	
		    TypeCheck.inst.getSig(decl).check();
		}
    }


    /**
     ** This method is called on the <code>TypeDecl</code> of each
     ** outside type that we are to process. <p>
     **
     ** Each <code>TypeDecl</code> is added to <code>decls</code>, and
     ** then has its supertype links resolved.
     **/
    public void handleTD(TypeDecl td) {
		decls.addElement(td);
	
		TypeCheck.inst.getSig(td).resolveSupertypeLinks();
    }
}
