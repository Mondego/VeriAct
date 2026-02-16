/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.test;


import java.util.Enumeration;
import java.util.Vector;

import javafe.*;
import javafe.ast.*;
import javafe.tc.*;
import javafe.parser.*;
import javafe.util.*;


/**
 ** <code>LocTool</code> is a <code>SrcTool</code> that parses input
 ** file(s) and displays the location pointers of all
 ** <code>Expr</code>'s in the resulting ASTs.
 **/

public class LocTool extends SrcTool {

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
    public String name() { return "LocTool"; }


    /***************************************************
     *                                                 *
     * Option processing:			       *
     *                                                 *
     ***************************************************/


    /***************************************************
     *                                                 *
     * Main processing code:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Start up an instance of this tool using command-line arguments
     ** <code>args</code>. <p> 
     **
     **/
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
		Tool t = new LocTool();
		int result = t.run(args);
		if (result != 0) System.exit(result);
    }


    /***************************************************
     *                                                 *
     * SrcTool-instance specific processing:	       *
     *                                                 *
     ***************************************************/

    /**
     ** This method is called on each <code>CompilationUnit</code>
     ** that this tool processes. <p>
     **/
    public void handleCU(CompilationUnit cu) {
		System.out.println("=== File: " + Location.toFileName(cu.loc)
				 +" ===");
		super.handleCU(cu);
    }


    /**
     ** This method is called on the TypeDecl of each
     ** outside type that SrcTool is to process. <p>
     **/
    public void handleTD(TypeDecl td) {
		findNodes(td);
    }


    //@ requires n != null;
    public void findNodes(ASTNode n) {
		handleNode(n);
		for (int i=0; i<n.childCount(); i++) {
		    Object child = n.childAt(i);
		    if (child instanceof ASTNode)
			findNodes((ASTNode)child);
		}
    }


    public void handleNode(ASTNode n) {
		if (n instanceof Expr) {
		    Expr e = (Expr)n;
	
		    System.out.print("EXPR <<");
		    PrettyPrint.inst.print(System.out, 0, e);
		    int l = e.getStartLoc();
		    System.out.println(">> @ " + Location.toString(l));
		    if (l != 0) javafe.util.ErrorSet.displayColumn(l);
		    System.out.println();
		}
    }
}
