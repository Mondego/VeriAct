/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;


import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;

import javafe.Options;
import javafe.SrcToolOptions;

import javafe.ast.*;

import javafe.reader.StandardTypeReader;
import escjava.reader.EscTypeReader;

import javafe.parser.PragmaParser;

import javafe.tc.TypeSig;
import javafe.tc.TypeCheck;

import javafe.util.*;

/**
 ** Top level control module for ESC for Java. <p>
 **
 ** This class (and its superclasses) handles parsing
 ** <code>escjava</code>'s command-line arguments and orchestrating the
 ** other pieces of the front end and escjava to perform the requested
 ** operations.<p>
 **
 ** @see javafe.Tool
 ** @see javafe.SrcTool
 **/

public class Main extends javafe.SrcTool {

     /** Our version number **/
     public final String version = "1.0.1, 26 May 2000";


    /***************************************************
     *                                                 *
     * Generating an options message:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Return the name of this tool.  E.g., "ls" or "cp".<p>
     **
     ** Used in usage and error messages.<p>
     **/
    public String name() { return "houdini annotation generator"; }

  
    /***************************************************
     *                                                 *
     * Option processing:			       *
     *                                                 *
     ***************************************************/
    public Options makeOptions() {
        return new HoudiniOptions();
    }

    private static HoudiniOptions options() {
        return (HoudiniOptions) options;
    }

    /**
     ** This method is called on each <code>CompilationUnit</code>
     ** that this tool processes.  This method overrides the implementation
     ** given in the superclass, adding a couple of lines before the
     ** superclass implementation is called.
     **/
    public void handleCU(CompilationUnit cu) {
	//System.out.println ("handleCU"+cu );
	super.handleCU(cu);
    }
  
    /***************************************************
     *                                                 *
     *  Front-end setup:		               *
     *                                                 *
     ***************************************************/

    /**
     ** Returns the Esc StandardTypeReader, EscTypeReader.
     **/
    public StandardTypeReader makeStandardTypeReader(String classpath,
						     String sourcepath,
						     PragmaParser P) {
        // parallel to code in Escjava/java/escjava/Main.java
        return escjava.reader.EscTypeReader.make(classpath, sourcepath, P, new escjava.AnnotationHandler ());
    }


    /***************************************************
     *                                                 *
     * Main processing code:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Start up an instance of this tool using command-line arguments
     ** <code>args</code>. <p> 
     **
     ** This is the main entry point for the <code>escjava</code>
     ** command.<p>
     **/
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
	//System.out.println ("main,"+args[2]);
	javafe.SrcTool t = new Main();

	t.run(args);
    }


    /***************************************************
     *                                                 *
     * SrcTool-instance specific processing:	       *
     *                                                 *
     ***************************************************/

    /**
     ** This method is called by SrcTool on the TypeDecl of each
     ** outside type that SrcTool is to process. <p>
     **
     ** In addition, it calls itself recursively to handle types
     ** nested within outside types.<p>
     **/
    public void handleTD(TypeDecl td) {
	//System.out.println ("handleTD");
	
      TypeSig sig = TypeCheck.inst.getSig(td);
      sig.typecheck();
      typeDecls.addElement(td);
    }

    Vector typeDecls = new Vector();

    /**
     ** Hook for any work needed after <code>handleCU</code> has been called
     ** on each <code>CompilationUnit</code> to process them.
     **/
    public void postprocess() {

	InferThrows.compute(typeDecls);
	AnnotationVisitor v = new AnnotationVisitor(options().ag);
	v.visitTypeDeclVec(typeDecls);
    }

}

class HoudiniOptions extends SrcToolOptions
{
    AnnotationGuesser ag = new StandardAnnotationGuesser();
    public static boolean ignoreBodies = false;

    /**
     ** Process next option. <p>
     **
     ** See <code>Options.processOption</code> for the complete
     ** specification of this routine.<p>
     **/
    public int processOption(String option, String[] args, int offset) throws javafe.util.UsageError {
	//System.out.println ("processOption"+option+offset);
	if (option.equals("-guesser")) {
	    if (offset == args.length) {
		usage("Houdini");
		throw (new javafe.util.UsageError("Unknown UsageError"));
	    }
	    if (args[offset].equals("Standard")) {
	      ag = new StandardAnnotationGuesser();
	    } else if (args[offset].equals("Common")) {
	      ag = new CommonAnnotationGuesser();
	    } else if (args[offset].equals("ReqFalse")) {
	      ag = new ReqFalseAnnotationGuesser();
	    } else if (args[offset].equals("Library")) {
	      ag = new LibraryAnnotationGuesser();
	    } else if (args[offset].equals("Swing")) {
	      ag = new SwingGuesser(false);
	    } else if (args[offset].equals("SwingWithReqFalse")) {
	      ag = new SwingGuesser(true);
	    } else if (args[offset].equals("MoreReachable")) {
	      ag = new MoreReachableAnnotationGuesser();
	    } else {
	      //System.err.println("Unknown guesser '" + args[offset] + "'");
	      usage("Houdini");
	      //System.exit(usageError);
              throw (new javafe.util.UsageError("Unknown guesser '" + args[offset] + "'"));
	    }
	    return offset+1;
	} else if (option.equals("-ignoreBodies")) {
	  javafe.tc.FlowInsensitiveChecks.dontAddImplicitConstructorInvocations = true;
	  ignoreBodies = true;
	  return offset;
	} else {
	    // Pass on unrecognized options:
	    return super.processOption(option, args, offset);
	}
    }

    /**
     ** Print option option information to
     ** <code>System.err</code>. <p>
     **/
    public String showOptions(boolean all) {
      StringBuffer sb = new StringBuffer(super.showOptions(all));
      String[][] data1 = {{"-common", "TODO"}};
      sb.append(data1);
      return sb.toString();
    }
}
