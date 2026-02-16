/* Copyright 2000, 2001, Compaq Computer Corporation */

package instrumenter;


import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;


import javafe.reader.StandardTypeReader;
import escjava.reader.EscTypeReader;

import javafe.parser.PragmaParser;

import escjava.translate.*;

import javafe.tc.TypeSig;
import escjava.tc.TypeCheck;

import javafe.util.*;


/**
 ** Top level control module for Houdini instrumenter. <p>
 **
 ** This class (and its superclasses) handles parsing
 ** <code>instrumenter</code>'s command-line arguments and orchestrating the
 ** other pieces of the front end and instrumenter to perform the requested
 ** operations.<p>
 **
 ** @see javafe.Tool
 ** @see javafe.SrcTool
 **/

public class Main extends javafe.SrcTool {

    /** Our version number **/
    public final String version = "1.0, 16 Nov 1999";


    /***************************************************
     *                                                 *
     * Global instrumenter flags:		       *
     *                                                 *
     ***************************************************/

    public static boolean pjt  = false; // print java w/ types
    public static String toDir = "instrumented"; // the instrucmented code
  
    /*
     * These are set by the command-line switches of the same name.
     * They default to false (unset).
     */


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
    public String name() { return "instrumenter"; }

    /**
     ** Print option option information to
     ** <code>System.err</code>. <p>
     **/
    public void showOptions() {
      super.showOptions();
      System.err.println(" -pjt");
    }


    /***************************************************
     *                                                 *
     * Option processing:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Process next tool option. <p>
     **
     ** See <code>Tool.processOption</code> for the complete
     ** specification of this routine.<p>
     **/
    public int processOption(String option, String[] args, int offset) {
	// Pass on unrecognized options:
      	if (option.equals("-pjt")) {
	  pjt = true;
	  return offset;
	}
	else if (option.equals("-todir")) {
	  if (offset>=args.length) {
	    usage();
	    System.exit(usageError);
	  }
	  toDir = args[offset];
	  return offset+1;
	}
	return super.processOption(option, args, offset);
    }

    private void badOptionUsage(String s) {
      System.err.println(s);
      System.err.println();
      usage();
      System.exit(usageError);
    }

    /***************************************************
     *                                                 *
     *  Front-end setup:		               *
     *                                                 *
     ***************************************************/

    /**
     ** Returns the Esc StandardTypeReader, EscTypeReader.
     **/
    public StandardTypeReader makeStandardTypeReader(String path,
						     PragmaParser P) {
        return EscTypeReader.make(path, P);
    }

    /**
     ** Returns the EscPragmaParser.
     **/
    public javafe.parser.PragmaParser makePragmaParser() {
      return new escjava.parser.EscPragmaParser();
    }

    /**
     ** Returns the pretty printer to set
     ** <code>PrettyPrint.inst</code> to. 
     **/
    public PrettyPrint makePrettyPrint() {
      DelegatingPrettyPrint p = new InstrumenterPrettyPrint();
      p.del = new StandardPrettyPrint(p);
      return p;
    }

    /**
     ** Called to obtain an instance of the javafe.tc.TypeCheck class
     ** (or a subclass thereof). May not return <code>null</code>.  By
     ** default, returns <code>javafe.tc.TypeCheck</code>.
     **/
    public javafe.tc.TypeCheck makeTypeCheck() {
	return new escjava.tc.TypeCheck();
    }


    /**
     ** Override SrcTool.notify to ensure all lexicalPragmas get
     ** registered as they are loaded.
     **/
    public void notify(CompilationUnit justLoaded) {
	super.notify(justLoaded);

	NoWarn.registerNowarns(justLoaded.lexicalPragmas);
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
	javafe.SrcTool t = new Main();

	t.run(args);
    }


    /***************************************************
     *                                                 *
     * SrcTool-instance specific processing:	       *
     *                                                 *
     ***************************************************/


    /**
     ** Hook for any work needed before <code>handleCU</code> is
     ** called on each <code>CompilationUnit</code> to process them.
     **/
    public void preprocess() { /*SKIP*/ }

    Vector cunits = new Vector();

    /**
     ** Hook for any work needed after <code>handleCU</code> has been called
     ** on each <code>CompilationUnit</code> to process them.
     **/
    public void postprocess() {
      // instrument the code:
      for (int i = 0; i < cunits.size(); i++) {
	CompilationUnit cu = (CompilationUnit)cunits.elementAt(i);
	Instrument.instrument(cu);
      }
      // save the instrumented version:
      try {
	escjava.Main.nvu = true;
	PrettyPrint pr = makePrettyPrint();
	File toDirFile = new File(toDir);
	toDirFile.mkdirs();
	for (int i = 0; i < cunits.size(); i++) {
	  CompilationUnit cu = (CompilationUnit)cunits.elementAt(i);
	  String fname = Location.toFileName(cu.getStartLoc());
	  File file = new File(toDir, fname);
	  File pfile = new File(file.getParent());
	  pfile.mkdirs();
	  FileOutputStream fos = new FileOutputStream(file);
	  pr.print(fos, cu);
	  fos.close();
	}
	File infoFile = new File(toDir, "houdini/Info.java");
	Instrument.dumpAnnotationInfo(infoFile);
      }
      catch (IOException e) {
	throw new RuntimeException(e.getMessage());
      }
    }

    /**
     ** This method is called on each <code>CompilationUnit</code>
     ** that this tool processes.  This method overrides the implementation
     ** given in the superclass, adding a couple of lines before the
     ** superclass implementation is called.
     **/
    public void handleCU(CompilationUnit cu) {
	super.handleCU(cu);
	cunits.addElement(cu);
    }
  

    /**
     ** This method is called by SrcTool on the TypeDecl of each
     ** outside type that SrcTool is to process. <p>
     **
     ** In addition, it calls itself recursively to handle types
     ** nested within outside types.<p>
     **/
    public void handleTD(TypeDecl td) {
	TypeSig sig = TypeCheck.inst.getSig(td);
	if (sig.getTypeDecl().specOnly)	  // do not process specs
	    return;

	long startTime = java.lang.System.currentTimeMillis();

	// Do actual work:
	boolean aborted = processTD(td);

	/*
	 * Handled any nested types:  [1.1]
	 */
	TypeDecl decl = sig.getTypeDecl();
	for (int i=0; i<decl.elems.size(); i++) {
	    if (decl.elems.elementAt(i) instanceof TypeDecl)
		handleTD((TypeDecl)decl.elems.elementAt(i));
	}
    }


    /**
     ** Run all the requested stages on a given TypeDecl;
     ** return true if we had to abort. <p>
     **
     ** Precondition: td is not from a binary file.<p>
     **/
    private boolean processTD(TypeDecl td) {
	// ==== Start stage 1 ====

	/*
	 * Do Java type checking then print Java types if we've been
	 * asked to:
	 */
	int errorCount = ErrorSet.errors;
	TypeSig sig = TypeCheck.inst.getSig(td);
	sig.typecheck();


	if (pjt) {
	    // Create a pretty-printer that shows types
	    DelegatingPrettyPrint p = new javafe.tc.TypePrint();
	    p.del = new InstrumenterPrettyPrint(p, new StandardPrettyPrint(p));

	    System.out.println("\n**** Source code with types:");
	    p.print(System.out, 0, td);
	}

	// Turn off instrumentation and abort if any errors
	// occured while type checking *this* TypeDecl:
	if (errorCount < ErrorSet.errors) {
	    ErrorSet.fatal("Abort due to type errors.");
	    return true;
	}

	return false;
    }


    /**
     ** Run stages 3+..6 as requested on a TypeDeclElem. <p>
     **
     ** Precondition: te is not from a binary file,
     **		      sig is the TypeSig for te's parent, and
     **		      initState != null.<p>
     **/
    private void processTypeDeclElem(TypeDeclElem te, TypeSig sig,
				     InitialState initState) {

    }


    /***************************************************
     *                                                 *
     * Misc. Utility routines:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Compute the time used from a start time to now, then return it
     ** in a user readable form.
     **/

    //@ ensures \result != null;
    String timeUsed(long startTime) {
	long delta = java.lang.System.currentTimeMillis() - startTime;

	return (delta/1000.0) + " s";
    }


}


