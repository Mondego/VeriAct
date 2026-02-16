/* Copyright 2000, 2001, Compaq Computer Corporation */

package tohtml;

/*
 * Change history:
 * 12 Sep 2000  flanagan           Created

*/

import java.util.Enumeration;
import java.util.Hashtable;

import javafe.ast.*;

import javafe.reader.StandardTypeReader;
import escjava.reader.EscTypeReader;

import javafe.parser.PragmaParser;

import javafe.tc.TypeSig;
import javafe.tc.TypeCheck;

import javafe.util.*;


public class DeclLinks extends javafe.SrcTool {

     /** Our version number **/
     public static final String version = "1.0.0, 12 Sep 2000";


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
    public String name() { return "decl linker"; }

    /**
     ** Print option option information to
     ** <code>System.err</code>. <p>
     **/
    public void showOptions() {
      super.showOptions();
      System.err.println("");
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
	return super.processOption(option, args, offset);
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
    public StandardTypeReader makeStandardTypeReader(String path,
						     PragmaParser P) {
        return EscTypeReader.make(path, P);
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
	javafe.SrcTool t = new DeclLinks();
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

      DeclLinkVisitor v = new DeclLinkVisitor();
      td.accept(v);
    }
}

class DeclLinkVisitor extends DefaultVisitor {

  public void visitVariableAccess(VariableAccess x) {
      super.visitVariableAccess(x);
      report(x.loc, x.decl.locId);
  }

  public void visitFieldAccess(FieldAccess x) {
      super.visitFieldAccess(x);
      report(x.locId, x.decl.locId);
  }

  public void visitConstructorInvocation(ConstructorInvocation x) {
      super.visitConstructorInvocation(x);
      report(x.locKeyword, x.decl.locId);
  }

  public void visitNewInstanceExpr(NewInstanceExpr x) {
      super.visitNewInstanceExpr(x);
      report(x.loc, x.decl.locId);
  }


    // Put hyperlink on class method decls to the method they override
  public void visitMethodDecl(MethodDecl x) {
      super.visitMethodDecl(x);
      Set overrides = TypeCheck.inst. getAllOverrides(x);
      Enumeration e = overrides.elements();
      if( e.hasMoreElements() ) {
	  MethodDecl md = (MethodDecl)e.nextElement();
	  if( e.hasMoreElements() ) {
	      // Overrides multiple methods, put dummy link 
	      if( x.locId != Location.NULL && !Location.isWholeFileLoc(x.locId)) {
		  System.out.println(Location.toFileName(x.locId)+
				     " "+Location.toOffset(x.locId)+
				     " OverridesMultipleMethods 0");
	      }
	  } else {
	      // Overrides single method, link to it
	      report( x.locId, md.loc );
	  }
      } else {
	  // Overrides no method, no link
	      // Overrides multiple methods, put dummy link 
	      if( x.locId != Location.NULL && !Location.isWholeFileLoc(x.locId)) {
		  /*
		  System.out.println(Location.toFileName(x.locId)+
				     " "+Location.toOffset(x.locId)+
				     " OverridesNoMethods 0");
		  */
	      }
      }
  }

  public void visitMethodInvocation(MethodInvocation x) {
      super.visitMethodInvocation(x);
      report(x.locId, x.decl.locId);
  }

  public void visitTypeName(TypeName x) {
      super.visitTypeName(x);
      report(x.name.locIdAt(x.name.size()-1),
	     TypeSig.getSig(x).getTypeDecl().locId);
  }

/*
  public void visit( x) {
      super.visit(x);
      report(x.locKeyword, x.decl.locId);
  }
    */

    private void report(int locref, int locdecl) {
	if( locref != Location.NULL && !Location.isWholeFileLoc(locref) ) {
	    if( locdecl == Location.NULL ) {
		System.out.println(Location.toFileName(locref)+
				   " "+Location.toOffset(locref)+
				   " LinktoNullLoc 0");
	    } else if( Location.isWholeFileLoc(locdecl) ) {
	    System.out.println(Location.toFileName(locref)+
			       " "+Location.toOffset(locref)+
			       " "+Location.toFileName(locdecl)+" -1");
	    } else {
		System.out.println(Location.toFileName(locref)+
				   " "+Location.toOffset(locref)+
				   " "+Location.toFileName(locdecl)+
				   " "+Location.toLineNumber(locdecl));
	    }
	}
    }

    private String convert(int loc) {
	return Location.toFileName(loc)+
	    " "+Location.toLineNumber(loc)+
	    " "+Location.toColumn(loc);
    }
}

