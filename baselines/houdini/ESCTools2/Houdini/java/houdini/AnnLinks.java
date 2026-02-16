/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

/*
 * Change history:
 * 28 Sep 2000  flanagan           Created

*/

import java.util.Enumeration;
import java.util.Hashtable;
import java.io.ByteArrayOutputStream;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;

import javafe.reader.StandardTypeReader;
import escjava.reader.EscTypeReader;
import escjava.ast.EscPrettyPrint;

import javafe.parser.PragmaParser;

import javafe.tc.TypeSig;
import javafe.tc.TypeCheck;

import javafe.util.*;


public class AnnLinks extends javafe.SrcTool {

     /** Our version number **/
     public final String version = "1.0.0, 28 Sep 2000";


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
    public String name() { return "annotation linker"; }

    /**
     ** Print option option information to
     ** <code>System.err</code>. <p>
     **/
    public void showOptions() {
      options.showOptions(false);
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
    public int processOption(String option, String[] args, int offset) throws javafe.util.UsageError {

	// Pass on unrecognized options:
	return options.processOption(option, args, offset);
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
     *  Front-end setup: Use ESC stuff	               *
     *                                                 *
     ***************************************************/

    /**
     ** Returns the EscPragmaParser.
     **/
    public javafe.parser.PragmaParser makePragmaParser() {
      return new escjava.parser.EscPragmaParser();
    }

    /**
     ** Returns the Esc StandardTypeReader, EscTypeReader.
     **/
    public StandardTypeReader makeStandardTypeReader(String classpath,
						     String sourcepath,
						     PragmaParser P) {
        // parallel to code in Escjava/java/escjava/Main.java
        return escjava.reader.EscTypeReader.make(classpath, sourcepath, P, new escjava.AnnotationHandler ());
    }

    /**
     ** Returns the pretty printer to set
     ** <code>PrettyPrint.inst</code> to.
     **/
    public PrettyPrint makePrettyPrint() {
      DelegatingPrettyPrint p = new EscPrettyPrint();
      p.setDel(new StandardPrettyPrint(p));
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
	javafe.SrcTool t = new AnnLinks();
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

      AnnLinkVisitor v = new AnnLinkVisitor();
      td.accept(v);
    }
}

/** this visitor writes to stdout a table linking error locations to decl locations, 
 *  and decl locations to corresponding non_null annotations. 
 *
 *  Just does non_null on fields and parameters,
 *  and links from assignment statements to decl for rhs.
 */

class AnnLinkVisitor extends DefaultVisitor {

    public void visitModifierPragma(ModifierPragma x) {
	super.visitModifierPragma(x);
	ByteArrayOutputStream baos = new ByteArrayOutputStream();
	javafe.ast.PrettyPrint.inst.print(baos, 0, x);
	System.out.println("AnnText "+convert(x.getStartLoc())+" "+baos.toString());
    }

    public void visitTypeDeclElemPragma(TypeDeclElemPragma x) {
	super.visitTypeDeclElemPragma(x);
	ByteArrayOutputStream baos = new ByteArrayOutputStream();
	javafe.ast.PrettyPrint.inst.print(baos, 0, x);
	System.out.println("AnnText "+convert(x.getStartLoc())+" "+baos.toString());
    }

    private ModifierPragma searchModifierPragmas(ModifierPragmaVec x, String contains) {
	if( x != null ) {
	    for(int i=0; i<x.size(); i++) {
		ModifierPragma mp = x.elementAt(i);
		ByteArrayOutputStream baos = new ByteArrayOutputStream();
		javafe.ast.PrettyPrint.inst.print(baos, 0, mp);
		if( baos.toString().indexOf(contains) >= 0 )
		    return mp;
	    }
	}
	return null;    
    }

    private void reportNonNullModifier(GenericVarDecl x) {
	if( x.pmodifiers != null ) 
	    for(int i=0; i<x.pmodifiers.size(); i++)
		visitModifierPragma(x.pmodifiers.elementAt(i));
	ModifierPragma mp = searchModifierPragmas(x.pmodifiers, "non_null");
	if( mp != null ) {
	    reportDeclAnn( x.getStartLoc(), mp.getStartLoc());
	}
    }

    public void visitFieldDecl(FieldDecl x) {
	super.visitFieldDecl(x);
	/* todo: invariant x != null;
	for(int i=0; i<x.parent.elems.size(); i++) {
	    TypeDeclElem e = x.parent.elems.elementAt(i);
	    if( e instanceof ExprDeclPragma 
	*/
	reportNonNullModifier(x);
	System.out.println("DeclName "+convert(x.getStartLoc())+" field "+x.id);

    }

    public void visitRoutineDecl(RoutineDecl x) {
	super.visitRoutineDecl(x);
	if( x.pmodifiers != null ) 
	    for(int i=0; i<x.pmodifiers.size(); i++)
		visitModifierPragma(x.pmodifiers.elementAt(i));
	ModifierPragma mp = searchModifierPragmas(x.pmodifiers, "ensures \\result != null");
	if( mp != null ) {
	    reportDeclAnn( x.getStartLoc(), mp.getStartLoc());
	}
	if( x instanceof MethodDecl ) {
	    System.out.println("DeclName "+convert(x.getStartLoc())+" method "+((MethodDecl)x).id);
	} else {
	    System.out.println("DeclName "+convert(x.getStartLoc())+" constructor "+x.parent.id);
	}

    }

    public void visitFormalParaDecl(FormalParaDecl x) {
	super.visitFormalParaDecl(x);
	reportNonNullModifier(x);
	System.out.println("DeclName "+convert(x.getStartLoc())+" parameter "+x.id);
    }

    private void visitArgList(ExprVec args) {
	for(int i=0; i<args.size(); i++) {
	    ASTNode r = getDecl( args.elementAt(i) );
	    if( r != null ) {
		reportWarnDecl( args.elementAt(i).getStartLoc(), r.getStartLoc() );
	    }
	}
    }

    public void visitNewInstanceExpr(NewInstanceExpr x) {
	super.visitNewInstanceExpr(x);
	visitArgList(x.args);
    }

    public void visitMethodInvocation(MethodInvocation x) {
      super.visitMethodInvocation(x);
	visitArgList(x.args);
    }

    public void visitConstructorInvocation(ConstructorInvocation x) {
	super.visitConstructorInvocation(x);
	visitArgList(x.args);
    }
    



    public void visitBinaryExpr(BinaryExpr x) {
	super.visitBinaryExpr(x);
	if( x.getTag() == TagConstants.ASSIGN ) {
	    ASTNode r = getDecl( x.right );
	    if( r != null ) {
		reportWarnDecl( x.locOp, r.getStartLoc() );
	    }
	}
    }

    public void visitReturnStmt(ReturnStmt x) {
	super.visitReturnStmt(x);
	if( x.expr != null ) {
	    ASTNode r = getDecl( x.expr );
	    if( r != null ) {
		reportWarnDecl( x.loc, r.getStartLoc() );
	    }
	}
    }

  public void visitExprObjectDesignator(ExprObjectDesignator x) {
      super.visitExprObjectDesignator(x);
      ASTNode r = getDecl( x.expr );
      if( r != null ) {
	  reportWarnDecl( x.locDot, r.getStartLoc() );
      }
  }

    private ASTNode getDecl(Expr e) {
	switch( e.getTag() ) {
	case TagConstants.FIELDACCESS:
	    return ((FieldAccess)e).decl;
	case TagConstants.METHODINVOCATION:
	    return ((MethodInvocation)e).decl;
	case TagConstants.VARIABLEACCESS:
	    return ((VariableAccess)e).decl;

	case TagConstants.CASTEXPR:
	    return getDecl( ((CastExpr)e).expr );
	case TagConstants.PARENEXPR:
	    return getDecl( ((ParenExpr)e).expr );
	default:
	    return null;
	}
    }

    private void reportWarnDecl(int locdecl, int locann) {
	if( locdecl != Location.NULL && !Location.isWholeFileLoc(locdecl) &&
	    locann != Location.NULL && !Location.isWholeFileLoc(locann) ) {
	    System.out.println("WarnDecl "+convert(locdecl)+" "+convert(locann));
	}
    }

    private void reportDeclAnn(int locdecl, int locann) {
	if( locdecl != Location.NULL && !Location.isWholeFileLoc(locdecl) &&
	    locann != Location.NULL && !Location.isWholeFileLoc(locann) ) {
	    System.out.println("DeclAnn "+convert(locdecl)+" "+convert(locann));
	}
    }

    private String convert(int loc) {
	return Location.toFileName(loc)+
	    " "+Location.toLineNumber(loc)+
	    " "+Location.toColumn(loc);
    }
}

