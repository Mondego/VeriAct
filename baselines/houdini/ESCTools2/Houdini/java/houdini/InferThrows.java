/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

/*
 * Change history:
 * 11 Oct 2000  flanagan           Created


 * This class infers the exceptions that can be thrown by each routine.
*/

import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
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


public class InferThrows extends DefaultVisitor {

    
    public static void compute(Vector typeDecls) {

	InferThrows inst = new InferThrows();

	do {
	    // System.out.println("Iterating InferThrows");
	    atFixpoint = true;
	    inst.visitTypeDeclVec(typeDecls);
	} while(!atFixpoint);
    }


    /**
     ** Decorates <code>RoutineDecl</code>
     ** nodes to point to <code>Set</code> of TypeSigs for thrown exceptions.
     **/
    //@ invariant throwsDecoration != null;
    // outdated invariant(?): throwsDecoration.decorationType == \type(Type);
    private static ASTDecoration throwsDecoration
	= new ASTDecoration("throwsDecoration");

    private static boolean atFixpoint;

    public static Set getThrows(RoutineDecl d) {
	Set s = (Set)throwsDecoration.get(d);
	if( s == null ) {
	    s = new Set();
	    throwsDecoration.set(d, s);
	}
	for(int i=0; i<d.raises.size(); i++) {
	    s.add( TypeSig.getSig( d.raises.elementAt(i) ));
	}
	return s;
    }

    private static void extendThrows(RoutineDecl d, Set s) {
	Set t = getThrows(d);
	if( t.union(s) ) {
	    atFixpoint = false;
	}
    }

    private Set acc = new Set(); // accumulator

    public void visitTypeDeclVec(Vector tdv) {

	for(Enumeration e=tdv.elements(); e.hasMoreElements(); ) {
	    visitTypeDecl( (TypeDecl)e.nextElement() );
	}
    }



    public void visitRoutineDecl(RoutineDecl d) {
	acc = new Set();
	super.visitRoutineDecl(d);
	if( d instanceof ConstructorDecl ) {

	    // traverse instance field initializers to
	    // incorporate their possible exns

	    ConstructorDecl cd = (ConstructorDecl)d;
	    TypeDeclElemVec siblings = cd.parent.elems;
	    for(int i=0; i<siblings.size(); i++) {
		TypeDeclElem elem = siblings.elementAt(i);
		if( elem instanceof FieldDecl ) {
		    FieldDecl fd = (FieldDecl)elem;
		    if( !javafe.ast.Modifiers.isStatic(fd.modifiers) &&
			fd.init != null ) {
			fd.init.accept( this );
		    }
		}
	    }
	}
	extendThrows(d, acc);

	//TODO: overriding
    }

    public void visitTryCatchStmt(TryCatchStmt x) {
	Set old = acc;
	acc = new Set();
	x.tryClause.accept( this );
	Set tryThrows = acc;
	acc = old;
      addToAcc:
	for(Enumeration e = tryThrows.elements(); e.hasMoreElements(); ) {
	    TypeSig sig = (TypeSig)e.nextElement();
	    // check if sig or a superclass caught
	
	    for(int i=0; i<x.catchClauses.size(); i++) {
		Type t = x.catchClauses.elementAt(i).arg.type;
		TypeSig sigCaught = TypeSig.getSig( (TypeName)t );
		if( sig.isSubtypeOf( sigCaught ) ) {
		    continue addToAcc;
		}
	    }
	    acc.add(sig);
	}

	for(int i=0; i<x.catchClauses.size(); i++) {
	    x.catchClauses.elementAt(i).accept( this );
	}
    }

    public void visitMethodInvocation(MethodInvocation x) {
	super.visitMethodInvocation(x);
	// System.out.println("visitMethodInvocation");
	if( x.od instanceof ExprObjectDesignator &&
	    TypeCheck.inst.getType( ((ExprObjectDesignator)x.od).expr )
	    instanceof ArrayType &&
	    x.id.toString().equals("clone") ) {
	    // Array clone throws no exceptions
	} else {
	    acc.union( getThrows( x.decl ));
	}
    }

    public void visitConstructorInvocation(ConstructorInvocation x) {
	super.visitConstructorInvocation(x);
	acc.union( getThrows( x.decl ));
    }

    public void visitNewInstanceExpr(NewInstanceExpr x) {
	super.visitNewInstanceExpr(x);
	acc.union( getThrows( x.decl ));
    }

  public void visitThrowStmt(ThrowStmt x) {
      super.visitThrowStmt(x);
      acc.add( TypeCheck.inst.getType(x.expr) );
  }


}

