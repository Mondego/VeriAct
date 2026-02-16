/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 1999, Compaq Computer Corporation
// Authors:  Cormac Flanagan and Rustan Leino.

package houdini;

import java.io.ByteArrayOutputStream;
import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


public class ReqFalseAnnotationGuesser extends AnnotationGuesser {
    
    
    public void guessRoutine(RoutineDecl rd,
			     Annotator requiresAnnotator,
			     Annotator modifiesAnnotator,
			     Annotator ensuresAnnotator,
			     Annotator exsuresAnnotator,
			     Hashtable envReq,
			     Hashtable envMod,
			     Hashtable envEns) {
	if (!(rd instanceof MethodDecl &&
	      ((MethodDecl)rd).id.toString().equals("main"))) {
	    if( requiresAnnotator != null ) {
		requiresAnnotator.put("false;");
	    }
	    ensuresAnnotator.put("false;");
	    if( exsuresAnnotator != null ) {
		exsuresAnnotator.put("(Exception e) false;");
	    }
	    
	}
	
    }
    
    public void guessFieldDecl(FieldDecl fd,
			       Annotator propertyAnnotator,
			       Annotator modifierAnnotator ) {
	
    }
    
    public void guessExpr(String expr, Type type,
			  Annotator propertyAnnotator,
			  Annotator modifierAnnotator,
			  Hashtable env, ASTNode astNode) {
    }
    
}
