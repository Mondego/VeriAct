/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 1999, Compaq Computer Corporation
// Authors:  Rustan Leino and Cormac Flanagan.

package houdini;

import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


public class CommonAnnotationGuesser extends AnnotationGuesser {
    
  public void guessRoutine(RoutineDecl rd,
			   Annotator requiresAnnotator,
			   Annotator modifiesAnnotator,
			   Annotator ensuresAnnotator,
			   Annotator exsuresAnnotator,
			   Hashtable envReq,
			   Hashtable envMod,
			   Hashtable envEns) {
    ensuresAnnotator.put("false;");
    if (exsuresAnnotator != null) {
      exsuresAnnotator.put("(Throwable t) false;");
    }
    if (rd instanceof MethodDecl &&
	Types.isReferenceType(((MethodDecl)rd).returnType)) {
      ensuresAnnotator.put("\\fresh(\\result);");
    }
  }
    
  public void guessFieldDecl(FieldDecl fd,
			     Annotator propertyAnnotator,
			     Annotator modifierAnnotator ) {
    // injectivity
    if (!Modifiers.isStatic(fd.modifiers) &&
	Types.isReferenceType(fd.type)) {
      // TBW
    }
  }
    
  public void guessExpr(String expr, Type type,
			Annotator propertyAnnotator,
			Annotator modifierAnnotator,
			Hashtable env, 
			ASTNode astNode) {
    if (Types.isIntegralType(type)) {
      propertyAnnotator.put("0 <= " + expr + ";");

    } else if (Types.isReferenceType(type)) {
      if (modifierAnnotator != null) {
	modifierAnnotator.put("non_null");
      } else {
	propertyAnnotator.put(expr + " != null;");
      }

      if (isObjectArrayType(type)) {
	propertyAnnotator.put("\\nonnullelements(" + expr + ");");
		
	ArrayType at = (ArrayType)type;
	if (! isFinalType(at.elemType)) {
	  String typeName = PrettyPrint.inst.toString(at);
	  propertyAnnotator.put("\\typeof(" + expr + ") == \\type(" +
				typeName + ");");
	}
      }
    }
  }
}
