/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 2000, Compaq Computer Corporation
// Authors:  Cormac Flanagan and Rustan Leino.

package houdini;

/*
* Change history:
*   16 Sep 1999  rustan & flanagan  Created
*   21 Sep 2000  flanagan           Guess non-null properties of elements of 2d arrays
*/

import java.io.ByteArrayOutputStream;
import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


public class LibraryAnnotationGuesser extends AnnotationGuesser {
    
    CommonAnnotationGuesser cag = new CommonAnnotationGuesser();
    
    public void guessRoutine(RoutineDecl rd,
			     Annotator requiresAnnotator,
			     Annotator modifiesAnnotator,
			     Annotator ensuresAnnotator,
			     Annotator exsuresAnnotator,
			     Hashtable envReq,
			     Hashtable envMod,
			     Hashtable envEns) {
    }
    
    public void guessFieldDecl(FieldDecl fd,
			       Annotator propertyAnnotator,
			       Annotator modifierAnnotator ) {
    }
    
    public boolean shouldFieldsBeSpecPublic() {
	return true;
    }
    
    public void guessExpr(String expr, Type type,
			  Annotator propertyAnnotator,
			  Annotator modifierAnnotator,
			  Hashtable env, ASTNode astNode) {

      // guess consistent annotations

      if (Types.isIntegralType(type)) {
	propertyAnnotator.put("0 <= " + expr);

      } else if (Types.isReferenceType(type)) {
	if (modifierAnnotator != null) {
	  modifierAnnotator.put("non_null");
	} else {
	  propertyAnnotator.put(expr + " != null;");
	}

	if (isObjectArrayType(type)) {
	    //propertyAnnotator.put("\\nonnullelements(" + expr + ");");
	    propertyAnnotator.put(expr+" != null ==> \\nonnullelements(" + expr + ");");

	    ArrayType at = (ArrayType)type;
	    if( isObjectArrayType(at.elemType)) {
		String tmp="houdiniTmp";
		if( !expr.equals(tmp) ) {
		    String s = expr+" != null ==> "+
			"(\\forall int "+tmp+"; "+
			"0 <= "+tmp+" && "+tmp+" < "+expr +".length && "+
			expr+"["+tmp+"] != null ==> "+
			"\\nonnullelements("+expr+"));";
		    propertyAnnotator.put(s);
		}
	    }
	}
      }
    }
}

