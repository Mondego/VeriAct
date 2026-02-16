/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 2001, Compaq Computer Corporation
// Authors:  Rustan Leino and Shaz Qadeer.

package houdini;

/*
* Change history:
*   16 Mar 2001  rustan & qadeer  Created
*/

import java.io.ByteArrayOutputStream;
import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


public class SwingGuesser extends AnnotationGuesser {

  private final boolean guessFalsePrecondition;

  public SwingGuesser(boolean guessFalsePrecondition) {
    this.guessFalsePrecondition = guessFalsePrecondition;
  }
    
  public void guessRoutine(RoutineDecl rd,
			   Annotator requiresAnnotator,
			   Annotator modifiesAnnotator,
			   Annotator ensuresAnnotator,
			   Annotator exsuresAnnotator,
			   Hashtable envReq,
			   Hashtable envMod,
			   Hashtable envEns) {
    if (guessFalsePrecondition && requiresAnnotator != null) {
      requiresAnnotator.put("false");
    }

    if (isJavaAwtComponentSubclass(rd.parent)) {
      if (rd instanceof MethodDecl) {
	// method
	if (!Modifiers.isStatic(rd.modifiers)) {
	  if (requiresAnnotator != null) {
	    // non-overriding method
	    MethodDecl md = (MethodDecl)rd;
	    String name = md.id.toString();
	    if (name.startsWith("set")) {
	      requiresAnnotator.putPermanent("houdini.Swing.inEventDispatchThread || !houdiniCommitted;");
	    }
	  }
	}
      } else {
	// constructor
	ensuresAnnotator.put("!houdiniCommitted;");
      }
    }

    if (requiresAnnotator != null) {
      requiresAnnotator.put("houdini.Swing.inEventDispatchThread");
    }
  }

  private static TypeSig javaAwtComponent = null;

  private boolean isJavaAwtComponentSubclass(TypeDecl td) {
    if (td instanceof ClassDecl) {
      ClassDecl cd = (ClassDecl)td;
      if (javaAwtComponent == null) {
	String[] ss = new String[2];
	ss[0] = "java";
	ss[1] = "awt";
	javaAwtComponent = OutsideEnv.lookupDeferred(ss, "Component");
	Assert.notNull(javaAwtComponent);
      }
      return Types.isSubclassOf(TypeSig.getSig(td), javaAwtComponent);
    }
    return false;
  }

  public void guessFieldDecl(FieldDecl fd,
			     Annotator propertyAnnotator,
			     Annotator modifierAnnotator ) {
  }

  public boolean shouldFieldsBeSpecPublic() {
    return false;
  }

  public void guessExpr(String expr, Type type,
			Annotator propertyAnnotator,
			Annotator modifierAnnotator,
			Hashtable env, ASTNode astNode) {
  }
}
