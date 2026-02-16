/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 1999, Compaq Computer Corporation
// Authors:  Cormac Flanagan and Rustan Leino.
// Change history:
//      Sep 1999  rustan & flanagan  Created
//   17 Sep 2000  flanagan           Put helper annotations on private methods
//                                   called only from constructors
//   21 Sep 2000  flanagan           Put final modifiers on non-overridden methods

package houdini;

import java.util.Hashtable;

import javafe.ast.*;
import javafe.tc.*;


public abstract class AnnotationGuesser {

    public void guessThrows(RoutineDecl rd) {}

  //@ requires requiresAnnotator != null ==> envMod != null;
  /*@ requires envReq != null ==>
               envReq.keyType == \type(String) && envReq.elementType == \type(Type); */
  /*@ requires envMod.keyType == \type(String) && envMod.elementType == \type(Type); */
  /*@ requires envEns.keyType == \type(String) && envEns.elementType == \type(Type); */
  public abstract void guessRoutine(/*@ non_null */ RoutineDecl rd,
				    Annotator requiresAnnotator,
				    /*@ non_null */ Annotator modifiesAnnotator,
				    /*@ non_null */ Annotator ensuresAnnotator,
				    Annotator exsuresAnnotator,
				    /*--- readable_if requiresAnnotator != null; ---*/
				    Hashtable envReq,
				    /*@ non_null */ Hashtable envMod,
				    /*@ non_null */ Hashtable envEns);

  // "expr" is fully parenthesized
  // astNode is null (for RES), GenericVarDecl (for args), or a FieldDecl
  /*@ requires env.keyType == \type(String) && env.elementType == \type(Type); */
  public abstract void guessExpr(/*@ non_null */ String expr,
				 /*@ non_null */ Type type,
				 /*@ non_null */ Annotator propertyAnnotator,
				 Annotator modifierAnnotator,
				 /*@ non_null */ Hashtable env,
				 ASTNode astNode);
  
  public abstract void guessFieldDecl(/*@ non_null */ FieldDecl fd,
				      /*@ non_null */ Annotator propertyAnnotator,
				      Annotator modifierAnnotator);

  public boolean shouldFieldsBeSpecPublic() {
    // false by default
    return false;
  }
  
    // Should we put 'helper' on private methods only 
    // called from constructors?
  public boolean guessHelper() {
    // false by default
    return false;
  }
  
    // Should we put final modifier on non-final, 
    // non-overridden, non-abstract methods?
  public boolean guessFinalMethods() {
    // false by default
    return false;
  }
  
  //@ ensures \result ==> t instanceof ArrayType;
  public static boolean isObjectArrayType(/*@ non_null */ Type t) {
    if (t instanceof ArrayType) {
      ArrayType at = (ArrayType)t;
      return Types.isReferenceType(at.elemType);
    }
    return false;
  }

  protected boolean isFinalType(/*@ non_null */ Type t) {
    if (t instanceof PrimitiveType) {
      return true;
    }
    
    if (t instanceof ArrayType) {
      ArrayType at = (ArrayType)t;
      return isFinalType(at.elemType);
    }

    if (t instanceof TypeName) {
      TypeName tn = (TypeName)t;
      t = TypeSig.getSig(tn); //@ nowarn NonNull;
    }

    //@ assume t instanceof TypeSig;
    TypeSig sig = (TypeSig)t;
    TypeDecl td = sig.getTypeDecl();
    return Modifiers.isFinal(td.modifiers);
  }
}
