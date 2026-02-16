/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 1999-2001, Compaq Computer Corporation
// Authors:  Cormac Flanagan and Rustan Leino.

package houdini;

/*
* Change history:
*   16 Sep 1999  rustan & flanagan  Created
*   10 Sep 2000  flanagan           Guess small number of disjunctions such as
*                                         arg >= 0 ==> \result != null;
*   21 Sep 2000  flanagan           Guess non-null properties of elements of 2d arrays
*   15 Dec 2000  rustan             Removed identifier 'e' from the guess
*                                     "exsures (Exception e) false;"
*   16 Mar 2001  rustan             Added exsures-false for non-Exception exceptions
*                                   and suppressed "exsures (Exception) false;" if no
*                                   Exception appears in throws clause.  Added
*                                   antecedent in: a != null ==> \typeof(a)==\type(T[]).
*/

import java.io.ByteArrayOutputStream;
import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


public class StandardAnnotationGuesser extends AnnotationGuesser {
    
    CommonAnnotationGuesser cag = new CommonAnnotationGuesser();

    public void guessThrows(RoutineDecl rd) {

	// Throws clauses
	Set raises = InferThrows.getThrows(rd);
	if( raises != null && raises.elements().hasMoreElements() ) {

	    // We inferred some possible exceptions - add them to the throws clause
	    String raisesStr = "";
	    String prefix = "";
	    for(Enumeration myEnum = raises.elements(); myEnum.hasMoreElements(); ) {
		TypeSig sig = (TypeSig)myEnum.nextElement();
		raisesStr += prefix + sig.getExternalName();
		prefix = ", ";
	    }
	    
	    if( rd.raises != null && rd.raises.size() != 0 ) {
		TypeName tn = rd.raises.elementAt(0);
		new Annotator(".", tn.getStartLoc(), tn.name.identifierAt(0).toString(), 
			      "inferredexceptions", "")
		    .put("*/"+raisesStr+", /* Houdini inferred exceptions");
	    } else {
		// Put before open brace
		new Annotator(".", rd.locOpenBrace, "{", 
			      "inferredexceptions", "")
		    .put("*/ throws "+raisesStr+" /* Houdini inferred exceptions");
	    }
	}
    }
    
    private void guessExsuresNoExceptions(RoutineDecl rd,
					  Annotator exsuresAnnotator) {

	// Throws clauses
	Set raises = InferThrows.getThrows(rd);
	if( raises != null ) {
	    // Now add exsures clauses saying these exceptions are impossible
	    for(Enumeration myEnum = raises.elements(); myEnum.hasMoreElements(); ) {
		TypeSig sig = (TypeSig)myEnum.nextElement();
		String annot = "(" + sig.getExternalName() +" ex) (false";
	      findMaxChildren:
		for(Enumeration myEnumChildren = raises.elements(); myEnumChildren.hasMoreElements(); ) {
		    TypeSig subSig = (TypeSig)myEnumChildren.nextElement();
		    if( subSig != sig && subSig.isSubtypeOf( sig ) ) {
			for(Enumeration myEnumBetween = raises.elements(); 
			    myEnumBetween.hasMoreElements(); ) {
			    TypeSig sigBetween = (TypeSig)myEnumBetween.nextElement();
			    if( subSig != sigBetween &&
				subSig.isSubtypeOf( sigBetween ) &&
				sigBetween != sig &&
				sigBetween.isSubtypeOf( sig ) ) {
				// subSig is not a maximal descendent
				continue findMaxChildren;
			    }
			}
			// subSig is a maximal descendent
			// don't put in due to escjava bug
			// annot += " || (ex instanceof " + subSig.getExternalName() + ")";
		    }
		}
		
		if( exsuresAnnotator != null ) {
		    exsuresAnnotator.put(annot+");");
		}
	    }
	}
    }
    
    public void guessRoutine(RoutineDecl rd,
			     Annotator requiresAnnotator,
			     Annotator modifiesAnnotator,
			     Annotator ensuresAnnotator,
			     Annotator exsuresAnnotator,
			     Hashtable envReq,
			     Hashtable envMod,
			     Hashtable envEns) {

	guessExsuresNoExceptions( rd, exsuresAnnotator );


	if( requiresAnnotator != null ) {
	    requiresAnnotator.put("false;");
	}
	// ensuresAnnotator.put("false;");
	if( exsuresAnnotator != null ) {
	    // done elsewhere now
	    //exsuresAnnotator.put("(Exception e) false;");

	  boolean addedException = false;
	  for (int i = 0; i < rd.raises.size(); i++) {
	    TypeName tn = rd.raises.elementAt(i);
	    TypeSig tsig = TypeSig.getSig(tn);
	    if (!Types.isSubclassOf(tsig, Types.javaLangException())) {
	      exsuresAnnotator.put("(" + tn.name.printName() + ") false;");
	    } else if (!addedException) {
	      exsuresAnnotator.put("(Exception) false;");
	      addedException = true;
	    }
	  }

	}
	if( rd instanceof MethodDecl ) {
	    Type resultType = ((MethodDecl)rd).returnType;

	    if( Types.isReferenceType( resultType )) {
		ensuresAnnotator.put("\\fresh(\\result);");
	    }

	    Vector antecedents = new Vector();
	    for(int i = 0; i < rd.args.size(); i++) {
		GenericVarDecl decl = rd.args.elementAt(i);
		if( Types.isIntegralType(decl.type) ) {
		    antecedents.addElement( decl.id.toString()+" >= 0" );
		} else if( Types.isReferenceType(decl.type) ) {
		    antecedents.addElement( decl.id.toString()+" == null" );
		    antecedents.addElement( decl.id.toString()+" != null" );
		} else if( Types.isBooleanType(decl.type) ) {
		    antecedents.addElement( decl.id.toString() );
		    antecedents.addElement( "!" + decl.id.toString() );
		}
	    }

	    for(int i=0; i<antecedents.size(); i++) {
		String antecedent = (String)antecedents.elementAt(i);
		if( Types.isReferenceType( resultType )) {
		    ensuresAnnotator.put( antecedent +" ==> \\result != null;");
		    if( isObjectArrayType( resultType )) {
			ensuresAnnotator.put( antecedent +
					      " ==> \\nonnullelements(\\result);");
		    }
		} else if( Types.isBooleanType( resultType )) {
		    ensuresAnnotator.put( antecedent +" ==> \\result;");
		    ensuresAnnotator.put( antecedent +" ==> !\\result;");
		} else if( Types.isIntegralType( resultType )) {
		    ensuresAnnotator.put( antecedent +" ==> \\result >= 0;");
		}
	    }
	}
    }

    public void guessFieldDecl(FieldDecl fd,
			       Annotator propertyAnnotator,
			       Annotator modifierAnnotator ) {
	
	// injectivity
	if( !Modifiers.isStatic(fd.modifiers) &&
	    Types.isReferenceType(fd.type) ) {
	    // TBW
	}
    }
    
    public boolean shouldFieldsBeSpecPublic() {
	return true;
    }
    
    // Should we put 'helper' on private methods only 
    // called from constructors?
  public boolean guessHelper() {
    return true;
  }
  
    // Should we put final modifier on non-final, 
    // non-overridden, non-abstract methods?
  public boolean guessFinalMethods() {
    return true;
  }
  
    static final String[] comparators = { "<", "<=", "==", "!=", ">=", ">" };
    
    public void guessExpr(String expr, Type type,
			  Annotator propertyAnnotator,
			  Annotator modifierAnnotator,
			  Hashtable env, ASTNode astNode) {

	if( astNode instanceof FieldDecl &&
	    Modifiers.isStatic( ((FieldDecl)astNode).modifiers ) ) {
	    // guess consistent annotations for static fields
	    cag.guessExpr( expr, type, 
			   propertyAnnotator, modifierAnnotator, 
			   env, astNode );
	    return;
	}
	
	if (Types.isReferenceType(type)) {
	    if (modifierAnnotator != null) {
		modifierAnnotator.put("non_null");
	    } else {
		propertyAnnotator.put(expr + " != null;");
	    }
	    
	    if (isObjectArrayType(type)) {
		// propertyAnnotator.put("\\nonnullelements(" + expr + ");");
		propertyAnnotator.put(expr+" != null ==> \\nonnullelements(" + expr + ");");
		
		ArrayType at = (ArrayType)type;
		if (! isFinalType(at.elemType)) {
		    ByteArrayOutputStream baos = new ByteArrayOutputStream();
		    PrettyPrint.inst.print(baos, at);
		    propertyAnnotator.put(expr + " != null ==> \\typeof(" + expr +
					  ") == \\type(" + baos.toString() + ");");
		}
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
	
	if (Types.isBooleanType(type)) {
	    propertyAnnotator.put(expr + ";");
	    propertyAnnotator.put("!" + expr + ";");
	}
	
	for( Enumeration e = env.keys(); e.hasMoreElements(); ) {
	    String expr2 = (String)e.nextElement();
	    Type type2 = (Type)env.get(expr2);
	    
	    guessPair(expr, type, expr2, type2, propertyAnnotator);
	}
    }
    
    private void guessPair(String expr, Type type,
			   String expr2, Type type2,
			   Annotator propertyAnnotator) {
	
	if( Types.isIntegralType(type) && Types.isIntegralType(type2) ) {
	    for(int i=0; i<comparators.length; i++) {
		propertyAnnotator.put(expr+" "+comparators[i]+" "+expr2+";"); //-ARN
	    }
	} else if( type2 instanceof ArrayType ) {
	    guessTypeAndArray( expr, type, expr2, ((ArrayType)type2),
			       propertyAnnotator );
	}
	else if( type instanceof ArrayType ) {
	    guessTypeAndArray( expr2, type2, expr, ((ArrayType)type),
			       propertyAnnotator );
	}
    }
    
    
    
    private void guessTypeAndArray( /*@ non_null */ String expr,
				    Type type,
				    /*@ non_null */ String arrayExpr,
				    /*@ non_null */ ArrayType arrayType,
				    /*@ non_null */ Annotator propertyAnnotator ) {
	if (!expr.equals("0")) {
	    guessPair( expr, type,
		       arrayExpr+".length", Types.intType,
		       propertyAnnotator );
	    
	    if( Types.isIntegralType(type) &&
		Types.isReferenceType(arrayType.elemType) ) {
		String tmp="houdiniTmp";
		if( !expr.equals(tmp) && !arrayExpr.equals(tmp) ) {
		    propertyAnnotator.put("(\\forall int "+tmp
					  +"; 0 <= "+tmp+" && "+tmp+" < "+expr
					  +" ==> "+arrayExpr+"["+tmp+"] != null);");
		    propertyAnnotator.put("(\\forall int "+tmp
					  +"; "+expr+" <= "+tmp+" && "+tmp+" < "+arrayExpr+
					  ".length ==> "+arrayExpr+"["+tmp+"] != null);");
		}
	    }
	}
    }
    
}

