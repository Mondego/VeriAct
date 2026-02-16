/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

import javafe.ast.*;
import escjava.ast.SimpleModifierPragma;
import escjava.ast.TagConstants;

import javafe.tc.TypeSig;
import javafe.util.*;


/**
 ** This class contains a number of routines used in the
 ** interpretation of Java 1.1 as Java 1.0.
 **
 **/

public class Inner {

    /***************************************************
     *                                                 *
     * Getting the enclosing instance field (this$0):   *
     *                                                 *
     ***************************************************/

    /**
     ** Decorates TypeSig nodes to point to their enclosing instance
     ** field (this$0). <p>
     **/
    //@ invariant enclosingInstanceDecoration.decorationType == type(FieldDecl)
    private static final ASTDecoration enclosingInstanceDecoration
	= new ASTDecoration("enclosingInstanceDecoration");


    /**
     ** Return the FieldDecl for a given non-top-level TypeSig's
     ** enclosing instance field (this$0). <p>
     **
     ** The resulting field is public, final, and non_null. <p>
     **/
    //@ ensures RES != null;
    public static FieldDecl getEnclosingInstanceField(TypeSig T) {
	// Memorize our result:
	FieldDecl result = (FieldDecl)enclosingInstanceDecoration.get(T);
	if (result!=null)
	    return result;

	Assert.notFalse(!T.isTopLevelType(), T.toString());

	int loc = T.getTypeDecl().getStartLoc();
	// String name = "this$of$" + T.simpleName;  // for debugging use...
	String name = "this$0";

	ModifierPragmaVec pmodifiers = ModifierPragmaVec.make(1);
	pmodifiers.addElement(SimpleModifierPragma.make(TagConstants.NON_NULL,
							loc));
	
	result =  FieldDecl.make(Modifiers.ACC_PUBLIC|Modifiers.ACC_FINAL,
				 pmodifiers,
				 Identifier.intern(name),
				 T.enclosingType, // the Type of the decl...
				 loc,
				 null, Location.NULL);  // has no initializer
	result.parent = T.getTypeDecl();  // breaks invariant.... !!!!

	enclosingInstanceDecoration.set(T, result);
	return result;
    }
    

    /****************************************************************
     *                                                 		    *
     * Getting the enclosing-instance-field argument (this$0arg):   *
     *                                                 		    *
     ****************************************************************/

    /**
     ** Decorates ConstructorDecl nodes to point to their enclosing-instance-
     ** field argument (this$0arg). <p>
     **/
    /*@ invariant enclosingInstanceArgument.decorationType
                    == type(FormalParaDecl) */
    private static final ASTDecoration enclosingInstanceArgument
	= new ASTDecoration("enclosingInstanceArgument");


    /**
     ** Return the FormalParaDecl for a given inner-class constructor's
     ** enclosing-instance-field argument (this$0arg). <p>
     **
     ** The resulting argument is non_null. <p>
     **
     ** WARNING: Translate.call depends on the exact name of
     ** this$0arg.  If you change this$0arg's name, you must change
     ** Translate.call accordingly.<p>
     **/
    //@ ensures RES != null;
    public static FormalParaDecl getEnclosingInstanceArg(ConstructorDecl cd) {
	// Memorize our result:
	FormalParaDecl result =
	    (FormalParaDecl)enclosingInstanceArgument.get(cd);
	if (result!=null)
	    return result;

	TypeSig T = TypeSig.getSig(cd.parent);
	Assert.notFalse(!T.isTopLevelType(), T.toString());

	String name = "this$0arg";
	ModifierPragmaVec pmodifiers = ModifierPragmaVec.make(1);
	pmodifiers.addElement(SimpleModifierPragma.make(TagConstants.NON_NULL,
							cd.locId));
	
	result =  FormalParaDecl.make(Modifiers.NONE,
				 pmodifiers,
				 Identifier.intern(name),
				 T.enclosingType, //the Type of the argument...
				 cd.locId);

	enclosingInstanceArgument.set(cd, result);
	return result;
    }
    

    /***************************************************
     *                                                 *
     * Unfolding <class>.this Exprs:		       *
     *                                                 *
     ***************************************************/

    /**
     ** If non-null, the local variable or formal to use instead of
     ** this.this$0 when unfolding <class>.this's.
     **/
    static GenericVarDecl firstThis0 = null;

    /**
     ** Converts a 1.1 ThisExpr of the form <class>.this into an Java
     ** 1.0 expression of the form this.this$0.this$0.this$0... <p>
     **
     ** The type of this is taken from GC.thisvar.decl.type. <p>
     **
     ** The this$0 fields are the appropriate fields from the
     ** getEnclosingInstanceField(-) routine above. <p>
     **
     ** The resulting Expr has already been "prepped", but *not* type
     ** checked.<p>
     **
     **
     ** Exception: If firstThis0 is non-null, then it is used instead
     ** of this.this$0 at the start.<p>
     **/
    //@ requires t.classPrefix!=null
    static Expr unfoldThis(/*@ non_null @*/ ThisExpr t) {
	TypeSig target = javafe.tc.Types.toClassTypeSig(t.classPrefix);

	Expr result = ThisExpr.make(null, t.loc);
	TypeSig thisType = (TypeSig)GC.thisvar.decl.type;

	for (int level=0; thisType!=target; level++) {
	    /*
	     * add a .this$0 to result, using firstThis0 the first
	     * time if non-null:
	     */
	    if (level==0 && firstThis0!=null)
		result = TrAnExpr.makeVarAccess(firstThis0, t.loc);
	    else {
		FieldDecl thisDollar = getEnclosingInstanceField(thisType);
		result = FieldAccess.make(ExprObjectDesignator.make(t.loc,
								    result),
					  thisDollar.id, t.loc);
		((FieldAccess)result).decl = thisDollar;
	    }

	    thisType = thisType.enclosingType;
	    Assert.notNull(thisType);
	}

	return result;
    }
}
