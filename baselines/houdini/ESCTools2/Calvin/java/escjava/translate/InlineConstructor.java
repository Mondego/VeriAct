/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

import java.util.Vector;
import java.util.Hashtable;
import javafe.util.Assert;
import javafe.util.Location;
import javafe.ast.*;
import javafe.parser.ParseUtil;
import escjava.ast.TagConstants;
import escjava.translate.Substitute;

/* This class performs an experiment with inlining to remove spurious warnings
   caused by the lack of object invariant annotations.  In particular, the class
   takes a method declaration and replaces it by a method declaration consisting
   of an appropriate constructor call, followed by an inlined call to the
   original method. */

public class InlineConstructor {

    //@ requires cus != null
    public static void inlineConstructorsEverywhere(Vector cus) {
	int size = cus.size();
	for (int i = 0; i < size; i++) {
	    CompilationUnit cu = (CompilationUnit) cus.elementAt(i);
	    int numClasses = cu.elems.size();
	    for (int j = 0; j < numClasses; j++) {
		TypeDecl td = (TypeDecl) cu.elems.elementAt(j);
		if (!Modifiers.isAbstract(td.modifiers))
		    inlineConstructorsInAllMethods(td);
	    }
	}
    }
 
    public static void inlineConstructorsInAllMethods(TypeDecl td) {
	// find the appropriate modifier for STATIC

	// create a static field for the replacement for <code>this</code>
	Identifier newThis = uniquifyName("this");
	TypeName selfType0 = TypeName.make(SimpleName.make(td.id, td.locId));
	FieldDecl newField = FieldDecl.make(Modifiers.ACC_STATIC, null, 
					    newThis,
					    selfType0, td.locOpenBrace, null, 
					    td.locOpenBrace);
	newField.setParent(td);
	td.elems.addElement(newField);

	// find all constructors
	TypeDeclElemVec constructors = TypeDeclElemVec.make();
	int size = td.elems.size();
	for (int i = 0; i < size; i++) {
	    TypeDeclElem elem = td.elems.elementAt(i);
	    if (elem instanceof ConstructorDecl)
		constructors.addElement(elem);
	}
	
	int numConstructors = constructors.size();
	
	// if we couldn't find any constructors, we're done
	if (numConstructors == 0)
	    return;

	// for each non-static method, create a constructor-inlined version of
	// it with each constructor
	for (int j = 0; j < size; j++) {
	    TypeDeclElem tde = td.elems.elementAt(j);
	    if (isConstructorInlinable(tde)) {
		for (int i = 0; i < numConstructors; i++)
		    createMethodDecl((MethodDecl) tde, td,
				     (ConstructorDecl) constructors.elementAt(i), 
				     newThis, i);
	    }
	}
    }

	
    private static void createMethodDecl(MethodDecl md, TypeDecl td,
					 ConstructorDecl cd, 
					 Identifier newThis,
					 int count) {

	// create the constructor invocation statement
	TypeName selfType1 = TypeName.make(SimpleName.make(td.id, cd.locId));
	ExprVec evec = ExprVec.make();
	int size = cd.args.size();
	for (int i = 0; i < size; i++) {
	    Identifier argName = uniquifyName(cd.args.elementAt(i).id);
	    Name name = SimpleName.make(argName, cd.locId);
	    evec.addElement(AmbiguousVariableAccess.make(name));
	}
	NewInstanceExpr newexpr = NewInstanceExpr.make(null, cd.locId, 
						       selfType1, evec,
						       null, cd.locId, 
						       cd.locId);
	// inline this call, assuming the preconditions and all body checks
	Translate.inlineDecoration.set(newexpr,
				       new InlineSettings(true, true, true));

	AmbiguousVariableAccess newThisVar =
	    AmbiguousVariableAccess.make(SimpleName.make(newThis, cd.locId));
	BinaryExpr assignExpr = BinaryExpr.make(TagConstants.ASSIGN, newThisVar,
						newexpr, cd.locId);
	EvalStmt firstStmt = EvalStmt.make(assignExpr);


	// create the method invocation statement
	evec = ExprVec.make();
	size = md.args.size();
	for (int i = 0; i < size; i++) {
	    Name name = SimpleName.make(md.args.elementAt(i).id, md.locId);
	    evec.addElement(AmbiguousVariableAccess.make(name));
	}
	// inline this call, checking all of the body checks
	AmbiguousVariableAccess receiver =
	    AmbiguousVariableAccess.make(SimpleName.make(newThis, 
							 md.locId));
	ObjectDesignator od = ExprObjectDesignator.make(md.locId, receiver);
	MethodInvocation invocation =
	    MethodInvocation.make(od, md.id, md.tmodifiers, md.locId,
				  md.locId, evec);
	//inline the call, assuming the preconditions and checking the body
	Translate.inlineDecoration.set(invocation,
				       new InlineSettings(true, false, false));
	Stmt secondStmt;
	if (md.returnType instanceof PrimitiveType &&
	    ((PrimitiveType) md.returnType).tag == TagConstants.VOIDTYPE)
	    secondStmt = EvalStmt.make(invocation);
	else
	    secondStmt = ReturnStmt.make(invocation, md.locId);

	// create the new method
	StmtVec stmts = StmtVec.make(2);
	stmts.addElement(firstStmt);
	stmts.addElement(secondStmt);
	BlockStmt body = BlockStmt.make(stmts, md.loc, md.loc);

	FormalParaDeclVec args = FormalParaDeclVec.make();
	size = md.args.size();
	for (int i = 0; i < size; i++) {
	    FormalParaDecl arg = md.args.elementAt(i);
	    args.addElement(FormalParaDecl.make(arg.modifiers, arg.pmodifiers,
						arg.id, arg.type, arg.locId));
	}
	// we uniquify constructor arg names so they don't clash with
	// the names of the parameters above
	size = cd.args.size();
	for (int i = 0; i < size; i++) {
	    FormalParaDecl arg = cd.args.elementAt(i);
	    Identifier newName = uniquifyName(arg.id);
	    args.addElement(FormalParaDecl.make(arg.modifiers, arg.pmodifiers,
						newName, arg.type, 
						arg.locId));
	}


	/** ***TODO**

	// copy <code>md</code>'s postconditions, with "this" replaced by our
	// newThis.
	ModifierPragmaVec postconds = null;
	boolean first = true;
	if (md.pmodifiers != null) {
	    size = md.pmodifiers.size();
	    for (int i = 0; i < size; i++) {
		ModifierPragma mp = md.pmodifiers.elementAt(i);
		int tag = mp.getTag();
		if (tag == TagConstants.ENSURES || 
		    tag == TagConstants.ALSO_ENSURES) {
		    if (first) {
			postconds = ModifierPragmaVec.make();
			first = false;
		    }
		}
	    }
	}
	**/

	/* create the throws clause of the new method
	   Note:  We don't check for duplicates, so this may contain
	   the same exception twice.  Is that a problem?
	*/
	TypeNameVec raises = null;
	if (md.raises != null) {
	    size = md.raises.size();
	    raises = TypeNameVec.make();
	    for (int i = 0; i < size; i++)
		raises.addElement(copyTypeName(md.raises.elementAt(i), md));
	}	
	if (cd.raises != null) {
	    size = cd.raises.size();
	    if (raises == null)
		raises = TypeNameVec.make(size);
	    for (int i = 0; i < size; i++)
		raises.addElement(copyTypeName(cd.raises.elementAt(i), cd));
	}

	MethodDecl newMethod = MethodDecl.make(Modifiers.ACC_STATIC, null,
					       null, args, raises, body,
					       md.loc, md.loc, md.loc, md.loc,
					       uniquifyName(md.id.toString()
							    + "_" + count),
					       copyType(md.returnType, md), 
					       md.loc);
	newMethod.setParent(td);
	td.elems.addElement(newMethod);
	
    }


    /**
     ** Returns true if the given method is a constructor-inlined version
     ** of some other method
     **/
    public static boolean isConstructorInlinedMethod(MethodDecl md) {
	return md.id.toString().startsWith("*");
    }


    /* Returns true iff the given TypeDeclElem is a non-static, non-helper
       method.
    */
    public static boolean isConstructorInlinable(TypeDeclElem tde) {
	if (tde instanceof MethodDecl) {
	    MethodDecl md = (MethodDecl) tde;
	    if (!(Modifiers.isStatic(md.modifiers) ||
		  Helper.isHelper(md)))
		return true;
	}
	return false;
    }


    /* Return a new copy of a Type, using the location of the surrounding
       RoutineDecl.
    */
    private static Type copyType(Type t, RoutineDecl rd) {
	if (t instanceof PrimitiveType) {
	    PrimitiveType pt = (PrimitiveType) t;
	    return PrimitiveType.make(pt.tag, rd.loc);
	}
	else if (t instanceof TypeName) {
	    return copyTypeName((TypeName) t, rd);
	}
	else if (t instanceof ArrayType) {
	    ArrayType at = (ArrayType) t;
	    return ArrayType.make(copyType(at.elemType, rd), rd.loc);
	}
	else {
	    Assert.fail("Unknown kind of Type");
	    return null;
	}
    }

    private static TypeName copyTypeName(TypeName tn, RoutineDecl rd) {
	return TypeName.make(copyName(tn.name, rd));
    }

    /* Return a new copy of a Name, with all locations nulled out. */
    private static Name copyName(Name n, RoutineDecl rd) {
	if (n instanceof SimpleName) {
	    SimpleName sn = (SimpleName) n;
	    return SimpleName.make(sn.id, rd.loc);
	}
	else if (n instanceof CompoundName) {
	    CompoundName cn = (CompoundName) n;
	    int size = cn.ids.size();
	    IdentifierVec iv = IdentifierVec.make(size);
	    int[] ids = new int[size];
	    int[] dots = new int[size];
	    for (int i = 0; i < size; i++) {
		iv.addElement(cn.ids.elementAt(i));
		ids[i] = rd.loc;
		dots[i] = rd.loc;
	    }
	    return CompoundName.make(iv, ids, dots);
	}
	else {
	    Assert.fail("Unknown kind of Name");
	    return null;
	}
    }


    /* surround an identifier by '*'s. */
    private static Identifier uniquifyName(String s) {
	return Identifier.intern("*" + s + "*");
    }

    private static Identifier uniquifyName(Identifier i) {
	return uniquifyName(i.toString());
    }

}	

