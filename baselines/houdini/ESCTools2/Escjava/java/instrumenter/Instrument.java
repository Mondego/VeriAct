/* Copyright 2000, 2001, Compaq Computer Corporation */

package instrumenter;

import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.io.BufferedWriter;
import java.io.File;
import java.io.IOException;

import javafe.ast.*;
import javafe.tc.*;
import escjava.ast.*;
import escjava.ast.EscPrettyPrint;
import escjava.ast.TagConstants;

import javafe.reader.StandardTypeReader;
import escjava.reader.EscTypeReader;
import javafe.parser.PragmaParser;
import escjava.translate.*;
import javafe.tc.TypeSig;
import escjava.tc.TypeCheck;

import javafe.util.*;

/**
 ** Houdini instrumenter to instrument one compilation unit. <p>
 **
 ** This class (and its superclasses) instruments one compilation unit. <p>
 **
 **/

public class Instrument {

  static String varPrefix = "InStRuMeNtEr_";
  
  static TypeObjectDesignator checkClass =
    TypeObjectDesignator.make(Location.NULL,
			      TypeName.make(Name.make("instrumenter.Check", Location.NULL)));

  static Identifier checkNonNull = Identifier.intern("checkNonNull");
  static Identifier checkAssertion = Identifier.intern("checkAssertion");
  static Identifier checkInstanceInvariants = Identifier.intern(varPrefix + "cii");
  static Identifier checkStaticInvariants = Identifier.intern(varPrefix + "csi");
  static Identifier ciParam = Identifier.intern(varPrefix + "loc");
  static Expr ciParamRef = AmbiguousVariableAccess.make(Name.make(ciParam.toString(), Location.NULL));
  
  static Identifier resId = Identifier.intern(InstrumenterPrettyPrint.RES);
  static Expr resRef = AmbiguousVariableAccess.make(Name.make(resId.toString(), Location.NULL));

  static int annotationCnt = 0;
  //@ invariant annotationToNum.keyType == \type(ASTNode);
  //@ invariant annotationToNum.elementType == \type(Integer);
  final static Hashtable annotationToNum = new Hashtable();
  //@ invariant annotationVec.elementType == \type(ASTNode);
  final static Vector annotationVec = new Vector();
  
  static int annotationToNum(/*@ non_null */ ASTNode anno) {
    Integer num = (Integer)annotationToNum.get(anno);
    if (num != null) {
      return num.intValue();
    }
    annotationToNum.put(anno, new Integer(annotationCnt));
    annotationVec.addElement(anno);
    return annotationCnt++;
  }

  static void dumpAnnotationInfo(File infoFile) throws IOException {
    File pfile = new File(infoFile.getParent());
    pfile.mkdirs();
    FileWriter fw = new FileWriter(infoFile);
    PrintWriter pw = new PrintWriter(new BufferedWriter(fw));
    pw.println("package houdini;");
    pw.println("public class Info {");
    pw.println("public static String[] annotationLocs = {");
    for (int i = 0; i < annotationVec.size(); i++) {
      ASTNode anno = (ASTNode)annotationVec.elementAt(i);
      int loc = anno.getStartLoc();
      pw.print("\"");
      pw.print(Location.toFileName(loc));
      pw.print(" ");
      pw.print(Location.toLineNumber(loc));
      pw.print(" ");
      pw.print(Location.toColumn(loc));
      pw.print("\"");
      if (i+1 < annotationVec.size())
	pw.println(",");
      else
	pw.println();
    }
    pw.println("}; }");
    pw.close();
  }
  
  /* this is not completely correct for method overwrites. */
  private static SimpleModifierPragma getNonNullAnno(ModifierPragmaVec mps) {
    if (mps == null) return null;
    for (int i = 0; i < mps.size(); i++) {
      ModifierPragma mp = mps.elementAt(i);
      if (mp.getTag() == TagConstants.NON_NULL) {
	return (SimpleModifierPragma)mp;
      }
    }
    return null;
  }
  
  static void instrument(CompilationUnit cu) {
    traverse(cu);
  }

  static void traverse(ASTNode node) {
    switch (node.getTag()) {
    case TagConstants.FIELDDECL:
      {
	// We make everything public:
	FieldDecl fd = (FieldDecl)node;
	fd.modifiers |= Modifiers.ACC_PUBLIC;
	fd.modifiers &= ~Modifiers.ACC_PRIVATE;
	fd.modifiers &= ~Modifiers.ACC_PROTECTED;
	break;
      }
    case TagConstants.METHODDECL:
    case TagConstants.CONSTRUCTORDECL:
      {
	RoutineDecl rd = (RoutineDecl)node;
	if (rd.body == null) break;
	NewBody newBody = prepareBody(rd.body, rd.raises);

	if (! (rd instanceof ConstructorDecl)) {
	  MethodDecl md = (MethodDecl)rd;
	  if (!Types.isVoidType(md.returnType)) {
	    LocalVarDecl resDecl =
	      LocalVarDecl.make(Modifiers.NONE, null, resId, md.returnType,
				Location.NULL, zeroequiv(md.returnType),
				Location.NULL);
	    VarDeclStmt vds = VarDeclStmt.make(resDecl);
	    newBody.newBody.stmts.insertElementAt(vds, 1);
	  }
	}

	// add checks for non-null parameters:
	for (int i = 0; i < rd.args.size(); i++) {
	  FormalParaDecl arg = rd.args.elementAt(i);
	  SimpleModifierPragma mp = getNonNullAnno(arg.pmodifiers);
	  if (mp != null) {
	    Expr expr = AmbiguousVariableAccess.make(Name.make(arg.id.toString(), Location.NULL));
	    MethodInvocation mi = callToCheckNonNull(mp, arg.getStartLoc(), expr);
	    Stmt s = EvalStmt.make(mi);
	    newBody.mainBody.stmts.insertElementAt(s, 0);
	  }
	}
	// checks for the preconditions
	DerivedMethodDecl dmd = GetSpec.getCombinedMethodDecl(rd);
	for (int i = 0; i < dmd.requires.size(); i++) {
	  ExprModifierPragma emp = dmd.requires.elementAt(i);
	  if (isExecutable(emp.expr)) {
	    Stmt s = callToCheck(emp, rd.getStartLoc(), emp.expr);
	    newBody.mainBody.stmts.insertElementAt(s, 0);
	  }
	}
	// checks for the postconditions
	for (int i = 0; i < dmd.ensures.size(); i++) {
	  ExprModifierPragma emp = dmd.ensures.elementAt(i);
	  if (isExecutable(emp.expr)) {
	    Stmt s = callToCheck(emp, rd.getEndLoc(), emp.expr);
	    newBody.normalFinallyBlock.stmts.addElement(s);
	  }
	}
	if (dmd.nonnull != null) {
	  int loc = dmd.nonnull.getStartLoc();
	  LiteralExpr nulllit = LiteralExpr.make(TagConstants.NULLLIT,
						 null, loc);
	  Stmt s = callToCheck(dmd.nonnull, rd.getEndLoc(),
			       BinaryExpr.make(TagConstants.NE,
					       ResExpr.make(loc), nulllit,
					       loc));
	  newBody.normalFinallyBlock.stmts.addElement(s);
	}
	// checks for the exceptional postconditions
	// NOTE:  This code may not be correct!
	/***
	for (int i = 0; i < dmd.exsures.size(); i++) {
	  ExprModifierPragma emp = dmd.ensures.elementAt(i);
	  if (isExecutable(emp.expr)) {
	    Stmt s = callToCheck(mp, rd.getStartLoc(), emp.expr);
	    newBody[2].stmts.insertElementAt(s, 0);
	  }
	}
	***/
	// checks for non-null fields at the end of constructors
	ClassDecl td = (ClassDecl)rd.parent;
	if (rd instanceof ConstructorDecl) {
	  for (int i = 0; i < td.elems.size(); i++) {
	    TypeDeclElem elem = td.elems.elementAt(i);
	    if (elem.getTag() == TagConstants.FIELDDECL) {
	      FieldDecl fd = (FieldDecl)elem;
	      SimpleModifierPragma mp = getNonNullAnno(fd.pmodifiers);
	      if (mp != null) {
		ObjectDesignator od;
		if (Modifiers.isStatic(fd.modifiers)) {
		  od = TypeObjectDesignator.make(Location.NULL, TypeSig.getSig(fd.parent));
		}
		else {
		  Expr expr = ThisExpr.make(null, Location.NULL);
		  od = ExprObjectDesignator.make(Location.NULL, expr);
		}
		FieldAccess fa = FieldAccess.make(od, fd.id, Location.NULL);
		MethodInvocation mi = callToCheckNonNull(mp, rd.getEndLoc(), fa);
		newBody.normalFinallyBlock.stmts.insertElementAt(EvalStmt.make(mi), 0);
	      }
	    }
	  }
	}
	// checks for invariants
	if (!Helper.isHelper(rd)) {
	  Type type = TypeSig.getSig(td);
	  if (rd instanceof ConstructorDecl) {
	    newBody.normalFinallyBlock.stmts.addElement(callToInstanceInvariantChecks(rd.getEndLoc()));
	  } else {
	    newBody.mainBody.stmts.insertElementAt(callToStaticInvariantChecks(rd.getStartLoc(), type), 0);
	    if (!Modifiers.isStatic(rd.modifiers)) {
	      newBody.mainBody.stmts.insertElementAt(callToInstanceInvariantChecks(rd.getStartLoc()), 0);
	      newBody.allFinallyBlock.stmts.addElement(callToInstanceInvariantChecks(rd.getEndLoc()));
	    }
	  }
	  newBody.allFinallyBlock.stmts.addElement(callToStaticInvariantChecks(rd.getEndLoc(), type));
	}
	
	rd.body = newBody.newBody;
	break;
      }
    case TagConstants.ASSIGN:
      {
	BinaryExpr be = (BinaryExpr)node;
	if (be.left instanceof FieldAccess) {
	  FieldDecl fd = ((FieldAccess)be.left).decl;
	  SimpleModifierPragma mp = getNonNullAnno(fd.pmodifiers);
	  if (mp != null) {
	    MethodInvocation mi = callToCheckNonNull(mp, be.getStartLoc(), be.right);
	    be.right = CastExpr.make(mi, fd.type, Location.NULL, Location.NULL);
	  }
	}
	break;
      }
    case TagConstants.RETURNSTMT:
      {
	ReturnStmt rs = (ReturnStmt)node;
	if (rs.expr != null) {
	  rs.expr = BinaryExpr.make(TagConstants.ASSIGN, resRef, rs.expr, Location.NULL);
	}
	break;
      }
    default:
      {
	break;
      }
    }
    // Traverse all the children of the node:
    for (int i = 0; i < node.childCount(); i++) {
      Object child = node.childAt(i);
      if (child != null && (child instanceof ASTNode)) { 
	traverse((ASTNode)child);
      }
    }
    // Now, after the traversal, add the new method if "node" is a class declaration
    if (node.getTag() == TagConstants.CLASSDECL) {
      ClassDecl td = (ClassDecl)node;

      // add methods that will check invariants
      StmtVec ssStatic = StmtVec.make();
      StmtVec ssInstance = StmtVec.make();

      ExprDeclPragmaVec invs = getCombinedInvariants(td);
      for (int i = 0; i < invs.size(); i++) {
	ExprDeclPragma inv = invs.elementAt(i);
	if (isExecutable(inv.expr)) {
	  Stmt s = callToInvCheck(inv, inv.expr);
	  if (mentionsThis(inv.expr)) {
	    ssInstance.addElement(s);
	  } else {
	    ssStatic.addElement(s);
	  }
	}
      }

      FormalParaDeclVec formalparameters = FormalParaDeclVec.make();
      formalparameters.addElement(FormalParaDecl.make(Modifiers.NONE, null,
						      ciParam, Types.javaLangString(),
						      Location.NULL));
      TypeNameVec raises = TypeNameVec.make();
      MethodDecl md;
      md = MethodDecl.make(Modifiers.ACC_PRIVATE | Modifiers.ACC_STATIC, null, null,
			   formalparameters, raises,
			   BlockStmt.make(ssStatic, Location.NULL, Location.NULL),
			   td.getEndLoc(), td.getEndLoc(),
			   td.getEndLoc(), Location.NULL,
			   checkStaticInvariants,
			   Types.voidType, td.getEndLoc());
      md.setParent(td);
      td.elems.addElement(md);
      md = MethodDecl.make(Modifiers.ACC_PRIVATE, null, null,
			   formalparameters, raises,
			   BlockStmt.make(ssInstance, Location.NULL, Location.NULL),
			   td.getEndLoc(), td.getEndLoc(),
			   td.getEndLoc(), Location.NULL,
			   checkInstanceInvariants,
			   Types.voidType, td.getEndLoc());
      md.setParent(td);
      td.elems.addElement(md);
    }
  }

  static ExprDeclPragmaVec getCombinedInvariants(ClassDecl cd) {
    ExprDeclPragmaVec invs;
    if (cd.superClass == null) {
      invs = ExprDeclPragmaVec.make();
    }
    else {
      TypeDecl superDecl = TypeSig.getSig(cd.superClass).getTypeDecl();
      invs = getCombinedInvariants((ClassDecl)superDecl);
    }
    for (int i = 0; i < cd.elems.size(); i++) {
      TypeDeclElem elem = cd.elems.elementAt(i);
      if (elem.getTag() == TagConstants.INVARIANT) {
	invs.addElement((ExprDeclPragma)elem);
      }
    }
    return invs;
  }

  static MethodInvocation callToCheckNonNull(SimpleModifierPragma mp,
					     int loc, Expr expr) {
    int aNum = annotationToNum(mp);
    String srcLoc = Location.toFileName(loc) + ":" +
                    Location.toLineNumber(loc);
    Expr p0 = LiteralExpr.make(TagConstants.STRINGLIT, srcLoc, Location.NULL);
    Expr p1 = LiteralExpr.make(TagConstants.INTLIT, new Integer(aNum), Location.NULL);
    ExprVec params = ExprVec.make(3);
    params.addElement(p0);
    params.addElement(p1);
    params.addElement(expr);
    return MethodInvocation.make(checkClass, checkNonNull, null, Location.NULL,
				 Location.NULL, params);
  }

  static Stmt callToCheck(ASTNode mp, int loc, Expr expr) {
    int aNum = annotationToNum(mp);
    String srcLoc = Location.toFileName(loc) + ":" +
                    Location.toLineNumber(loc);
    Expr p0 = LiteralExpr.make(TagConstants.STRINGLIT, srcLoc, Location.NULL);
    Expr p1 = LiteralExpr.make(TagConstants.INTLIT, new Integer(aNum), Location.NULL);
    return callToCheckAux(checkAssertion, p0, p1, expr);
  }

  static Stmt callToInvCheck(ASTNode mp, Expr expr) {
    int aNum = annotationToNum(mp);
    Expr p1 = LiteralExpr.make(TagConstants.INTLIT, new Integer(aNum), Location.NULL);
    return callToCheckAux(checkAssertion, ciParamRef, p1, expr);
  }

  private static Stmt callToCheckAux(Identifier methodId,
				     Expr p0, Expr p1, Expr p2) {
    ExprVec params = ExprVec.make(3);
    params.addElement(p0);
    params.addElement(p1);
    params.addElement(p2);
    MethodInvocation mi = MethodInvocation.make(checkClass, methodId, null,
						Location.NULL,
						Location.NULL, params);
    EvalStmt call0 = EvalStmt.make(mi);

    params = ExprVec.make(3);
    params.addElement(p0);
    params.addElement(p1);
    params.addElement(GC.falselit);
    mi = MethodInvocation.make(checkClass, methodId, null, Location.NULL,
			       Location.NULL, params);
    EvalStmt call1 = EvalStmt.make(mi);

    Identifier tid = Identifier.intern(varPrefix + "t");
    Expr tRef = AmbiguousVariableAccess.make(Name.make(tid.toString(), Location.NULL));
    FormalParaDecl tDecl = FormalParaDecl.make(Modifiers.NONE, null, tid,
					       Types.javaLangRuntimeException(),
					       Location.NULL);
    StmtVec catchBodyVec = StmtVec.make(1);
    catchBodyVec.addElement(call1);
    BlockStmt catchBody = BlockStmt.make(catchBodyVec, Location.NULL, Location.NULL);
    CatchClause cc = CatchClause.make(tDecl, catchBody, Location.NULL);

    CatchClauseVec ccVec = CatchClauseVec.make(1);
    ccVec.addElement(cc);
    return TryCatchStmt.make(call0, ccVec, Location.NULL);
  }

  static EvalStmt callToInstanceInvariantChecks(int loc) {
    String srcLoc = Location.toFileName(loc) + ":" +
                    Location.toLineNumber(loc);
    ExprVec params = ExprVec.make(1);
    params.addElement(LiteralExpr.make(TagConstants.STRINGLIT, srcLoc, Location.NULL));
    MethodInvocation mi = MethodInvocation.
      make(ExprObjectDesignator.make(loc, ThisExpr.make(null, Location.NULL)),
	   checkInstanceInvariants,
	   null,
	   Location.NULL,
	   Location.NULL,
	   params);
    return EvalStmt.make(mi);
  }

  //@ requires loc != Location.NULL;
  static EvalStmt callToStaticInvariantChecks(int loc, /*@ non_null */ Type t) {
    String srcLoc = Location.toFileName(loc) + ":" +
                    Location.toLineNumber(loc);
    ExprVec params = ExprVec.make(1);
    params.addElement(LiteralExpr.make(TagConstants.STRINGLIT, srcLoc, Location.NULL));
    MethodInvocation mi = MethodInvocation.make(TypeObjectDesignator.make(loc, t),
						checkStaticInvariants, null,
						Location.NULL, Location.NULL, params);
    return EvalStmt.make(mi);
  }

  static boolean isExecutable(Expr expr) {
    switch (expr.getTag()) {
    case TagConstants.THISEXPR:
    case TagConstants.BOOLEANLIT:
    case TagConstants.INTLIT:
    case TagConstants.LONGLIT:
    case TagConstants.FLOATLIT:
    case TagConstants.DOUBLELIT:
    case TagConstants.STRINGLIT:
    case TagConstants.CHARLIT:
    case TagConstants.RESEXPR:
      {
	return true;
      }
    case TagConstants.LE:
    case TagConstants.LT:
    case TagConstants.GT:
    case TagConstants.GE:
    case TagConstants.EQ:
    case TagConstants.NE:
      {
	BinaryExpr be = (BinaryExpr)expr;
	return isExecutable(be.left) && isExecutable(be.right);
      }
    case TagConstants.UNARYADD:
    case TagConstants.UNARYSUB:
    case TagConstants.NOT:
    case TagConstants.BITNOT:
      {
	UnaryExpr ue = (UnaryExpr)expr;
	return isExecutable(ue.expr);
      }
    case TagConstants.PARENEXPR:
      {
	ParenExpr pe = (ParenExpr)expr;
	return isExecutable(pe.expr);
      }
    case TagConstants.VARIABLEACCESS:
    case TagConstants.CLASSLITERAL:      
      {
	return true;
      }
    case TagConstants.FIELDACCESS:
      {
	FieldAccess fa = (FieldAccess)expr;
	if (fa.od instanceof ExprObjectDesignator) {
	  ExprObjectDesignator ed = (ExprObjectDesignator)fa.od;
	  return isExecutable(ed.expr);
	}
	return true;
      }
    case TagConstants.AMBIGUOUSVARIABLEACCESS:
    case TagConstants.AMBIGUOUSMETHODINVOCATION:
    case TagConstants.NEWINSTANCEEXPR:
    case TagConstants.NEWARRAYEXPR:
    case TagConstants.METHODINVOCATION:
      {
        Assert.fail("unexpected tag" + expr.getTag());
        return false;
      }
    case TagConstants.ARRAYREFEXPR:
    case TagConstants.CONDEXPR:
    default:
      {
	return false;
      }
    }
  }

  static boolean mentionsThis(/*@ non_null */ Expr expr) {
    switch (expr.getTag()) {
    case TagConstants.THISEXPR:
      {
        return true;
      }
    case TagConstants.LE:
    case TagConstants.LT:
    case TagConstants.GT:
    case TagConstants.GE:
    case TagConstants.EQ:
    case TagConstants.NE:
      {
	BinaryExpr be = (BinaryExpr)expr;
	return mentionsThis(be.left) || mentionsThis(be.right);
      }
    case TagConstants.UNARYADD:
    case TagConstants.UNARYSUB:
    case TagConstants.NOT:
    case TagConstants.BITNOT:
      {
	UnaryExpr ue = (UnaryExpr)expr;
	return mentionsThis(ue.expr);
      }
    case TagConstants.PARENEXPR:
      {
	ParenExpr pe = (ParenExpr)expr;
	return mentionsThis(pe.expr);
      }
    case TagConstants.VARIABLEACCESS:
    case TagConstants.CLASSLITERAL:      
      {
	return true;
      }
    case TagConstants.FIELDACCESS:
      {
	FieldAccess fa = (FieldAccess)expr;
	switch (fa.od.getTag()) {
	  case TagConstants.TYPEOBJECTDESIGNATOR:
	    return false;
	  case TagConstants.SUPEROBJECTDESIGNATOR:
	    return true;
	  case TagConstants.EXPROBJECTDESIGNATOR: {
	    ExprObjectDesignator ed = (ExprObjectDesignator)fa.od;
	    return mentionsThis(ed.expr);
	  }
	}
	//@ unreachable;
	return false;
      }
    case TagConstants.AMBIGUOUSVARIABLEACCESS:
    case TagConstants.AMBIGUOUSMETHODINVOCATION:
    case TagConstants.NEWINSTANCEEXPR:
    case TagConstants.NEWARRAYEXPR:
    case TagConstants.METHODINVOCATION:
      {
        Assert.fail("unexpected tag" + expr.getTag());
        return false;
      }
    case TagConstants.BOOLEANLIT:
    case TagConstants.INTLIT:
    case TagConstants.LONGLIT:
    case TagConstants.FLOATLIT:
    case TagConstants.DOUBLELIT:
    case TagConstants.STRINGLIT:
    case TagConstants.CHARLIT:
    case TagConstants.ARRAYREFEXPR:
    case TagConstants.CONDEXPR:
    default:
      {
	return false;
      }
    }
  }

  /**
   * Takes a BlockStmt, and wraps it with a try/catch statement and
   * then a try/finally statement.
   **/
  static NewBody prepareBody(/*@ non_null */ BlockStmt body,
				 /*@ non_null */ TypeNameVec raises) {
    NewBody res = new NewBody();
    res.mainBody = body;
    
    StmtVec newBodyVec = StmtVec.make();
    // Promote the super constructor:
    if ((body.stmts.size() != 0) &&
	(body.stmts.elementAt(0) instanceof ConstructorInvocation)) {
      newBodyVec.addElement(body.stmts.elementAt(0));
      body.stmts.removeElementAt(0);
    }
    // Introduce the boolean "b" and exception "t":
    Identifier bid = Identifier.intern(varPrefix + "b");
    Expr bRef = AmbiguousVariableAccess.make(Name.make(bid.toString(), Location.NULL));
    LocalVarDecl bDecl =
      LocalVarDecl.make(Modifiers.NONE, null, bid, Types.booleanType, Location.NULL,
			GC.truelit, Location.NULL);
    VarDeclStmt vds = VarDeclStmt.make(bDecl);
    newBodyVec.addElement(vds);

    CatchClauseVec ccVec = CatchClauseVec.make();
    OUTER:
    for (int i = 0; i < raises.size(); i++) {
      TypeName tn = raises.elementAt(i);
      // Check if "tn" is a subclass of an exception we've already seen
      for (int j = 0; j < i; j++) {
	TypeName tnPrev = raises.elementAt(j);
	TypeSig tsPrev = TypeSig.getSig(tnPrev);
	if (Types.isSubclassOf(tn, tsPrev)) {
	  continue OUTER;
	}
      }

      Identifier tid = Identifier.intern("t");
      LocalVarDecl tDecl = LocalVarDecl.make(Modifiers.NONE, null, tid, tn,
					     Location.NULL, GC.nulllit, Location.NULL);
      Expr tRef = AmbiguousVariableAccess.make(Name.make(tid.toString(),
							 Location.NULL));

      StmtVec bsVec = StmtVec.make(2);
      bsVec.addElement(VarDeclStmt.make(tDecl));
      bsVec.addElement(ThrowStmt.make(tRef, Location.NULL));
      BlockStmt bs = BlockStmt.make(bsVec, Location.NULL, Location.NULL);
      IfStmt ifStmt = IfStmt.make(bRef, SkipStmt.make(Location.NULL), bs, Location.NULL);
      body.stmts.insertElementAt(ifStmt, 0);

      CatchClause cc = makeCatchClause(tn, bRef);
      ccVec.addElement(cc);
    }
    ccVec.addElement(makeCatchClause(Types.javaLangRuntimeException(), bRef));
    ccVec.addElement(makeCatchClause(Types.javaLangError(), bRef));
    TryCatchStmt tryCatch = TryCatchStmt.make(body, ccVec, Location.NULL);

    StmtVec ifBodyVec = StmtVec.make();
    BlockStmt ifBody = BlockStmt.make(ifBodyVec, Location.NULL, Location.NULL);
    res.normalFinallyBlock = ifBody;

    IfStmt ifStmt = IfStmt.make(bRef, ifBody, SkipStmt.make(Location.NULL), Location.NULL);

    StmtVec finallyBodyVec = StmtVec.make();
    finallyBodyVec.addElement(ifStmt);
    BlockStmt finallyBody = BlockStmt.make(finallyBodyVec, Location.NULL, Location.NULL);
    res.allFinallyBlock = finallyBody;

    TryFinallyStmt tryFinallyStmt =
      TryFinallyStmt.make(tryCatch, finallyBody, Location.NULL, Location.NULL);
    newBodyVec.addElement(tryFinallyStmt);
    res.newBody = BlockStmt.make(newBodyVec,  Location.NULL, Location.NULL);
    
    return res;
  }

  static private CatchClause makeCatchClause(/*@ non_null */ Type exceptionType,
					     /*@ non_null */ Expr bRef) {
    Identifier tid = Identifier.intern(varPrefix + "t");
    Expr tRef = AmbiguousVariableAccess.make(Name.make(tid.toString(), Location.NULL));
    FormalParaDecl tDecl =
      FormalParaDecl.make(Modifiers.NONE, null, tid, exceptionType,
			  Location.NULL);

    StmtVec catchBodyVec = StmtVec.make(2);
    Expr bFalse = BinaryExpr.make(TagConstants.ASSIGN, bRef, GC.falselit, Location.NULL);
    catchBodyVec.addElement(EvalStmt.make(bFalse));
    catchBodyVec.addElement(ThrowStmt.make(tRef, Location.NULL));

    BlockStmt catchBody = BlockStmt.make(catchBodyVec, Location.NULL, Location.NULL);
    return CatchClause.make(tDecl, catchBody, Location.NULL);
  }

  //@ ensures \result != null;
  private static Expr zeroequiv(/*@ non_null */ Type t) {
    switch (t.getTag()) {
    case TagConstants.ARRAYTYPE:
    case TagConstants.NULLTYPE:
    case TagConstants.TYPENAME:
    case TagConstants.TYPESIG:
      return GC.nulllit;

    case TagConstants.BOOLEANTYPE:
      return GC.falselit;

    case TagConstants.INTTYPE:
    case TagConstants.LONGTYPE:
    case TagConstants.CHARTYPE:
    case TagConstants.BYTETYPE:
    case TagConstants.SHORTTYPE:
    case TagConstants.DOUBLETYPE:
    case TagConstants.FLOATTYPE:
      return GC.zerolit;
    }
    /*@ unreachable; */
    Assert.fail("Bad tag");
    return null;
  }
}

class NewBody {
  /**
   *  <pre>
   *  // 'newBody' is the whole thing
   *  {
   *    [super(...);]
   *    boolean b = true;
   *    try {
   *      try {
   *        // 'mainBody' is this block
   *        body;
   *      }
   *      catch (X0 t) {
   *        b = false;
   *        throw t;
   *      }
   *      catch (X1 t) {
   *        b = false;
   *        throw t;
   *      }
   *      catch (Error t) {
   *        b = false;
   *        throw t;
   *      }
   *      catch (RuntimeException t) {
   *        b = false;
   *        throw t;
   *      }
   *    }
   *    finally {
   *      // 'allFinallyBlock' is the body of this finally
   *      if (b) {
   *        // 'normalFinallyBlock' is this block
   *        // checking
   *      }
   *    }
   *  }
   *  </pre>
   */
  BlockStmt newBody;
  BlockStmt mainBody;
  BlockStmt normalFinallyBlock;
  BlockStmt allFinallyBlock;
}
