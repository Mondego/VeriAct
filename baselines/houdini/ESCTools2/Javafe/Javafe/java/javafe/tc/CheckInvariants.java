/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;


import javafe.ast.*;
import javafe.util.Assert;


class CheckInvariants {

  //@ requires sig != null;
  public static void checkTypeDeclOfSig(TypeSig sig) {
    if (sig.state < TypeSig.PARSED) return;

    TypeDecl decl = sig.getTypeDecl();

    // Check super links
    for(int i = 0; i < decl.superInterfaces.size(); i++) {
      TypeName supern = decl.superInterfaces.elementAt(i);
      TypeSig supers = TypeSig.getRawSig(supern);
      if (sig.state >= TypeSig.LINKSRESOLVED)
	Assert.notNull(supers);  //@ nowarn Pre;
      if (supers != null) {
	Assert.notFalse(supers == sig.getEnclosingEnv()  //@ nowarn Pre;
		   .lookupTypeName(null,supern.name)); 
	Assert.notFalse((sig.state < TypeSig.CHECKED		//@ nowarn Pre;
			 && sig.state <= supers.state)
			|| supers.state >= TypeSig.PREPPED);
      }
    }
    if (decl.getTag() == TagConstants.CLASSDECL) {
      TypeName supern = ((ClassDecl)decl).superClass;
      if (sig == Types.javaLangObject()) {
	if (sig.state >= TypeSig.LINKSRESOLVED)
	  Assert.notFalse(supern == null);		//@ nowarn Pre;
      } else if (supern != null) {
	TypeSig supers = TypeSig.getRawSig(supern);
	if (sig.state >= TypeSig.LINKSRESOLVED)
	    Assert.notNull(supers);			//@ nowarn Pre;
	if (supers != null) {
	  Assert.notFalse(supers ==			//@ nowarn Pre;
		sig.getEnclosingEnv().lookupTypeName(null,supern.name));
	  Assert.notFalse((sig.state < TypeSig.CHECKED	//@ nowarn Pre;
			   && sig.state <= supers.state)
			  || supers.state >= TypeSig.PREPPED);
	}
      } else
	  Assert.notFalse(sig.state < TypeSig.LINKSRESOLVED); //@ nowarn Pre;
    }

    for(int i = 0; i < decl.elems.size(); i++) {
      TypeDeclElem e = decl.elems.elementAt(i);

      switch (e.getTag()) {

      case TagConstants.FIELDDECL:
	{
	  FieldDecl f = (FieldDecl)e;
	  checkType(f.type, sig.state >= TypeSig.PREPPED);
	  if (f.init != null) checkExpr(sig, f.init);
	  break;
	}

      case TagConstants.METHODDECL:
      case TagConstants.CONSTRUCTORDECL:
	{
	  RoutineDecl r = (RoutineDecl) e;
	  for(int j = 0; j < r.args.size(); j++)
	    checkType(r.args.elementAt(j).type, sig.state >= TypeSig.PREPPED);
	  for(int j = 0; j < r.raises.size(); j++)
	    checkType(r.raises.elementAt(j), sig.state >= TypeSig.PREPPED);
	  if (r.getTag() == TagConstants.METHODDECL)
	    checkType(((MethodDecl)r).returnType,
		      sig.state >= TypeSig.PREPPED);
	  if (r.body != null) checkStmt(sig, r.body);
	  break;
	}

      case TagConstants.INITBLOCK:
	checkStmt(sig, ((InitBlock)e).block);
	break;

      case TagConstants.CLASSDECL:
      case TagConstants.INTERFACEDECL:
	checkTypeDeclOfSig(TypeSig.getSig((TypeDecl)e));
	break;

      default: Assert.fail("Fall through.");
      }
    }
  }


  //@ requires t != null;
  public static void checkType(Type t, boolean resolved) {
    if (t.getTag() != TagConstants.TYPENAME)
	return;
    TypeSig sig = TypeSig.getRawSig((TypeName) t);
    if (resolved)
	Assert.notNull(sig);			//@ nowarn Pre;
    else
	Assert.notFalse(sig == null);		//@ nowarn Pre;
  }


  //@ requires sig != null && s != null;
  public static void checkStmt(TypeSig sig, Stmt s) {

    // System.out.println("checkStmt: sig.state = "+sig.state);

    switch(s.getTag()) {

    case TagConstants.SWITCHSTMT:
      checkExpr(sig, ((SwitchStmt)s).expr);
      // Fall through
    case TagConstants.BLOCKSTMT:
      {
	GenericBlockStmt block = (GenericBlockStmt)s;
	for(int i = 0; i < block.stmts.size(); i++)
	  checkStmt(sig, block.stmts.elementAt(i));
	return;
      }

    case TagConstants.VARDECLSTMT:
      {
	LocalVarDecl d = ((VarDeclStmt)s).decl;
	if (d.init != null) checkExpr(sig, d.init);
	return;
      }

    case TagConstants.CLASSDECLSTMT:
      {
	ClassDeclStmt cds = (ClassDeclStmt)s;
	checkTypeDeclOfSig(TypeSig.getSig(cds.decl));
	return;
      }

    case TagConstants.WHILESTMT:
      {
	WhileStmt w = (WhileStmt)s;
	checkExpr(sig, w.expr);
	checkStmt(sig, w.stmt);
	return;
      }
      
    case TagConstants.DOSTMT:
      {
	DoStmt d = (DoStmt)s;
	checkExpr(sig, d.expr);
	checkStmt(sig, d.stmt);
	return;
      }
      
    case TagConstants.SYNCHRONIZESTMT:
      {
	SynchronizeStmt ss = (SynchronizeStmt)s;
	checkExpr(sig, ss.expr);
	checkStmt(sig, ss.stmt);
	return;
      }
      
    case TagConstants.EVALSTMT:
      {
	EvalStmt v = (EvalStmt) s;
	checkExpr(sig, v.expr);
	return;
      }

      
    case TagConstants.RETURNSTMT:
      {
	ReturnStmt r = (ReturnStmt)s;
	if( r.expr != null ) checkExpr(sig, r.expr);
	return;
      }

    case TagConstants.THROWSTMT:
      {
	ThrowStmt t = (ThrowStmt)s;
	checkExpr(sig, t.expr);
	return;
      }

    case TagConstants.BREAKSTMT:
    case TagConstants.CONTINUESTMT:
      return;
      
    case TagConstants.LABELSTMT:
      checkStmt(sig, ((LabelStmt)s).stmt);
      return;
      
    case TagConstants.IFSTMT:
      {
	IfStmt i = (IfStmt)s;
	checkExpr(sig, i.expr);
	checkStmt(sig, i.thn);
	checkStmt(sig, i.els);
	return;
      }
      
    case TagConstants.FORSTMT:
      {
	ForStmt f = (ForStmt) s;
	for(int i = 0; i < f.forInit.size(); i++)
	  checkStmt(sig, f.forInit.elementAt(i));
	checkExpr(sig, f.test);
	for(int i = 0; i < f.forUpdate.size(); i++)
	  checkExpr(sig, f.forUpdate.elementAt(i));
	checkStmt(sig, f.body);
	return;
      }

    case TagConstants.SKIPSTMT:
      return;

    case TagConstants.ASSERTSTMT:
      {
	AssertStmt a = (AssertStmt)s;
	checkExpr(sig,a.pred);
	if (a.label != null) checkExpr(sig,a.label);
      }
      return;

    case TagConstants.SWITCHLABEL:
      {
	SwitchLabel sl = (SwitchLabel) s;
	if (sl.expr != null) checkExpr(sig, sl.expr);
	return;
      }

    case TagConstants.TRYFINALLYSTMT:
      {
	TryFinallyStmt tf = (TryFinallyStmt)s;
	checkStmt(sig, tf.tryClause);
	checkStmt(sig, tf.finallyClause);
	return;
      }
      
    case TagConstants.TRYCATCHSTMT:
      {
	TryCatchStmt tc = (TryCatchStmt)s;
	checkStmt(sig, tc.tryClause);
	for(int i = 0; i < tc.catchClauses.size(); i++)
	  checkStmt(sig, tc.catchClauses.elementAt(i).body);
	return;
      }
      
    case TagConstants.CONSTRUCTORINVOCATION:
      {
	ConstructorInvocation ci = (ConstructorInvocation) s;
	if (ci.enclosingInstance != null)
	    checkExpr(sig, ci.enclosingInstance);
	for(int i = 0; i < ci.args.size(); i++)
	  checkExpr(sig, ci.args.elementAt(i));
	if (sig.state < TypeSig.CHECKED) 
	  Assert.notFalse(ci.decl == null);		//@ nowarn Pre;
	else
	  Assert.notNull(ci.decl);			//@ nowarn Pre;
	return;
      }

    default:
      Assert.fail("Switch fall through");
    }
  }


  //@ requires sig != null && expr != null;
  public static void checkExpr(TypeSig sig, VarInit expr) {

    // System.out.println("Checking inv on "+PrettyPrint.inst.toString(expr));

    if( sig.state >= TypeSig.CHECKED ) {
      // All expressions should have types
      if( expr instanceof Expr )
	FlowInsensitiveChecks.getType((Expr)expr);
    }
    
    switch (expr.getTag()) {
    case TagConstants.ARRAYINIT:
      {
	ArrayInit ai = (ArrayInit)expr;
	for(int i = 0; i < ai.elems.size(); i++)
	  checkExpr(sig, ai.elems.elementAt(i));
	return;
      }
      
    case TagConstants.THISEXPR:
    case TagConstants.STRINGLIT:
    case TagConstants.CHARLIT:
    case TagConstants.BOOLEANLIT: 
    case TagConstants.FLOATLIT: case TagConstants.DOUBLELIT: 
    case TagConstants.INTLIT: case TagConstants.LONGLIT:
    case TagConstants.NULLLIT:
      return;
      
    case TagConstants.ARRAYREFEXPR:
      {
        ArrayRefExpr r = (ArrayRefExpr) expr;
	checkExpr(sig, r.array);
	checkExpr(sig, r.index);
	return;
      }

    case TagConstants.NEWINSTANCEEXPR:
      { 
        NewInstanceExpr ne = (NewInstanceExpr) expr;
	if (ne.enclosingInstance != null)
	    checkExpr(sig, ne.enclosingInstance);
	checkType(ne.type, sig.state >= TypeSig.CHECKED);
	for(int i = 0; i < ne.args.size(); i++)
	  checkExpr(sig, ne.args.elementAt(i));

	if (ne.anonDecl != null && sig.state>=TypeSig.CHECKED)
	    checkTypeDeclOfSig(TypeSig.getSig(ne.anonDecl));

//	if (sig.state < TypeSig.CHECKED) Assert.notFalse(ne.decl == null);
//	else Assert.notNull(ne.decl);

	return;
      }

    case TagConstants.NEWARRAYEXPR:
      {
        NewArrayExpr na = (NewArrayExpr) expr;
	checkType(na.type, sig.state >= TypeSig.CHECKED);
	for(int i = 0; i < na.dims.size(); i++)
	  checkExpr(sig, na.dims.elementAt(i));
        return;
      }

    case TagConstants.CONDEXPR:
      {
        CondExpr ce = (CondExpr) expr;
	checkExpr(sig, ce.test);
	checkExpr(sig, ce.thn);
	checkExpr(sig, ce.els);
        return;
      }

    case TagConstants.INSTANCEOFEXPR:
      {
        InstanceOfExpr ie = (InstanceOfExpr) expr;
	checkExpr(sig, ie.expr);
	checkType(ie.type, sig.state >= TypeSig.CHECKED);
        return;
      }

    case TagConstants.CASTEXPR:
      {
        CastExpr ce = (CastExpr) expr;
	checkExpr(sig, ce.expr);
	checkType(ce.type, sig.state >= TypeSig.CHECKED);
        return;
      }

    case TagConstants.CLASSLITERAL:
      {
        ClassLiteral cl = (ClassLiteral) expr;
	checkType(cl.type, sig.state >= TypeSig.CHECKED);
        return;
      }

    case TagConstants.OR: case TagConstants.AND:
    case TagConstants.BITOR: case TagConstants.BITXOR:
    case TagConstants.BITAND: case TagConstants.NE:
    case TagConstants.EQ: case TagConstants.GE:
    case TagConstants.GT: case TagConstants.LE:
    case TagConstants.LT: case TagConstants.LSHIFT:
    case TagConstants.RSHIFT: case TagConstants.URSHIFT:
    case TagConstants.ADD: case TagConstants.SUB:
    case TagConstants.DIV: case TagConstants.MOD:
    case TagConstants.STAR:
    case TagConstants.ASSIGN: case TagConstants.ASGMUL:
    case TagConstants.ASGDIV: case TagConstants.ASGREM:
    case TagConstants.ASGADD: case TagConstants.ASGSUB:
    case TagConstants.ASGLSHIFT: case TagConstants.ASGRSHIFT:
    case TagConstants.ASGURSHIFT: case TagConstants.ASGBITAND:
    case TagConstants.ASGBITOR: case TagConstants.ASGBITXOR:
      {
        BinaryExpr be = (BinaryExpr) expr;
	checkExpr(sig, be.left);
	checkExpr(sig, be.right);
        return;
      }

    case TagConstants.UNARYSUB: case TagConstants.UNARYADD:
    case TagConstants.NOT: case TagConstants.BITNOT:
    case TagConstants.INC: case TagConstants.DEC:
    case TagConstants.POSTFIXINC: case TagConstants.POSTFIXDEC:
      checkExpr(sig, ((UnaryExpr)expr).expr);
      return;

    case TagConstants.PARENEXPR:
      checkExpr(sig, ((ParenExpr)expr).expr);
      return;

    case TagConstants.AMBIGUOUSVARIABLEACCESS:
      Assert.notFalse(sig.state < TypeSig.CHECKED);	//@ nowarn Pre;
      return;

    case TagConstants.VARIABLEACCESS:
      Assert.notFalse(sig.state >= TypeSig.CHECKED);	//@ nowarn Pre;
      Assert.notNull(((VariableAccess)expr).decl);	//@ nowarn Pre;
      return;

    case TagConstants.FIELDACCESS:
      {
	FieldAccess xp = (FieldAccess)expr;
	checkObjectDesignator( sig, xp.od );
	Assert.notFalse(				//@ nowarn Pre;
	    (sig.state < TypeSig.CHECKED && xp.decl == null)
			|| xp.decl != null);
	return;
      }
      
    case TagConstants.AMBIGUOUSMETHODINVOCATION:
      {
	Assert.notFalse(sig.state < TypeSig.CHECKED);	//@ nowarn Pre;
	AmbiguousMethodInvocation ami = (AmbiguousMethodInvocation)expr;
	for(int i = 0; i < ami.args.size(); i++)
	  checkExpr(sig, ami.args.elementAt(i));
	return;
      }

    case TagConstants.METHODINVOCATION:
      {
	MethodInvocation mi = (MethodInvocation)expr;
	checkObjectDesignator( sig, mi.od );
	for(int i = 0; i < mi.args.size(); i++)
	  checkExpr(sig, mi.args.elementAt(i));
	Assert.notFalse((sig.state < TypeSig.CHECKED	//@ nowarn Pre;
			&& mi.decl == null)
			|| mi.decl != null);
	return;
      }

    default: Assert.fail("Unexpected tag "+expr.getTag());
    }
  }


  //@ requires sig != null && od != null;
  public static void checkObjectDesignator(TypeSig sig, ObjectDesignator od) {
    switch (od.getTag()) {

    case TagConstants.EXPROBJECTDESIGNATOR:
      {
	ExprObjectDesignator eod = (ExprObjectDesignator)od;
	checkExpr(sig, eod.expr);
	break;
      }
      
    case TagConstants.TYPEOBJECTDESIGNATOR:
      {
	Assert.notFalse(sig.state >= TypeSig.CHECKED);	//@ nowarn Pre;
	TypeObjectDesignator tod = (TypeObjectDesignator)od;
	checkType(tod.type, sig.state >= TypeSig.CHECKED);
	Assert.notFalse(tod.type instanceof TypeName || 
			tod.type instanceof TypeSig);
	break;
      }

    case TagConstants.SUPEROBJECTDESIGNATOR:
      {
	SuperObjectDesignator sod = (SuperObjectDesignator)od;
	break;
      }

    default: Assert.fail("Unexpected tag "+od.getTag());
    }
  }



}
