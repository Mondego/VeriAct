package javafe.ast;

//@ model import javafe.tc.TagConstants;

/**
 ** Represents either a ClassBodyDeclaration or an
 ** InterfaceMemberDeclaration.
 **/

public interface TypeDeclElem {

    /** Return the tag of a node. */
  //@ ensures (this instanceof LiteralExpr) ==> \result==((LiteralExpr)this).tag;
  //@ ensures (this instanceof BinaryExpr) ==> \result==((BinaryExpr)this).op;
  //@ ensures (this instanceof UnaryExpr) ==> \result==((UnaryExpr)this).op;
  //@ ensures (this instanceof PrimitiveType) ==>\result==((PrimitiveType)this).tag;
  //
  //@ ensures (\result==javafe.tc.TagConstants.TYPESIG) ==> \typeof(this) <: \type(javafe.tc.TypeSig);
  //
  // Remaining part of spec is automatically generated:
  //@ ensures (\result==TagConstants.COMPILATIONUNIT) ==> \typeof(this) <: \type(javafe.ast.CompilationUnit);
  //@ ensures (\result==TagConstants.SINGLETYPEIMPORTDECL) ==> \typeof(this) <: \type(javafe.ast.SingleTypeImportDecl);
  //@ ensures (\result==TagConstants.ONDEMANDIMPORTDECL) ==> \typeof(this) <: \type(javafe.ast.OnDemandImportDecl);
  //@ ensures (\result==TagConstants.CLASSDECL) ==> \typeof(this) <: \type(javafe.ast.ClassDecl);
  //@ ensures (\result==TagConstants.INTERFACEDECL) ==> \typeof(this) <: \type(javafe.ast.InterfaceDecl);
  //@ ensures (\result==TagConstants.CONSTRUCTORDECL) ==> \typeof(this) <: \type(javafe.ast.ConstructorDecl);
  //@ ensures (\result==TagConstants.METHODDECL) ==> \typeof(this) <: \type(javafe.ast.MethodDecl);
  //@ ensures (\result==TagConstants.INITBLOCK) ==> \typeof(this) <: \type(javafe.ast.InitBlock);
  //@ ensures (\result==TagConstants.LOCALVARDECL) ==> \typeof(this) <: \type(javafe.ast.LocalVarDecl);
  //@ ensures (\result==TagConstants.FIELDDECL) ==> \typeof(this) <: \type(javafe.ast.FieldDecl);
  //@ ensures (\result==TagConstants.FORMALPARADECL) ==> \typeof(this) <: \type(javafe.ast.FormalParaDecl);
  //@ ensures (\result==TagConstants.BLOCKSTMT) ==> \typeof(this) <: \type(javafe.ast.BlockStmt);
  //@ ensures (\result==TagConstants.SWITCHSTMT) ==> \typeof(this) <: \type(javafe.ast.SwitchStmt);
  //@ ensures (\result==TagConstants.ASSERTSTMT) ==> \typeof(this) <: \type(javafe.ast.AssertStmt);
  //@ ensures (\result==TagConstants.VARDECLSTMT) ==> \typeof(this) <: \type(javafe.ast.VarDeclStmt);
  //@ ensures (\result==TagConstants.CLASSDECLSTMT) ==> \typeof(this) <: \type(javafe.ast.ClassDeclStmt);
  //@ ensures (\result==TagConstants.WHILESTMT) ==> \typeof(this) <: \type(javafe.ast.WhileStmt);
  //@ ensures (\result==TagConstants.DOSTMT) ==> \typeof(this) <: \type(javafe.ast.DoStmt);
  //@ ensures (\result==TagConstants.SYNCHRONIZESTMT) ==> \typeof(this) <: \type(javafe.ast.SynchronizeStmt);
  //@ ensures (\result==TagConstants.EVALSTMT) ==> \typeof(this) <: \type(javafe.ast.EvalStmt);
  //@ ensures (\result==TagConstants.RETURNSTMT) ==> \typeof(this) <: \type(javafe.ast.ReturnStmt);
  //@ ensures (\result==TagConstants.THROWSTMT) ==> \typeof(this) <: \type(javafe.ast.ThrowStmt);
  //@ ensures (\result==TagConstants.BREAKSTMT) ==> \typeof(this) <: \type(javafe.ast.BreakStmt);
  //@ ensures (\result==TagConstants.CONTINUESTMT) ==> \typeof(this) <: \type(javafe.ast.ContinueStmt);
  //@ ensures (\result==TagConstants.LABELSTMT) ==> \typeof(this) <: \type(javafe.ast.LabelStmt);
  //@ ensures (\result==TagConstants.IFSTMT) ==> \typeof(this) <: \type(javafe.ast.IfStmt);
  //@ ensures (\result==TagConstants.FORSTMT) ==> \typeof(this) <: \type(javafe.ast.ForStmt);
  //@ ensures (\result==TagConstants.SKIPSTMT) ==> \typeof(this) <: \type(javafe.ast.SkipStmt);
  //@ ensures (\result==TagConstants.SWITCHLABEL) ==> \typeof(this) <: \type(javafe.ast.SwitchLabel);
  //@ ensures (\result==TagConstants.TRYFINALLYSTMT) ==> \typeof(this) <: \type(javafe.ast.TryFinallyStmt);
  //@ ensures (\result==TagConstants.TRYCATCHSTMT) ==> \typeof(this) <: \type(javafe.ast.TryCatchStmt);
  //@ ensures (\result==TagConstants.CONSTRUCTORINVOCATION) ==> \typeof(this) <: \type(javafe.ast.ConstructorInvocation);
  //@ ensures (\result==TagConstants.CATCHCLAUSE) ==> \typeof(this) <: \type(javafe.ast.CatchClause);
  //@ ensures (\result==TagConstants.ARRAYINIT) ==> \typeof(this) <: \type(javafe.ast.ArrayInit);
  //@ ensures (\result==TagConstants.THISEXPR) ==> \typeof(this) <: \type(javafe.ast.ThisExpr);
  //@ ensures (\result==TagConstants.ARRAYREFEXPR) ==> \typeof(this) <: \type(javafe.ast.ArrayRefExpr);
  //@ ensures (\result==TagConstants.NEWINSTANCEEXPR) ==> \typeof(this) <: \type(javafe.ast.NewInstanceExpr);
  //@ ensures (\result==TagConstants.NEWARRAYEXPR) ==> \typeof(this) <: \type(javafe.ast.NewArrayExpr);
  //@ ensures (\result==TagConstants.CONDEXPR) ==> \typeof(this) <: \type(javafe.ast.CondExpr);
  //@ ensures (\result==TagConstants.INSTANCEOFEXPR) ==> \typeof(this) <: \type(javafe.ast.InstanceOfExpr);
  //@ ensures (\result==TagConstants.CASTEXPR) ==> \typeof(this) <: \type(javafe.ast.CastExpr);
  //@ ensures (\result==TagConstants.PARENEXPR) ==> \typeof(this) <: \type(javafe.ast.ParenExpr);
  //@ ensures (\result==TagConstants.AMBIGUOUSVARIABLEACCESS) ==> \typeof(this) <: \type(javafe.ast.AmbiguousVariableAccess);
  //@ ensures (\result==TagConstants.VARIABLEACCESS) ==> \typeof(this) <: \type(javafe.ast.VariableAccess);
  //@ ensures (\result==TagConstants.FIELDACCESS) ==> \typeof(this) <: \type(javafe.ast.FieldAccess);
  //@ ensures (\result==TagConstants.AMBIGUOUSMETHODINVOCATION) ==> \typeof(this) <: \type(javafe.ast.AmbiguousMethodInvocation);
  //@ ensures (\result==TagConstants.METHODINVOCATION) ==> \typeof(this) <: \type(javafe.ast.MethodInvocation);
  //@ ensures (\result==TagConstants.CLASSLITERAL) ==> \typeof(this) <: \type(javafe.ast.ClassLiteral);
  //@ ensures (\result==TagConstants.EXPROBJECTDESIGNATOR) ==> \typeof(this) <: \type(javafe.ast.ExprObjectDesignator);
  //@ ensures (\result==TagConstants.TYPEOBJECTDESIGNATOR) ==> \typeof(this) <: \type(javafe.ast.TypeObjectDesignator);
  //@ ensures (\result==TagConstants.SUPEROBJECTDESIGNATOR) ==> \typeof(this) <: \type(javafe.ast.SuperObjectDesignator);
  //@ ensures (\result==TagConstants.ERRORTYPE) ==> \typeof(this) <: \type(javafe.ast.ErrorType);
  //@ ensures (\result==TagConstants.TYPENAME) ==> \typeof(this) <: \type(javafe.ast.TypeName);
  //@ ensures (\result==TagConstants.ARRAYTYPE) ==> \typeof(this) <: \type(javafe.ast.ArrayType);
  //@ ensures (\result==TagConstants.SIMPLENAME) ==> \typeof(this) <: \type(javafe.ast.SimpleName);
  //@ ensures (\result==TagConstants.COMPOUNDNAME) ==> \typeof(this) <: \type(javafe.ast.CompoundName);
  //@ ensures (\result==TagConstants.OR) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.AND) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.BITOR) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.BITXOR) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.BITAND) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.NE) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.EQ) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.GE) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.GT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.LE) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.LT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.LSHIFT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.RSHIFT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.URSHIFT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ADD) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.SUB) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.DIV) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.MOD) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.STAR) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASSIGN) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGMUL) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGDIV) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGREM) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGADD) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGSUB) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGLSHIFT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGRSHIFT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGURSHIFT) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGBITAND) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGBITOR) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.ASGBITXOR) ==> \typeof(this) <: \type(javafe.ast.BinaryExpr);
  //@ ensures (\result==TagConstants.UNARYADD) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.UNARYSUB) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.NOT) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.BITNOT) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.INC) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.DEC) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.POSTFIXINC) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.POSTFIXDEC) ==> \typeof(this) <: \type(javafe.ast.UnaryExpr);
  //@ ensures (\result==TagConstants.BOOLEANTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.INTTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.LONGTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.CHARTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.FLOATTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.DOUBLETYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.VOIDTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.NULLTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.BYTETYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.SHORTTYPE) ==> \typeof(this) <: \type(javafe.ast.PrimitiveType);
  //@ ensures (\result==TagConstants.BOOLEANLIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
  //@ ensures (\result==TagConstants.INTLIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
  //@ ensures (\result==TagConstants.LONGLIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
  //@ ensures (\result==TagConstants.CHARLIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
  //@ ensures (\result==TagConstants.FLOATLIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
  //@ ensures (\result==TagConstants.DOUBLELIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
  //@ ensures (\result==TagConstants.STRINGLIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
  //@ ensures (\result==TagConstants.NULLLIT) ==> \typeof(this) <: \type(javafe.ast.LiteralExpr);
    public int getTag();

    public void accept(/*@ non_null */ Visitor v);

    public /*@ non_null */ Object accept(/*@ non_null */ VisitorArgResult v, /*@ nullable */ Object o);	

    /**
     ** Do we have a parent field? <p>
     **
     ** A TypeDecl "invariant" requires all of its TypeDeclElem elements
     ** to have a parent. <p>
     **
     ** Known cases without parents:<p>
     **
     **   (a) non-member type TypeDecl's
     **   (b) TypeDeclElems before they are installed in a TypeDecl
     **   (c) The length FieldDecl  (belongs to no TypeDecl)
     **/
     //@ ghost public boolean hasParent;

    /**
     ** The TypeDecl we are an element of, or null if we do not have a
     **  parent (cf. hasParent).
     **/
    //@ ensures hasParent ==> \result!=null;
    public TypeDecl getParent();

    public void setParent(/*@non_null*/ TypeDecl p);

    public int getModifiers();
    public void setModifiers(int m);
    public ModifierPragmaVec getPModifiers();
	
    public int getStartLoc();
}

