// -*- mode: java -*-
/* Copyright 2000, 2001, Compaq Computer Corporation */

/* IF THIS IS A JAVA FILE, DO NOT EDIT IT!  

   Most Java files in this directory which are part of the Javafe AST
   are automatically generated using the astgen comment (see
   ESCTools/Javafe/astgen) from the input file 'hierarchy.h'.  If you
   wish to modify AST classes or introduce new ones, modify
   'hierarchy.j.'
 */

package javafe.ast;

import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.ErrorSet;
import javafe.tc.TagConstants;

// Convention: unless otherwise noted, integer fields named "loc" refer
// to the location of the first character of the syntactic unit


/**
 * <code>ASTNode</code> is the root of the abstract syntax tree node
 * hierarchy.  Every class representing a node in the AST is a (direct
 * or indirect) subclass of this class.
 *
 * <p> In designing our AST classes, the following, broad approach was
 * taken:
 *
 * <ul>
 * 
 * <li> <b>Design package javafe.ast to stand alone.</b>
 * 
 * <li> <b>Have a large, deep hierarchy, using the type system to
 * encode many invariants.</b> (Used a program to generate boilerplate
 * code, making it the large hierarchy more manageable.)
 * 
 * <li> <b>Represent children with public fields.</b>
 * 
 * <li> <b>Put most code outside AST classes.</b>
 * 
 * <li> <b>Support both switch-based and visitor-based traversal.</b>
 * 
 * <li> <b>Liberal use of locations.</b> AST classes contain location
 * fields that indicate the corresponding part of the source program.
 * 
 * <li> <b>Use generic decorations.</b>
 * 
 * <li> <b>Rewrite tree to handle disambiguation.</b> Expression names
 * and method names are ambiguous: the be of the form
 * <i>package.type.members</i> or <i>type.members</i> or
 * <i>local-variable.members</i> or even <i>members</i>.  The parser
 * does not attempt to disambiguate among these, but rather generates
 * either <code>ExprName</code> or <code>MethodName</code>.  A later
 * pass replaces these nodes with either <code>VariableAccess</code>
 * or concrete subclasses of <code>FieldAccess</code> or
 * <code>MethodInvocation</code>.
 * 
 * <li> <b>Fill in fields and decorations to handle resolution.</b>
 * The nodes that need to be resolved are: <code>TypeName</code>,
 * <code>VariableAccess</code>, (three subclasses of)
 * <code>FieldAccess</code>, (three subclasses of)
 * <code>MethodInvocation</code>, <code>ConstructorInvocation</code>,
 * and <code>NewInstanceExpr</code>.  All nodes except
 * <code>TypeName</code> have a field named <code>decl</code> of the
 * approriate type that is initialized to <code>null</code> and is
 * filled in by name resolution.  For <code>TypeName</code> we use a
 * decoration rather than a field to avoid a dependency on the
 * <code>javafe.tc</code> package.
 * 
 * </ul>
 * 
 * <p> Measurements (circa SRC ESC/Java):
 * <blockquote>
 * Node abstract classes: 13
 * <br> Node concrete classes: 59
 * <br> Sequence classes: 12
 * <br> Total classes: 84
 * <br>
 * <br> Input to code generator: 842 lines of code (513 excluding comments)
 * <br> Code generator: 993 (675)
 * <br> Total lines of code: 1835 (1188)
 * <br> Output of code generator: 4298
 * </blockquote>
 * 
 * <p> The complete list of subclasses is shown below.  The names of
 * subclasses of this class correspond (in general) to the syntactic
 * units that they represent.  Any exceptions to this rule are
 * documented.  All non-leaf classes are abstract.  In parenthesis,
 * the names and types of fields holding the locally-defined children
 * classes are listed; a postfix <code>*</code> on a type indicates a
 * sequence.  Square brackets indicate optional elements; the only
 * these fields can be null.
 * 
 * <pre>
 * ASTNode
 *    CompilationUnit ([Name pkgName], [LexicalPragma* lexicalPragmas], ImportDecl* imports, TypeDecl* elems)
 *    ImportDecl ()
 *       SingleTypeImport (TypeName typeName)
 *       OnDemandImport (Name pkgName)
 *    TypeDecl (int modifiers, [ModifierPragma* pmodifiers], Identifier id, [TypeModifierPragma* tmodifiers], TypeName* superInterfaces, TypeDeclElem* elems)
 *       ClassDecl ([TypeName superClass])
 *       InterfaceDecl ()
 *    TypeDeclElem () NOTE: This is an <em>interface</em>
 *       TypeDecl
 *       FieldDecl
 *       RoutineDecl (int modifiers, [ModifierPragma* pmodifiers], FormalParaDecl* args, TypeName* raises, [BlockStmt body])
 *          ConstructorDecl ([TypeModifierPragma* tmodifiers])
 *          MethodDecl (Identifier id, Type returnType)
 *       InitBlock (int modifiers, [ModifierPragma* pmodifiers], BlockStmt block)
 *       TypeDeclElemPragma ()
 *    GenericVarDecl (int modifiers, [ModifierPragma* pmodifiers], Identifier id, Type type)
 *       LocalVarDecl ([VarInit init])
 *       FieldDecl ([VarInit init])
 *       FormalParaDecl ()
 *    Stmt ()
 *       GenericBlockStmt (Stmt* stmts)
 *          BlockStmt ()
 *          SwitchStmt (Expr expr)
 *       AssertStmt (Expr expr, String l)
 *       VarDeclStmt (LocalVarDecl decl)
 *       ClassDeclStmt (ClassDecl anonDecl)
 *       WhileStmt (Expr expr, Stmt stmt)
 *       DoStmt (Expr expr, Stmt stmt)
 *       SynchronizeStmt (Expr expr, BlockStmt stmt)
 *       EvalStmt (Expr expr)
 *       ReturnStmt ([Expr expr])
 *       ThrowStmt (Expr expr)
 *       BranchStmt ([Identifier label])
 *          BreakStmt ()
 *          ContinueStmt ()
 *       LabelStmt (Identifier label, Stmt stmt)
 *       IfStmt (Expr expr, Stmt thn, Stmt els)
 *       ForStmt (Stmt* forInit, Expr test, Expr* forUpdate, Stmt body)
 *       SkipStmt ()
 *       SwitchLabel ([Expr expr])
 *       TryFinallyStmt (Stmt tryClause, Stmt finallyClause)
 *       TryCatchStmt (Stmt tryClause, CatchClause* catchClauses)
 *       StmtPragma ()
 *       ConstructorInvocation (boolean superCall, [Expr enclosingInstance], Expr* args)
 *    CatchClause (FormalParaDecl arg, Stmt body)
 *    VarInit ()
 *       ArrayInit (VarInit* elems)
 *       Expr ()
 *          ThisExpr (Type classPrefix)
 *          LiteralExpr (int tag, [Object value])
 *          ArrayRefExpr (Expr array, Expr index)
 *          NewInstanceExpr ([Expr enclosingInstance], TypeName type, Expr* args, [ClassDecl decl])
 *          NewArrayExpr (Type type, Expr* dims, [ArrayInit init])
 *          CondExpr (Expr test, Expr thn, Expr els)
 *          InstanceOfExpr (Expr expr, Type type)
 *          CastExpr (Expr expr, Type type)
 *          BinaryExpr (int op, Expr left, Expr right)
 *          UnaryExpr (int op, Expr expr)
 *          ParenExpr (Expr expr)
 *          AmbiguousVariableAccess (Name name)
 *          VariableAccess (Identifier id)
 *          FieldAccess (ObjectDesignator od, Identifier id)
 *          AmbiguousMethodInvocation (Name name, Expr* args)
 *          MethodInvocation (ObjectDesignator od, Identifier id, Expr* args)
 *          ClassLiteral (Type type)
 *    ObjectDesignator ()  // "expr.", "type." or "super."
 *       ExprObjectDesignator (Expr expr)
 *       TypeObjectDesignator (Type type)
 *       SuperObjectDesignator ()
 *    Type ()
 *       ErrorType() // was previously represented as a PrimitiveType
 *       PrimitiveType (int tag)
 *         JavafePrimitiveType  ()  // this is a new type introduced to allow for other (e.g., esc) primitive types
 *       TypeName (Name name)
 *       ArrayType (Type elemType)
 *    Name ()
 *       SimpleName (Identifier id)
 *       CompoundName (Identifier* ids)
 *    ModifierPragma ()
 *    LexicalPragma ()
 *    TypeModifierPragma ()
 * </pre>
 */

public abstract class ASTNode implements Cloneable
{
  /**
   * The decorations that have been attached to this node.  This is a
   * package-level field accessed by the <code>ASTDecoration</code>
   * class.  There are different ways to implement decorations; this
   * way is not space-efficient, but it's pretty fast.
   */

  //@ invariant decorations != null ==> (\typeof(decorations) == \type(Object[]));
  /*@spec_public*/ private /*@nullable*/ Object[] decorations;  

    public int bogusField;
    //@ public model int startLoc;
    //@ public model boolean isInternal;
    //@ public represents isInternal <- startLoc == Location.NULL;

    /*@ public normal_behavior
      @ ensures \result == startLoc;
      @*/
    public /*@ pure @*/ abstract int getStartLoc();

    public /*@ pure @*/ int getEndLoc() { 
	return getStartLoc();
    }

    //@ public model int endLoc;
    //@ public represents endLoc <- getEndLoc();
    //+@ public invariant startLoc == Location.NULL <==> endLoc == Location.NULL;    

  /** returns true iff this in an internally declared node. Note
   * that internally declared types have no associated location.
   */
  /*@ public normal_behavior
    @ ensures \result == isInternal;
    @*/
    public /*@ pure @*/ boolean isInternal() {
	return getStartLoc() == Location.NULL;
    }

  //@ ensures \typeof(\result) == \typeof(this);
  //@ ensures \result.owner == null;
  public /*@ non_null */ Object clone(boolean b)  {
    ASTNode n = null;
    try {
      n = (ASTNode)super.clone();
      if (!b) {
        n.decorations = null;
      }
    } catch(java.lang.CloneNotSupportedException e) {
      ErrorSet.fatal("Internal error in AST hierarchy: no clone");
      n = this; // dummy assingment to appease escjava.
    }
    return n;
  }
 
  /**
   * Clone node, where the clone has the same decorations as original.
   */
  public /*@non_null*/ Object clone() {
    return clone(true);
  }
  
  public /*@nullable*/ Object[] getDecorations() {
    return decorations;
  }

  //@ requires d != null ==> \typeof(d) == \type(Object[]);
  public void setDecorations(/*@nullable*/ Object d[]) {
    decorations = d;
  }
    
  //$$
  static public int dotCounter = 0 ;
  public int dotId;

    /*
     * Constructor needed for GRAPHICAL representation (dot output) 
     *
     * This is temporary as I don't know how the class generator works.
     * And it always generate a constructor empty (which classes with this one
     * if I remove this useless boolean).
     */
//   protected ASTNode(){
//     dotCounter += 1;
//     dotId = dotCounter;
//   }    
  //$$



// Generated boilerplate constructors:

   protected ASTNode() {
   }
   

// Generated boilerplate methods:

   /** Return the number of children a node has. */
   //@ ensures \result >= 0;
   public abstract int childCount();

   /** Return the first-but-ith child of a node. */
   //@ requires 0 <= i;
   public abstract /*@ nullable */ Object childAt(int i);

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
   public abstract int getTag();

   /** Return a string representation of <code>this</code>.
   Meant for debugging use only, not for presentation. */
   public abstract /*@non_null*/ String toString();

   /** Accept a visit from <code>v</code>.  This method simply
   calls the method of <code>v</code> corresponding to the
   allocated type of <code>this</code>, passing <code>this</code>
   as the argument.  See the design patterns book. */
   //@ requires v != null;
   public abstract void accept(/*@non_null*/ Visitor v);

   //@ requires v != null;
//@ ensures \result != null;
public abstract /*@non_null*/ Object accept(/*@non_null*/ VisitorArgResult v, /*@nullable*/Object o);

   public void check() {
   }

}
