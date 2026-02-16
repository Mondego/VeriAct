// -*- mode: java -*-
/* Copyright 2000, 2001, Compaq Computer Corporation */

/* IF THIS IS A JAVA FILE, DO NOT EDIT IT!  

   Most Java files in this directory which are part of the Javafe AST
   are automatically generated using the astgen comment (see
   ESCTools/Javafe/astgen) from the input file 'hierarchy.h'.  If you
   wish to modify AST classes or introduce new ones, modify
   'hierarchy.j.'
 */

package escjava.ast;

import javafe.ast.ASTDecoration;
import javafe.ast.ASTDotVisitor;
import javafe.ast.ASTNode;
import javafe.ast.AmbiguousMethodInvocation;
import javafe.ast.AmbiguousVariableAccess;
import javafe.ast.ArrayInit;
import javafe.ast.ArrayRefExpr;
import javafe.ast.ArrayType;
import javafe.ast.AssertStmt;
import javafe.ast.BinaryExpr;
import javafe.ast.BlockStmt;
import javafe.ast.BranchStmt;
import javafe.ast.BreakStmt;
import javafe.ast.CastExpr;
import javafe.ast.CatchClause;
import javafe.ast.CatchClauseVec;
import javafe.ast.ClassDecl;
import javafe.ast.ClassDeclStmt;
import javafe.ast.ClassLiteral;
import javafe.ast.CompilationUnit;
import javafe.ast.CompoundName;
import javafe.ast.CondExpr;
import javafe.ast.ConstructorDecl;
import javafe.ast.ConstructorInvocation;
import javafe.ast.ContinueStmt;
import javafe.ast.DefaultVisitor;
import javafe.ast.DelegatingPrettyPrint;
import javafe.ast.DoStmt;
import javafe.ast.ErrorType;
import javafe.ast.EvalStmt;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.FieldDeclVec;
import javafe.ast.ForStmt;
import javafe.ast.FormalParaDecl;
import javafe.ast.FormalParaDeclVec;
// import javafe.ast.GeneratedTags;
import javafe.ast.GenericBlockStmt;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.IdPragma;
import javafe.ast.Identifier;
import javafe.ast.IdentifierNode;
import javafe.ast.IdentifierVec;
import javafe.ast.IfStmt;
import javafe.ast.ImportDecl;
import javafe.ast.ImportDeclVec;
import javafe.ast.InitBlock;
import javafe.ast.InstanceOfExpr;
import javafe.ast.InterfaceDecl;
import javafe.ast.JavafePrimitiveType;
import javafe.ast.LabelStmt;
import javafe.ast.LexicalPragma;
import javafe.ast.LexicalPragmaVec;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.LocalVarDeclVec;
import javafe.ast.MethodDecl;
import javafe.ast.MethodDeclVec;
import javafe.ast.MethodInvocation;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.Modifiers;
import javafe.ast.Name;
import javafe.ast.NewArrayExpr;
import javafe.ast.NewInstanceExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.OnDemandImportDecl;
import javafe.ast.OperatorTags;
import javafe.ast.ParenExpr;
import javafe.ast.PrettyPrint;
import javafe.ast.PrimitiveType;
import javafe.ast.ReturnStmt;
import javafe.ast.RoutineDecl;
import javafe.ast.SimpleName;
import javafe.ast.SingleTypeImportDecl;
import javafe.ast.SkipStmt;
import javafe.ast.StandardPrettyPrint;
import javafe.ast.Stmt;
import javafe.ast.StmtPragma;
import javafe.ast.StmtVec;
import javafe.ast.SuperObjectDesignator;
import javafe.ast.SwitchLabel;
import javafe.ast.SwitchStmt;
import javafe.ast.SynchronizeStmt;
// import javafe.ast.TagConstants;
import javafe.ast.ThisExpr;
import javafe.ast.ThrowStmt;
import javafe.ast.TryCatchStmt;
import javafe.ast.TryFinallyStmt;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElem;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.TypeDeclVec;
import javafe.ast.TypeModifierPragma;
import javafe.ast.TypeModifierPragmaVec;
import javafe.ast.TypeName;
import javafe.ast.TypeNameVec;
import javafe.ast.TypeObjectDesignator;
import javafe.ast.UnaryExpr;
import javafe.ast.Util;
import javafe.ast.VarDeclStmt;
import javafe.ast.VarInit;
import javafe.ast.VarInitVec;
import javafe.ast.VariableAccess;
// import javafe.ast.Visitor;
// import javafe.ast.VisitorArgResult;
import javafe.ast.WhileStmt;

import java.util.Hashtable;
import java.util.Set;
import java.util.ArrayList;

import javafe.util.Assert;
import javafe.util.Location;
import escjava.ParsedRoutineSpecs;

// Convention: unless otherwise noted, integer fields named "loc" refer
// to the location of the first character of the syntactic unit

//# TagBase javafe.tc.TagConstants.LAST_TAG + 1
//# VisitorRoot javafe.ast.Visitor

//# EndHeader

/**
 * The files in this package extend the AST classes defined in
 * <code>javafe.ast</code>.  The following diagram shows how the new
 * classes related to the old classes in the class hierarchy:
 *
 * <pre>
 * - ASTNode
 *    - VarInit ()
 *       - Expr ()
 *         + GCExpr
 *           + LabelExpr (Identifier label, Expr expr)
 *           + NaryExpr (int op, Identifier methodName, Expr* exprs)
 *           + QuantifiedExpr (GenericVarDecl* vars, Expr rangeExpr, Expr expr)
 *           + GeneralizedQuantifiedExpr (GenericVarDecl* vars, Expr expr)
 *                // Sum, Product, Max, Min
 *           + NumericalQuantifiedExpr (GenericVarDecl* vars, Expr expr)
 *                // NumOf
 *           + SubstExpr (GenericVarDecl var, Expr val, Expr target)
 *           + TypeExpr (Type type)
 *         + EverythingExpr ()
 *         + LockSetExpr ()
 *         + NotSpecifiedExpr ()
 *         + NothingExpr ()
 *         + StoreRefExpr(Expr,Expr)
 *         + NotModifiedExpr(Expr)
 *         + ResExpr ()
 *	   + SetCompExpr(Type type, Type typeBound, Identifier id, Expr e)
 *         + WildRefExpr (Expr expr)
 *         + GuardExpr (Expr expr)
 *         + DefPredLetExpr (DefPred* preds, Expr body)
 *         + DefPredApplExpr (Identifier predId, Expr* args)
 *	   + ArrayRangeRefExpr(Expr, Expr, Expr)
 *    - Type ()
 *      - PrimitiveType (tag)
 *        + EscPrimitiveType ()
 *    + GuardedCmd
 *      + SimpleCmd (int cmd) // Skip, Raise
 *      + ExprCmd (int cmd, Expr pred) // Assert, Assume
 *      + AssignCmd (VariableAccess v, Expr rhs)
 *        + GetsCmd ()
 *        + SubGetsCmd (Expr index)
 *        + SubSubGetsCmd (Expr index1, Expr index2)
 *        + RestoreFromCmd ()
 *      + VarInCmd (GenericVarDecl v*, GuardedCmd g)
 *      + DynInstCmd (String s, GuardedCmd g)
 *      + SeqCmd (GuardedCmd cmds*)
 *      + LoopCmd (Condition invariants*, DecreasesInfo decreases*,
 *                 LocalVarDecl skolemConstants*, Expr predicates*,
 *                 GenericVarDecl tempVars*, GuardedCmd guard, GuardedCmd body)
 *      + CmdCmdCmd (int cmd, GuardedCmd g1, GuardedCmd g2)// Try, Choose
 *      + Call (RoutineDecl rd, Expr* args, TypeDecl scope)
 *    - TypeDeclElem ()
 *       - TypeDeclElemPragma ()
 *         + ExprDeclPragma (Expr expr) // Axiom, ObjectInvariant
 *	   + GhostDeclPragma (GhostFieldDecl decl)
 *	   + ModelDeclPragma (ModelFieldDecl decl)
 *         + ModelTypePragma (TypeDecl decl)
 *         + NamedExprDeclPragma (Expr target, Expr expr)
 *         + IdExprDeclPragma (Id target, Expr expr)
 *         + ModelMethodDeclPragma (MethodDecl decl)
 *         + ModelConstructorPragma (ConstructorDecl decl)
 *         + StillDeferredDeclPragma (Identifier var)
 *         + DependsPragma (Expr* exprs) // Depends clause
 *    - Stmt ()
 *       - StmtPragma ()
 *         + SimpleStmtPragma () // Unreachable
 *         + ExprStmtPragma (Expr expr, Expr label) 
 *             // Assume, Assert, LoopInvariant, LoopPredicate
 *         + SetStmtPragma (Expr target, Expr value) 
 *         + SkolemConstantPragma (LocalVarDecl* decl)
 *    - ModifierPragma ()
 *         + SimpleModifierPragma () 
 *                   // Uninitialized, Monitored, NonNull, WritableDeferred,
 *                   // Helper, \Peer, \ReadOnly, \Rep,
 *                   // non_null, nullable, nullable_by_default, non_null_by_default, 
 *                   // obs_pure,
 *                   // code_java_math, code_safe_math, code_bigint_math,
 *                   // spec_java_math, spec_safe_math, spec_bigint_math
 *	   + NestedModifierPragma (ArrayList list)
 *         + ExprModifierPragma (Expr expr) 
 *                   // DefinedIf, Writable, Requires, Pre, Ensures, Post,
 *                   // AlsoEnsures, MonitoredBy, Constraint, InvariantFor, Space, 
 *                   // \Duration, \WorkingSpace,
 *                   // \java_math, \safe_math, \bigint_math
 *         + IdentifierModifierPramga (Identifier id)
 *                   // IsInitialized
 *         + ReachModifierPragma (Expr expr, Identifier id, StoreRefExpr)
 *                   // \Reach
 *	   + CondExprModifierPragma (Expr expr, Expr cond) 
 *                   // Modifies, AlsoModifiers, Assignable, Modifiable
 *         + ModifiesGroupPragma
 *                   // Group of Modifies, etc., pragmas from one spec case
 *         + MapsExprModifierPragma (Identifier id, Expr mapsexpr, Expr expr) 
 *		     // maps
 *         + VarExprModifierPragma (GenericVarDecl arg, Expr expr)
 *                   // Exsures, AlsoExsures, Signals, AlsoSignals
 *         + ModelProgramModifierPragma()
 *	   + VarDeclModifierPragma (LocalVarDecl decl)
 *    - LexicalPragma ()
 *      + NowarnPragma (Identifier* checks)
 *      + ImportPragma (ImportDecl decl)
 *      + RefinePragma (String filename)
 *    + Spec (MethodDecl md, Expr* targets, Hashtable preVarMap, 
 *            Condition* pre, Condition* post)
 *    + Condition (int label, Expr pred)
 *    + DefPred (Identifier predId, GenericVarDecl* args, Expr body)
 *    + DecreasesInfo (Expr f, VariableAccess fOld)
 * </pre>
 * 
 * <p> (Classes with a <code>-</code> next to them are defined in
 * <code>javafe.ast</code>; classes with a <code>+</code> are defined
 * in this package. </p>
 *
 * <p> (P.S. Ignore the stuff that appears below; the only purpose of
 * this class is to contain the above overview.) </p>
 *
 */

public abstract class AnOverview extends ASTNode
{ }

//// Spec expressions

public abstract class GCExpr extends Expr
{
  //# int sloc
  //# int eloc

  //@ public represents startLoc <- sloc;
  public /*@ pure @*/ int getStartLoc() { return sloc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == eloc;
    @*/
  public /*@ pure @*/ int getEndLoc() { return eloc; }
}

public class NaryExpr extends GCExpr
{
  //# int op
  //# Identifier methodName
  //# Expr* exprs  
  
  public ASTNode symbol;

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)

  //# ManualTag
  public final int getTag() { return op; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag = 
      (
       op == TagConstants.CLASSLITERALFUNC
       || op == TagConstants.DTTFSA
       || op == TagConstants.ELEMTYPE
       || op == TagConstants.FRESH
       || op == TagConstants.MAX
       || op == TagConstants.NOWARN_OP
       || op == TagConstants.TYPEOF
       || op == TagConstants.WACK_BIGINT_MATH
       || op == TagConstants.WACK_DURATION
       || op == TagConstants.WACK_JAVA_MATH
       || op == TagConstants.WACK_NOWARN
       || op == TagConstants.WACK_SAFE_MATH
       || op == TagConstants.WACK_WORKING_SPACE
       || op == TagConstants.WARN
       || op == TagConstants.WARN_OP
       || (TagConstants.FIRSTFUNCTIONTAG <= op 
	   && op <= TagConstants.LASTFUNCTIONTAG)
       );
    Assert.notFalse(goodtag);
  }

}

public class QuantifiedExpr extends GCExpr
{
  //# int quantifier
  //# GenericVarDecl* vars
  //# Expr rangeExpr
  //# Expr expr
  //# Expr* nopats NullOK
  //# Expr* pats NullOK

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)

  //# ManualTag
  public final int getTag() { return quantifier; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (quantifier == TagConstants.FORALL
       || quantifier == TagConstants.EXISTS);
    Assert.notFalse(goodtag);
  }
  
}

public class GeneralizedQuantifiedExpr extends GCExpr
{
  // Sum, Product, Max, Min
  //# int quantifier
  //# GenericVarDecl* vars
  //# Expr expr
  //# Expr rangeExpr
  //# Expr* nopats NullOK

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)

  //# ManualTag
  public final int getTag() { return quantifier; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (quantifier == TagConstants.MIN
       || quantifier == TagConstants.PRODUCT
       || quantifier == TagConstants.MAXQUANT
       || quantifier == TagConstants.SUM);
    Assert.notFalse(goodtag);
  }
}

public class NumericalQuantifiedExpr extends GCExpr
{
  // NumOf
  //# int quantifier
  //# GenericVarDecl* vars
  //# Expr rangeExpr
  //# Expr expr
  //# Expr* nopats NullOK

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)

  //# ManualTag
  public final int getTag() { return quantifier; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (quantifier == TagConstants.NUM_OF);
    Assert.notFalse(goodtag);
  }
}

public class SubstExpr extends GCExpr
{
  //# GenericVarDecl var
  //# Expr val
  //# Expr target

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)
}

/**
 * @note If <code>loc</code> is <code>Location.NULL</code>, then this
 * node is <em>not</em> due to a source-level "type" construct in an
 * annotation expression but rather was created during translations.
 */

public class TypeExpr extends GCExpr
{
  //# Type type

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)
}

public class LabelExpr extends GCExpr
{
  //# boolean positive
  //# Identifier label
  //# Expr expr

  //@ public represents startLoc <- sloc; //FIXME should not have to repeat this (because its in GCExpr)
}

public class WildRefExpr extends Expr
{
  //# Expr var
  //# ObjectDesignator od

  //@ public represents startLoc <- od != null ? od.getStartLoc() : var.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return od != null ? od.getStartLoc() : var.getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == (od != null ? od.getEndLoc() : var.getEndLoc());
    @*/
  public /*@ pure @*/ int getEndLoc() { return od != null ? od.getEndLoc() : var.getEndLoc(); }
}

public class GuardExpr extends Expr
{
  //# Expr expr
  //# int locPragmaDecl

  //@ public represents startLoc <- expr.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return expr.getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == expr.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
}

public class ResExpr extends Expr
{
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class SetCompExpr extends Expr
{
  //# Type type
  //# FormalParaDecl fp
  //# Expr expr

  //@ public represents startLoc <- fp.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return fp.getStartLoc(); }
}

public class LockSetExpr extends Expr
{
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class EverythingExpr extends Expr
{
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class NothingExpr extends Expr
{
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class NotSpecifiedExpr extends Expr
{
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class NotModifiedExpr extends Expr
{
  //# int loc
  //# Expr expr
 
  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class ArrayRangeRefExpr extends Expr
{
  //# int locOpenBracket
  //# Expr array
  //# Expr lowIndex	NullOK
  //# Expr highIndex	NullOK

  //@ public represents startLoc <- locOpenBracket;
  public /*@ pure @*/ int getStartLoc() { return locOpenBracket; }

  public void bogusMethod() { int i = bogusField; }
}

public class DefPredLetExpr extends Expr
{
    //# DefPred* preds
    //# Expr body

  //@ public represents startLoc <- body.getStartLoc();
    public /*@ pure @*/ int getStartLoc() { return body.getStartLoc(); }
}

public class DefPredApplExpr extends Expr
{
    //# Identifier predId
    //# Expr* args

  //@ public represents startLoc <- Location.NULL;
    public /*@ pure @*/ int getStartLoc() { return Location.NULL; }
}

//// Types

public class EscPrimitiveType extends PrimitiveType
{
  //# ManualTag
    /*@ public normal_behavior
      @   ensures  \result <==> JavafePrimitiveType.isValidTag(tag)
      @                         || tag == TagConstants.ANY
      @                         || tag == TagConstants.TYPECODE
      @                         || tag == TagConstants.LOCKSET
      @                         || tag == TagConstants.OBJECTSET
      @                         || tag == TagConstants.DOTDOT
      @                         || tag == TagConstants.BIGINTTYPE
      @                         || tag == TagConstants.REALTYPE;
      @   ensures_redundantly JavafePrimitiveType.isValidTag(tag) ==> \result;
      @*/
    public static /*@pure*/ boolean isValidTag(int tag) {
	return JavafePrimitiveType.isValidTag(tag)
	    || tag == TagConstants.ANY
            || tag == TagConstants.TYPECODE
            || tag == TagConstants.LOCKSET
            || tag == TagConstants.OBJECTSET
            || tag == TagConstants.DOTDOT
            || tag == TagConstants.BIGINTTYPE
            || tag == TagConstants.REALTYPE;
    }

    /*@ also
      @ public normal_behavior
      @   ensures  \result == EscPrimitiveType.isValidTag(tag);
      @*/
    public /*@pure*/ boolean isValidTag() {
	return EscPrimitiveType.isValidTag(tag);
    }

    //# NoMaker
    /*@ public normal_behavior
      @   requires EscPrimitiveType.isValidTag(tag);
      @*/
    public static /*@pure non_null*/ EscPrimitiveType make(int tag) {
	EscPrimitiveType result = new EscPrimitiveType(null, tag, Location.NULL);
	return result;
    }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag = EscPrimitiveType.isValidTag(tag);
    Assert.notFalse(goodtag); 
  }

    /*@ protected normal_behavior
      @   requires EscPrimitiveType.isValidTag(tag);
      @   ensures  this.tmodifiers == tmodifiers;
      @   ensures  this.tag == tag;
      @   ensures  this.loc == loc;
      @*/
    protected /*@pure*/ EscPrimitiveType(TypeModifierPragmaVec tmodifiers, int tag, int loc) {
	super(tmodifiers, tag, loc);
    }
 }

//// Guarded commands

public abstract class GuardedCmd extends ASTNode
{ }

public class SimpleCmd extends GuardedCmd
{
  // Denotes skip or raise
  //# int cmd
  //# int loc

  //# ManualTag
  public final int getTag() { return cmd; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (cmd == TagConstants.SKIPCMD || cmd == TagConstants.RAISECMD);
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class ExprCmd extends GuardedCmd
{
  // Denotes assert or assume
  //# int cmd
  //# Expr pred
  //# int loc

  //# ManualTag
  public final int getTag() { return cmd; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (cmd == TagConstants.ASSERTCMD || cmd == TagConstants.ASSUMECMD);
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public abstract class AssignCmd extends GuardedCmd
{
  // denotes a subtype-dependent assignment to v
  // rhs must be pure
  //# VariableAccess v
  //# Expr rhs

  //@ public represents startLoc <- v.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return v.getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == rhs.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return rhs.getEndLoc(); }
}

public class GetsCmd extends AssignCmd
{
  // denotes v := rhs
}

public class SubGetsCmd extends AssignCmd
{
  // denotes v[index] := rhs
  //# Expr index
}

public class SubSubGetsCmd extends AssignCmd
{
  // denotes v[index][index] := rhs.  I expect that v will be "elems".
  //# Expr index1
  //# Expr index2
}

public class RestoreFromCmd extends AssignCmd
{
  // denotes RESTORE v FROM rhs
  // which has the same effect as v := rhs
  // but does not count "v" as a target
}

public class VarInCmd extends GuardedCmd
{
  // denotes VAR v IN g END.
  //# GenericVarDecl* v
  //# GuardedCmd g

  //@ public represents startLoc <- g.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return g.getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == g.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return g.getEndLoc(); }
}

public class DynInstCmd extends GuardedCmd
{
  // denotes DynamicInstanceCommand s IN g END.
  //# String s NoCheck
  //# GuardedCmd g

  //@ public represents startLoc <- g.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return g.getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == g.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return g.getEndLoc(); }
}

public class SeqCmd extends GuardedCmd
{
  // denotes g1 ; g2 ; ... ; gn
  //# GuardedCmd* cmds

  //# PostCheckCall
  private void postCheck() {
    Assert.notFalse(1 < cmds.size());
  }

  //@ public represents startLoc <- cmds.elementAt(0).getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return cmds.elementAt(0).getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == cmds.elementAt(cmds.size()-1).getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return cmds.elementAt(cmds.size()-1).getEndLoc(); }
}

public class DecreasesInfo extends ASTNode {
  // the location of the 'decreases' pragma
  //# int locStart
  //# int locEnd
  // the translated expression
  //# Expr f
  // a local variable storing the previous value of expr "f"
  //# VariableAccess fOld

  //@ public represents startLoc <- locStart;
  public /*@ pure @*/ int getStartLoc() { return locStart; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == locEnd;
    @*/
  public /*@ pure @*/ int getEndLoc() { return locEnd; }
}

public class LoopCmd extends GuardedCmd
{
  //# int locStart
  //# int locEnd
  //# int locHotspot
  //# Hashtable oldmap NoCheck
  //# Condition* invariants
  //# DecreasesInfo* decreases
  //# LocalVarDecl* skolemConstants
  //# Expr* predicates
  //# GenericVarDecl* tempVars
  //# GuardedCmd guard
  //# GuardedCmd body

  public GuardedCmd desugared;
  
  //@ public represents startLoc <- locStart;
  public /*@ pure @*/ int getStartLoc() { return locStart; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == locEnd;
    @*/
  public /*@ pure @*/ int getEndLoc() { return locEnd; }
}
 

public class CmdCmdCmd extends GuardedCmd
{
  // denotes g1 ! g2  or  g1 [] g2
  //# int cmd
  //# GuardedCmd g1
  //# GuardedCmd g2

  //# ManualTag
  public final int getTag() { return cmd; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (cmd == TagConstants.TRYCMD || cmd == TagConstants.CHOOSECMD);
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- g1.getStartLoc(); 
  public /*@ pure @*/ int getStartLoc() { return g1.getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == g2.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return g2.getEndLoc(); }
}

public class Call extends GuardedCmd
{
  //# Expr* args
  //# int scall
  //# int ecall

  //# boolean inlined

  // This is a backedge, so it can't be a child:
  //@ invariant rd != null;
  public /*@non_null*/ RoutineDecl rd;

  public Spec spec;
  public GuardedCmd desugared;

  //@ public represents startLoc <- scall;
  public /*@ pure @*/ int getStartLoc() { return scall; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == ecall;
    @*/
  public /*@ pure @*/ int getEndLoc() { return ecall; }
}

//// Pragmas

public class ExprDeclPragma extends TypeDeclElemPragma
{
  //# int tag
  //# Expr expr
  //# int modifiers
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }
  public int getModifiers() { return modifiers; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.AXIOM || 
       tag == TagConstants.INVARIANT );
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == expr.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
  public boolean isRedundant() { return redundant; }
  public void setRedundant(boolean v) { redundant = v; }
}

public class IdExprDeclPragma extends TypeDeclElemPragma
{
  //# int tag
  //# Identifier id      NoCheck
  //# Expr expr
  //# int modifiers
  //# int loc
  //# int locId

  //# ManualTag
  public final int getTag() { return tag; }
  public int getModifiers() { return modifiers; }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == expr.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
}

public class NamedExprDeclPragma extends TypeDeclElemPragma 
{
  //# int tag
  //# Expr expr
  //# Expr target
  //# int modifiers
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      ( tag == TagConstants.REPRESENTS);
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == expr.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
  public int getModifiers() { return modifiers; }
  public boolean isRedundant() { return redundant; }
  public void setRedundant(boolean v) { redundant = v; }
}

public class ModelDeclPragma extends TypeDeclElemPragma
{
  //# FieldDecl decl
  //# int loc

  public void setParent(/*@non_null*/TypeDecl p) {
    super.setParent(p);
    if (decl != null)
	decl.setParent(p);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == decl.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return decl.getEndLoc(); }
  public void decorate(ModifierPragmaVec modifierPragmas) {
    if (decl.pmodifiers == null) {
	decl.pmodifiers = modifierPragmas;
    } else if (modifierPragmas != null) {
	// FIXME - should be prepen???
	decl.pmodifiers.append(modifierPragmas); 
    }
  }
  public ModifierPragmaVec getPModifiers() { return decl.pmodifiers; }
}

public class DependsPragma extends TypeDeclElemPragma
{
  //# int tag
  //# Expr target
  //# Expr* exprs
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == exprs.elementAt(exprs.size()-1).getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return exprs.elementAt(exprs.size()-1).getEndLoc(); }
}

public class ModelConstructorDeclPragma extends TypeDeclElemPragma
{
  //# ConstructorDecl decl
  //# int loc
  //# SimpleName id

  public void setParent(/*@non_null*/TypeDecl p) {
    super.setParent(p);
    if (decl != null) 
	decl.setParent(p);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == decl.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return decl.getEndLoc(); }
  public void decorate(ModifierPragmaVec modifierPragmas) {
    if (decl.pmodifiers == null) {
	decl.pmodifiers = modifierPragmas;
    } else if (modifierPragmas != null) {
	// FIXME - should be prepen???
	decl.pmodifiers.append(modifierPragmas); 
    }
  }
}

public class ModelTypePragma extends TypeDeclElemPragma
{
  //# TypeDecl decl
  //# int loc

  public void setParent(/*@non_null*/TypeDecl p) {
    super.setParent(p);
    if (decl != null) 
	decl.setParent(p);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == decl.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return decl.getEndLoc(); }
  public void decorate(ModifierPragmaVec modifierPragmas) {
    if (decl.pmodifiers == null) {
	decl.pmodifiers = modifierPragmas;
	// FIXME - should be prepen???
    } else if (modifierPragmas != null) {
	decl.pmodifiers.append(modifierPragmas); 
    }
  }
}

public class ModelMethodDeclPragma extends TypeDeclElemPragma
{
  //# MethodDecl decl
  //# int loc

  public void setParent(/*@non_null*/TypeDecl p) {
    super.setParent(p);
    if (decl != null) 
	decl.setParent(p);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == decl.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return decl.getEndLoc(); }
  public void decorate(ModifierPragmaVec modifierPragmas) {
    if (decl.pmodifiers == null) {
	decl.pmodifiers = modifierPragmas;
    } else if (modifierPragmas != null) {
	// FIXME - should be prepen???
	decl.pmodifiers.append(modifierPragmas); 
    }
  }
}

public class GhostDeclPragma extends TypeDeclElemPragma
{
  //# FieldDecl decl
  //# int loc

  public void setParent(/*@non_null*/TypeDecl p) {
    super.setParent(p);
    if (decl != null)
	decl.setParent(p);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == decl.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return decl.getEndLoc(); }
  public void decorate(ModifierPragmaVec modifierPragmas) {
    if (decl.pmodifiers == null) {
	decl.pmodifiers = modifierPragmas;
    } else if (modifierPragmas != null) {
	// FIXME - should be prepen???
	decl.pmodifiers.append(modifierPragmas); 
    }
  }
  public ModifierPragmaVec getPModifiers() { return decl.pmodifiers; }
}

public class StillDeferredDeclPragma extends TypeDeclElemPragma
{
  //# Identifier var
  //# int loc
  //# int locId

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class SimpleStmtPragma extends StmtPragma
{
  //# int tag
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag = (tag == TagConstants.UNREACHABLE);
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class IdentifierModifierPragma extends ModifierPragma
{
  //# int tag
  //# Identifier id
  //# int loc

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.IS_INITIALIZED);
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class ExprStmtPragma extends StmtPragma
{
  //# int tag
  //# Expr expr
  //# Expr label NullOK
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag = (tag == TagConstants.ASSERT || 
                       tag == TagConstants.ASSUME || 
                       tag == TagConstants.DECREASES ||
                       tag == TagConstants.DECREASING ||
                       tag == TagConstants.MAINTAINING || 
                       tag == TagConstants.LOOP_INVARIANT || 
                       tag == TagConstants.LOOP_PREDICATE);
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == expr.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
}

public class SetStmtPragma extends StmtPragma
{
  // set 'target' = 'value':

  //# Expr target
  //# int locOp
  //# Expr value
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == value.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return value.getEndLoc(); }
}

public class SkolemConstantPragma extends StmtPragma
{
  //# LocalVarDecl* decls
  //# int sloc
  //# int eloc

  //@ public represents startLoc <- sloc;
  public /*@ pure @*/ int getStartLoc() { return sloc; }
  /*@ also
    @ public normal_behavior
    @ ensures \result == eloc;
    @*/
  public /*@ pure @*/ int getEndLoc() { return eloc; }
}

public class ModelProgamModifierPragma extends ModifierPragma
{
  //# int tag
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class NestedModifierPragma extends ModifierPragma
{
	// This is a list of ModifierPragmaVec
  //# ArrayList list   NoCheck

  //# ManualTag
  public final int getTag() { return TagConstants.NESTEDMODIFIERPRAGMA; }

  // FIXME - need more robust setting of this
  //@ public represents startLoc <- ((ModifierPragmaVec)list.get(0)).elementAt(0).getStartLoc();
  public /*@ pure @*/ int getStartLoc() { 
      return ((ModifierPragmaVec)list.get(0)).elementAt(0).getStartLoc(); 
  }
}

public class ParsedSpecs extends ModifierPragma
{
  //# RoutineDecl decl         NoCheck
  //# ParsedRoutineSpecs specs NoCheck

  //# ManualTag
  public final int getTag() { return TagConstants.PARSEDSPECS; }

  //@ public represents startLoc <- decl.locId;
  public /*@ pure @*/ int getStartLoc() { return decl.locId; }
}

public class SimpleModifierPragma extends ModifierPragma
{
  //# int tag
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.CODE_BIGINT_MATH
       || tag == TagConstants.CODE_JAVA_MATH
       || tag == TagConstants.CODE_SAFE_MATH
       || tag == TagConstants.HELPER
       || tag == TagConstants.IMMUTABLE
       || tag == TagConstants.NULLABLE
       || tag == TagConstants.MONITORED
       || tag == TagConstants.NON_NULL
       || tag == TagConstants.NON_NULL_BY_DEFAULT
       || tag == TagConstants.NULLABLE_BY_DEFAULT
       || tag == TagConstants.OBS_PURE
       || tag == TagConstants.PEER
       || tag == TagConstants.READONLY
       || tag == TagConstants.REP
       || tag == TagConstants.SPEC_BIGINT_MATH
       || tag == TagConstants.SPEC_JAVA_MATH
       || tag == TagConstants.SPEC_PUBLIC
       || tag == TagConstants.SPEC_SAFE_MATH
       || tag == TagConstants.UNINITIALIZED
       || tag == TagConstants.WRITABLE_DEFERRED
       );
    Assert.notFalse(goodtag);
  }

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class ExprModifierPragma extends ModifierPragma
{
    // Extended to support JML

    //# int tag
    //# Expr expr
    //# int loc
    public int errorTag = 0;

    //# ManualTag
    public final int getTag() { return tag; }

    //# PostCheckCall
    private void postCheck() {
        boolean goodtag = (tag == TagConstants.ALSO_ENSURES
                           || tag == TagConstants.ALSO_REQUIRES
                           || tag == TagConstants.ENSURES 
                           || tag == TagConstants.MONITORED_BY 
                           || tag == TagConstants.READABLE_IF 
                           || tag == TagConstants.REQUIRES 
                           || tag == TagConstants.WACK_DURATION
                           || tag == TagConstants.WACK_WORKING_SPACE
                           || tag == TagConstants.WRITABLE_IF
                           );
        boolean isJMLExprModifier = isJMLExprModifier();
        Assert.notFalse(goodtag || isJMLExprModifier);
    }

    private boolean isJMLExprModifier() {
        return tag == TagConstants.ALSO 
            || tag == TagConstants.PRECONDITION
            || tag == TagConstants.POSTCONDITION;
    }

    //@ public represents startLoc <- loc;
    public /*@ pure @*/ int getStartLoc() { return loc; }
    /*@ also
      @ public normal_behavior
      @ ensures \result == expr.getEndLoc();
      @*/
    public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
}

public class ModifiesGroupPragma extends ModifierPragma
{
    //# int tag
    //# CondExprModifierPragma* items
    //# Expr precondition
    //# int clauseLoc

    //@ public represents startLoc <- clauseLoc;
    public /*@ pure @*/ int getStartLoc() { return clauseLoc; }

    static public ModifiesGroupPragma make(int tag, int loc) {
	ModifiesGroupPragma t = new ModifiesGroupPragma(
                                    tag,
                                    CondExprModifierPragmaVec.make(),
                                    null,
                                    loc);
	return t;
    }

    public void addElement(CondExprModifierPragma e) {
	items.addElement(e);
    }

    public ModifiesGroupPragma append(ModifiesGroupPragma m) {
	items.append(m.items);
	return this;
    }

    public ModifiesGroupPragma append(CondExprModifierPragmaVec ev) {
	items.append(ev);
	return this;
    }
}

public class CondExprModifierPragma extends ModifierPragma
{
    // Extended to support JML

    //# int tag
    //# Expr expr
    //# int loc
    //# Expr cond

    //# ManualTag
    public final int getTag() { return tag; }

    //# PostCheckCall
    private void postCheck() {
        boolean goodtag = (tag == TagConstants.ALSO_MODIFIES ||
                           tag == TagConstants.MODIFIES);
        boolean isJMLExprModifier = isJMLExprModifier();
        Assert.notFalse(goodtag || isJMLExprModifier);
    }

    private boolean isJMLExprModifier() {
        return (tag == TagConstants.ASSIGNABLE ||
                tag == TagConstants.ONLY_ASSIGNED ||
                tag == TagConstants.MODIFIABLE);
    }

    //@ public represents startLoc <- loc;
    public /*@ pure @*/ int getStartLoc() { return loc; }
    /*@ also
      @ public normal_behavior
      @ ensures \result == cond.getEndLoc();
      @*/
    public /*@ pure @*/ int getEndLoc() { return cond.getEndLoc(); }
}

public class MapsExprModifierPragma extends ModifierPragma
  implements javafe.ast.IdPragma
{
    // Extended to support JML

    //# int tag
    //# Identifier id
    //# Expr mapsexpr
    //# int loc
    //# Expr expr

    //# ManualTag
    public final int getTag() { return tag; }

    //# PostCheckCall
    private void postCheck() {
        boolean goodtag = (tag == TagConstants.MAPS);
        Assert.notFalse(goodtag);
    }

    public Identifier id() { return id; }
    //@ public represents startLoc <- loc;
    public /*@ pure @*/ int getStartLoc() { return loc; }
    /*@ also
      @ public normal_behavior
      @ ensures \result == expr.getEndLoc();
      @*/
    public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
}

public class ReachModifierPragma extends ModifierPragma
{
    //# Expr expr
    //# Identifier id
    //# Identifier* storerefs
    //# int loc

    //@ public represents startLoc <- loc;
    public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class VarDeclModifierPragma extends ModifierPragma
{
    //# int tag
    //# LocalVarDecl decl
    //# int loc
    //# int locId

    //# ManualTag
    public int getTag() { return tag; }

    //@ public represents startLoc <- loc;
    public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class VarExprModifierPragma extends ModifierPragma
{
    // Extended to support JML

    //# int tag
    //# GenericVarDecl arg
    //# Expr expr
    //# int loc

    //# ManualTag
    public final int getTag() { return tag; }

    //# PostCheckCall
    private void postCheck() {
        boolean goodtag =
            (tag == TagConstants.EXSURES 
             || tag == TagConstants.ALSO_EXSURES
             || tag == TagConstants.SIGNALS);
        Assert.notFalse(goodtag);
    }

    //@ public represents startLoc <- loc;
    public /*@ pure @*/ int getStartLoc() { return loc; }
    /*@ also
      @ public normal_behavior
      @ ensures \result == expr.getEndLoc();
      @*/
    public /*@ pure @*/ int getEndLoc() { return expr.getEndLoc(); }
}

public class NowarnPragma extends LexicalPragma
{
  //# Identifier* checks NoCheck
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class ImportPragma extends LexicalPragma
{
  //# ImportDecl decl
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class RefinePragma extends LexicalPragma
{
  //# String filename NoCheck
  //# int loc

  //@ public represents startLoc <- loc;
  public /*@ pure @*/ int getStartLoc() { return loc; }
}

public class Spec extends ASTNode
{
  //# DerivedMethodDecl dmd NoCheck
  //# Expr* targets
  //# Expr* specialTargets
  //# Hashtable preVarMap NoCheck
  //# Expr* preAssumptions
  //# Condition* pre
  //# Expr* postAssumptions
  //# Condition* post
  //# boolean modifiesEverything
  //# Set postconditionLocations  NoCheck

  //@ public represents startLoc <- dmd.original.getStartLoc();
  public /*@ pure @*/ int getStartLoc() { return dmd.original.getStartLoc(); }
  /*@ also
    @ public normal_behavior
    @ ensures \result == dmd.original.getEndLoc();
    @*/
  public /*@ pure @*/ int getEndLoc() { return dmd.original.getEndLoc(); }
}

public class Condition extends ASTNode
{
  //# int label
  //# Expr pred
  //# int locPragmaDecl

  //@ public represents startLoc <- locPragmaDecl;
  public /*@ pure @*/ int getStartLoc() { return locPragmaDecl; }

  public String prettyPrint() {
	return "Condition: label = " + TagConstants.toString(label) + "\n"
		+ "     loc = " + Location.toString(locPragmaDecl) + "\n"
		+ "     pred = " + pred;
  }
}

public class DefPred extends ASTNode
{
    //# Identifier predId
    //# GenericVarDecl* args
    //# Expr body

    //@ public represents startLoc <- body.getStartLoc();
    public /*@ pure @*/ int getStartLoc() { return body.getStartLoc(); }
}
