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
{ 

// Generated boilerplate constructors:

   protected AnOverview() {
   }
   

// Generated boilerplate methods:

   /** Return the number of children a node has. */
   public abstract int childCount();

   /** Return the first-but-ith child of a node. */
   public abstract /*@ nullable */ Object childAt(int i);

   /** Return the tag of a node. */
   public abstract int getTag();

   /** Return a string representation of <code>this</code>.
   Meant for debugging use only, not for presentation. */
   public abstract /*@non_null*/ String toString();

   /** Accept a visit from <code>v</code>.  This method simply
   calls the method of <code>v</code> corresponding to the
   allocated type of <code>this</code>, passing <code>this</code>
   as the argument.  See the design patterns book. */
   public abstract void accept(/*@non_null*/ javafe.ast.Visitor v);

public abstract /*@ non_null */ Object accept(/*@ non_null */ javafe.ast.VisitorArgResult v, Object o);

   public void check() {
   }

}
