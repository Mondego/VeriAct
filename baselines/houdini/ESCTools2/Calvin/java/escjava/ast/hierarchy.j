/* Copyright Hewlett-Packard, 2002 */

package escjava.ast;


import java.util.Hashtable;
import java.util.Vector;

import javafe.ast.*;
import escjava.ast.Visitor;      // Work around 1.0.2 compiler bug
import escjava.ast.TagConstants; // Work around 1.0.2 compiler bug
import escjava.ast.GeneratedTags;// Work around 1.0.2 compiler bug
import escjava.ast.AnOverview;   // Work around 1.0.2 compiler bug
import escjava.translate.Reduction;
import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.Set;

// Convention: unless otherwise noted, integer fields named "loc"g refer
// to the locaction of the first character of the syntactic unit

//# TagBase javafe.tc.TagConstants.LAST_TAG + 1
//# VisitorRoot javafe.ast.Visitor


//# EndHeader

/**

The files in this package extend the AST classes defined in
<code>javafe.ast</code>.  The following diagram shows how the new
classes related to the old classes in the class hierarchy:

  * <PRE>
  * - ASTNode
  *    + PerformsStmt 
  *      + PerformsAction (Expr* exprs, Expr pred, Identifier label) // Performs
  *      + PerformsChoice (PerformsStmt left, PerformsStmt right) // Performs
  *      + PerformsSequence (PerformsStmt *seq) // Performs
  *    + MapType (Type indexType, Type elemType)
  *    - VarInit ()
  *       - Expr ()
  *         + GCExpr
  *           + LabelExpr (Identifier label, Expr expr)
  *           + NaryExpr (int op, Expr* exprs)
  *           + QuantifiedExpr (GenericVarDecl* vars, Expr expr)
  *           + SubstExpr (GenericVarDecl var, Expr val, Expr target)
  *           + TypeExpr (Type type)
  *         + TidExpr ()      
  *         + MainExpr ()      
  *         + WitnessExpr ()      
  *         + LockSetExpr ()
  *         + ResExpr ()
  *         + WildRefExpr (Expr expr)
  *         + GuardExpr (Expr expr)
  *         + DefPredLetExpr (DefPred* preds, Expr body)
  *         + DefPredApplExpr (Identifier predId, Expr* args)
  *    + GuardedCmd
  *      + YieldCmd ()
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
  *      + LoopCmd (Condition invariants*, LocalVarDecl skolemConstants*, Expr predicates*, GenericVarDecl tempVars*, HashTable oldmap, GuardedCmd guard, GuardedCmd body)
  *      + CmdCmdCmd (int cmd, GuardedCmd g1, GuardedCmd g2)// Try, Choose
  *      + Call (RoutineDecl rd, Expr* args, TypeDecl scope)
  *    - TypeDeclElem ()
  *       - TypeDeclElemPragma ()
  *         + ExprDeclPragma (Expr expr) // Axiom, ObjectInvariant
  *	    + GhostDeclPragma (GhostFieldDecl decl)
  *         + StillDeferredDeclPragma (Identifier var)
  *    - Stmt ()
  *       - StmtPragma ()
  *         + SimpleStmtPragma () // Unreachable
  *         + ExprStmtPragma (Expr expr) // Assume, Assert, LoopInvariant, LoopPredicate
  *         + SetStmtPragma (Expr target, Expr value) 
  *         + SkolemConstantPragma (LocalVarDecl* decl)
  *    - ModifierPragma ()
  *         + SimpleModifierPragma () // Uninitialized, Monitored, Nonnull, WritableDeferred, Helper
  *         + ExprModifierPragma (Expr expr) // DefinedIf, Writable, Requires, Ensures, AlsoEnsures, MonitoredBy, Modifies, AlsoModifies
  *         + VarExprModifierPragma (GenericVarDecl arg, Expr expr) // Exsures, AlsoExsures
  *         + PerformsModifierPragma (PerformsStmt) // Exsures, AlsoExsures
  *    - LexicalPragma ()
  *      + NowarnPragma (Identifier* checks)
  *    + Spec (MethodDecl md, Expr* targets, Hashtable preVarMap, Condition* pre, Condition* post)
  *    + Condition(int label, Expr pred)
  *    + DefPred (Identifier predId, GenericVarDecl* args, Expr body)
  * </PRE>

(Classes with a <code>-</code> next to them are defined in
<code>javafe.ast</code>; classes with a <code>+</code> are defined in
this package.

(P.S. Ignore the stuff that appears below; the only purpose of this class
is to contain the above overview.)

@see javafe.ast.AnOverview

*/

public abstract class AnOverview extends ASTNode {
}

public class MapType extends javafe.ast.Type {
  //# Type indexType
  //# Type elemType
  //# int loc

  public int getStartLoc() { return loc; }
}

//// Spec expressions

public abstract class GCExpr extends Expr {
  //# int sloc
  //# int eloc

  public int getStartLoc() { return sloc; }
  public int getEndLoc() { return eloc; }
}

public class NaryExpr extends GCExpr {
  //# int op
  //# Expr* exprs
 
  //# ManualTag
  public final int getTag() { return op; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag = 
      (op == TagConstants.CLASSLITERALFUNC
       || op == TagConstants.DTTFSA
       || op == TagConstants.ELEMTYPE
       || op == TagConstants.FRESH
       || op == TagConstants.MAX
       || op == TagConstants.TYPEOF
       || (TagConstants.FIRSTFUNCTIONTAG <= op 
	   && op <= TagConstants.LASTFUNCTIONTAG));
    Assert.notFalse(goodtag);
  }

}

public class QuantifiedExpr extends GCExpr {
  //# int quantifier
  //# GenericVarDecl* vars
  //# Expr expr
  //# Expr* nopats NullOK

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

public class SubstExpr extends GCExpr {
  //# GenericVarDecl var
  //# Expr val
  //# Expr target

}

/** Note: if <code>loc</code> is <code>Location.NULL</code>, then this
  node is <em>not</em> due to a source-level "type" construct in an
  annotation expression but rather was created during translations. */

public class TypeExpr extends GCExpr {
  //# Type type

}

public class LabelExpr extends GCExpr {
  //# boolean positive
  //# Identifier label
  //# Expr expr


}

public class WildRefExpr extends Expr {
  //# Expr expr
  //# int locOpenBracket
  //# int locCloseBracket

  public int getStartLoc() { return expr.getStartLoc(); }
  public int getEndLoc() { return expr.getEndLoc(); }
}

public class GuardExpr extends Expr {
  //# Expr expr
  //# int locPragmaDecl

  public int getStartLoc() { return expr.getStartLoc(); }
  public int getEndLoc() { return expr.getEndLoc(); }
}

public class TidExpr extends Expr {
  //# int loc

  public int getStartLoc() { return loc; }
}

public class MainExpr extends Expr {
  //# int loc

  public int getStartLoc() { return loc; }
}

public class WitnessExpr extends Expr {
  //# int loc

  public int getStartLoc() { return loc; }
}

public class ResExpr extends Expr {
  //# int loc

  public int getStartLoc() { return loc; }
}

public class LockSetExpr extends Expr {
  //# int loc

  public int getStartLoc() { return loc; }
}

public class DefPredLetExpr extends Expr {
    //# DefPred* preds
    //# Expr body

    public int getStartLoc() { return body.getStartLoc(); }
}

public class DefPredApplExpr extends Expr {
    //# Identifier predId
    //# Expr* args

    public int getStartLoc() { return Location.NULL; }
}

//// Guarded commands

public abstract class GuardedCmd extends ASTNode { }

public class SimpleCmd extends GuardedCmd {
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

  public int getStartLoc() { return loc; }
}

public class ExprCmd extends GuardedCmd {
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

  public int getStartLoc() { return loc; }
}

public abstract class AssignCmd extends GuardedCmd {
  // denotes a subtype-dependent assignment to v
  // rhs must be pure
  //# VariableAccess v
  //# Expr rhs

  public int getStartLoc() { return v.getStartLoc(); }
  public int getEndLoc() { return rhs.getEndLoc(); }
}

public class GetsCmd extends AssignCmd {
  // denotes v := rhs
}

public class SubGetsCmd extends AssignCmd {
  // denotes v[index] := rhs
  //# Expr index
}

public class SubSubGetsCmd extends AssignCmd {
  // denotes v[index][index] := rhs.  I expect that v will be "elems".
  //# Expr index1
  //# Expr index2
}

public class RestoreFromCmd extends AssignCmd {
  // denotes RESTORE v FROM rhs
  // which has the same effect as v := rhs
  // but does not count "v" as a target
}

public class VarInCmd extends GuardedCmd {
  // denotes VAR v IN g END.
  //# GenericVarDecl* v
  //# GuardedCmd g

  public int getStartLoc() { return g.getStartLoc(); }
  public int getEndLoc() { return g.getEndLoc(); }
}

public class DynInstCmd extends GuardedCmd {
  // denotes DynamicInstanceCommand s IN g END.
  //# String s NoCheck
  //# GuardedCmd g

  public int getStartLoc() { return g.getStartLoc(); }
  public int getEndLoc() { return g.getEndLoc(); }
}

public class SeqCmd extends GuardedCmd {
  // denotes g1 ; g2 ; ... ; gn
  //# GuardedCmd* cmds

  //# PostCheckCall
  private void postCheck() {
    Assert.notFalse(1 < cmds.size());
  }

  public int getStartLoc() { return cmds.elementAt(0).getStartLoc(); }
  public int getEndLoc() { return cmds.elementAt(cmds.size()-1).getEndLoc(); }
}

public class LoopCmd extends GuardedCmd {
  //# int locStart
  //# int locEnd
  //# int locHotspot
  //# Hashtable oldmap NoCheck
  //# Condition* invariants
  //# LocalVarDecl* skolemConstants
  //# Expr* predicates
  //# GenericVarDecl* tempVars
  //# GuardedCmd guard
  //# GuardedCmd body

  public GuardedCmd desugared;
  
  public int getStartLoc() { return locStart; }
  public int getEndLoc() { return locEnd; }
}
 
public class YieldCmd extends GuardedCmd {
  //# int loc
  //# GuardedCmd code 
  //# Set readVars NoCheck
  //# Set writeVars NoCheck
  //# ConditionVec conds NoCheck
  //# ConditionVec rmConds NoCheck
  //# ConditionVec lmConds NoCheck
  //# Vector localVarDecls NoCheck

  public Spec spec;
  public GuardedCmd desugared;
  public int mark = Reduction.NONMOVER;
  public int inN;
  public int inE;
//  public FindContributors scope;

  public int getStartLoc() { return loc; }
  public void setCode(GuardedCmd gc) { code = gc; }
}

public class CmdCmdCmd extends GuardedCmd {
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

  public int getStartLoc() { return g1.getStartLoc(); }
  public int getEndLoc() { return g2.getEndLoc(); }
}

public class Call extends GuardedCmd {
  //# Expr* args
  //# int scall
  //# int ecall

    //# boolean inlined

  // This is a backedge, so it can't be a child:
  //@ invariant rd!=null
  public RoutineDecl rd;

  public Spec spec;
  public GuardedCmd desugared;

  public int getStartLoc() { return scall; }
  public int getEndLoc() { return ecall; }
}

//// Pragmas

public class ExprDeclPragma extends TypeDeclElemPragma {
  //# int tag
  //# Expr expr
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.AXIOM || tag == TagConstants.INVARIANT);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return expr.getEndLoc(); }
}

public class GhostDeclPragma extends TypeDeclElemPragma {
  //# FieldDecl decl
  //# int loc

  public void setParent(TypeDecl p) {
    super.setParent(p);
    if (decl!=null)
	decl.setParent(p);
  }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return decl.getEndLoc(); }
}

public class StillDeferredDeclPragma extends TypeDeclElemPragma {
  //# Identifier var
  //# int loc
  //# int locId

  public int getStartLoc() { return loc; }
}


public class SimpleStmtPragma extends StmtPragma {
  //# int tag
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag = (tag == TagConstants.UNREACHABLE);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
}

public class ExprStmtPragma extends StmtPragma {
  //# int tag
  //# Expr expr
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.ASSUME || tag == TagConstants.ASSERT
       || tag == TagConstants.LOOP_INVARIANT);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return expr.getEndLoc(); }
}

public class SetStmtPragma extends StmtPragma {
  // set 'target' = 'value':

  //# Expr target
  //# int locOp
  //# Expr value
  //# int loc

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return value.getEndLoc(); }
}

public class SkolemConstantPragma extends StmtPragma {
  //# LocalVarDecl* decls
  //# int sloc
  //# int eloc
  public int getStartLoc() { return sloc; }
  public int getEndLoc() { return eloc; }
}

public class SimpleModifierPragma extends ModifierPragma {
  //# int tag
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.UNINITIALIZED
       || tag == TagConstants.MONITORED
       || tag == TagConstants.NON_NULL
       || tag == TagConstants.SPEC_PUBLIC
       || tag == TagConstants.WRITABLE_DEFERRED
       || tag == TagConstants.HELPER 
       || tag == TagConstants.THREAD_LOCAL
       || tag == TagConstants.THREAD_LOCAL_RESULT);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
}

public class ExprModifierPragma extends ModifierPragma {
  //# int tag
  //# Expr expr
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.DEFINED_IF || tag == TagConstants.WRITABLE_IF
       || tag == TagConstants.REQUIRES || tag == TagConstants.ALSO_REQUIRES
       || tag == TagConstants.ENSURES || tag == TagConstants.ALSO_ENSURES
       || tag == TagConstants.MONITORED_BY || tag == TagConstants.MODIFIES
       || tag == TagConstants.ALSO_MODIFIES || tag == TagConstants.UNWRITABLE_BY_ENV_IF);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return expr.getEndLoc(); }
}

public class ParamExprModifierPragma extends ModifierPragma {
  //# int tag
  //# Identifier* ids
  //# Expr expr
  //# int loc

  public GenericVarDeclVec vds = GenericVarDeclVec.make();

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.ELEMS_UNWRITABLE_BY_ENV_IF);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return expr.getEndLoc(); }
}

public class VarExprModifierPragma extends ModifierPragma {
  //# int tag
  //# GenericVarDecl arg
  //# Expr expr
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.EXSURES || tag == TagConstants.ALSO_EXSURES);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return expr.getEndLoc(); }
}



public class PerformsModifierPragma extends ModifierPragma {
  //# int loc
  //# PerformsStmt stmt
  //# LabelRelation labelRelation NoCheck

  //# ManualTag
  public final int getTag() { return TagConstants.PERFORMS; }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return stmt.getEndLoc(); }
}


public abstract class PerformsStmt extends ASTNode {
  //# int loc
}


public class PerformsAction extends PerformsStmt {
  //# Expr* exprs
  //# Expr pred
  //# String label NoCheck

  //# ManualTag
  public final int getTag() { return TagConstants.ACTION; }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { return pred.getEndLoc(); }
}


public class PerformsChoice extends PerformsStmt {
  //# PerformsStmt left
  //# PerformsStmt right

  //# ManualTag
  public final int getTag() { return TagConstants.CHOICE; }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { 
    return right.getEndLoc();
  }
}

public class PerformsSequence extends PerformsStmt {
  //# PerformsStmt* seq

  //# ManualTag	
  public final int getTag() { return TagConstants.SEMICOLON; }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { 
    if (seq.size()==0)
      return super.getEndLoc();

    return seq.elementAt(seq.size()-1).getEndLoc();
  }
}



public class ExprsModifierPragma extends ModifierPragma {
  //# int tag
  //# Expr* expr
  //# int loc

  //# ManualTag
  public final int getTag() { return tag; }

  //# PostCheckCall
  private void postCheck() {
    boolean goodtag =
      (tag == TagConstants.MONITORED_BY || tag == TagConstants.MODIFIES
       || tag == TagConstants.ALSO_MODIFIES);
    Assert.notFalse(goodtag);
  }

  public int getStartLoc() { return loc; }
  public int getEndLoc() { 
    if (expr.size()==0)
      return super.getEndLoc();

    return expr.elementAt(expr.size()-1).getEndLoc();
  }
}


public class NowarnPragma extends LexicalPragma {
  //# Identifier* checks NoCheck
  //# int loc

  public int getStartLoc() { return loc; }
}

public class Spec extends ASTNode {
  //# DerivedMethodDecl dmd NoCheck
  //# Expr* targets
  //# Hashtable preVarMap NoCheck
  //# Condition* pre
  //# Condition* post

  public int getStartLoc() { return dmd.original.getStartLoc(); }
  public int getEndLoc() { return dmd.original.getEndLoc(); }
}

public class Condition extends ASTNode {
  //# int label
  //# Expr pred
  //# int locPragmaDecl

  public int getStartLoc() { return locPragmaDecl; }
}

public class DefPred extends ASTNode {
    //# Identifier predId
    //# GenericVarDecl* args
    //# Expr body

    public int getStartLoc() { return body.getStartLoc(); }
}

