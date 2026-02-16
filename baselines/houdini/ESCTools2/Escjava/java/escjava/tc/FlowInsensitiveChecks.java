// $Id$
/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.tc;

import java.util.Enumeration;

import javafe.ast.ASTDecoration;
import javafe.ast.ASTNode;
import javafe.ast.AmbiguousMethodInvocation;
import javafe.ast.AmbiguousVariableAccess;
import javafe.ast.ArrayRefExpr;
import javafe.ast.ArrayType;
import javafe.ast.BinaryExpr;
import javafe.ast.CastExpr;
import javafe.ast.ClassDecl;
import javafe.ast.ClassDeclStmt;
import javafe.ast.ConstructorDecl;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.ForStmt;
import javafe.ast.FormalParaDecl;
import javafe.ast.GenericVarDecl;
import javafe.ast.Identifier;
import javafe.ast.InitBlock;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.LocalVarDeclVec;
import javafe.ast.MethodDecl;
import javafe.ast.MethodInvocation;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.Name;
import javafe.ast.ObjectDesignator;
import javafe.ast.ParenExpr;
import javafe.ast.PrimitiveType;
import javafe.ast.RoutineDecl;
import javafe.ast.SimpleName;
import javafe.ast.Stmt;
import javafe.ast.StmtPragma;
import javafe.ast.ThisExpr;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElem;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.TypeName;
import javafe.ast.TypeObjectDesignator;
import javafe.ast.UnaryExpr;
import javafe.ast.VarDeclStmt;
import javafe.ast.VarInit;
import javafe.ast.VariableAccess;
import javafe.tc.Env;
import javafe.tc.EnvForLocals;
import javafe.tc.EnvForTypeSig;
import javafe.tc.LookupException;
import javafe.tc.TypeSig;
import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Location;
import javafe.util.Set;
import escjava.Main;
import escjava.ast.ArrayRangeRefExpr;
import escjava.ast.CondExprModifierPragma;
import escjava.ast.CondExprModifierPragmaVec;
import escjava.ast.DependsPragma;
import escjava.ast.ExprDeclPragma;
import escjava.ast.ExprModifierPragma;
import escjava.ast.ExprStmtPragma;
import escjava.ast.ExprStmtPragmaVec;
import escjava.ast.GeneralizedQuantifiedExpr;
import escjava.ast.GhostDeclPragma;
import escjava.ast.IdExprDeclPragma;
import escjava.ast.LabelExpr;
import escjava.ast.MapsExprModifierPragma;
import escjava.ast.ModelConstructorDeclPragma;
import escjava.ast.ModelDeclPragma;
import escjava.ast.ModelMethodDeclPragma;
import escjava.ast.Modifiers;
import escjava.ast.ModifiesGroupPragma;
import escjava.ast.NamedExprDeclPragma;
import escjava.ast.NaryExpr;
import escjava.ast.NestedModifierPragma;
import escjava.ast.NotModifiedExpr;
import escjava.ast.NumericalQuantifiedExpr;
import escjava.ast.ParsedSpecs;
import escjava.ast.QuantifiedExpr;
import escjava.ast.SetCompExpr;
import escjava.ast.SetStmtPragma;
import escjava.ast.SkolemConstantPragma;
import escjava.ast.StillDeferredDeclPragma;
import escjava.ast.TagConstants;
import escjava.ast.TypeExpr;
import escjava.ast.Utils;
import escjava.ast.VarDeclModifierPragma;
import escjava.ast.VarExprModifierPragma;
import escjava.ast.WildRefExpr;

public class FlowInsensitiveChecks extends javafe.tc.FlowInsensitiveChecks
{
  static {
    inst = new FlowInsensitiveChecks();
  }
  static public javafe.tc.FlowInsensitiveChecks inst() { return inst; }

  // Setup for ghost variables

  /**
   * Are we in the middle of processing an annotation? (Used by {@link GhostEnv}.)
   */
  public static boolean inAnnotation = false;
  public static boolean inModelBody = false;
  // FIXME - the above two variables are a hack!  Note that below we have to save and
  // restore their values so that the appropriate values are read out of these
  // global-static variables by GhostEnv.  It would be much better to create a sub
  // Env that understands what to do and pass that along for the checks. -- DRCok

  public escjava.AnnotationHandler annotationHandler = 
    new escjava.AnnotationHandler();
  /**
   * Indicates whether <code>LS</code> is allowed in this context.  The default is
   * <code>true</code>.  For contexts where <code>LS</code> is not allowed,
   * <code>isLocksetContext</code> should be set <code>false</code> only temporarily.
   */
  protected boolean isLocksetContext = true;

  /**
   * <code>\result</code> is allowed in this context when <code>isRESContext</code>
   * is <code>true</code> and <code>returnType != null</code>.  The default value
   * of <code>isRESContext</code> is <code>false</code>.  For contexts where
   * <code>isRESContext</code> should be <code>true</code>,
   * <code>isRESContext</code> should be set to <code>true</code> only temporarily.
   */
  protected boolean isRESContext = false;

  /**
   * Indicates whether <code>\old</code> and <code>\fresh</code> are allowed in
   * this context.  The default is <code>false</code>.  For contexts where these
   * functions are allowed, <code>isTwoStateContext</code> should be set
   * <code>true</code> only temporarily.
   */
  protected boolean isTwoStateContext = false;

  /**
   * Indicates whether checking is currently being done inside a <code>PRE</code>.
   * This is used by the code that disallows nested <code>PRE</code> expressions.
   * Note: alternatively, one could use <code>isTwoStateContext</code> to implement
   * this functionality, but by having a separate bit, a more precise error message
   * can be produced.
   */
  protected boolean isInsidePRE = false;

  /**
   * Indicates whether a quantification or labeled predicate is allowed in this
   * context.  This boolean is used only between one call of <code>checkExpr</code>
   * to a following recursive call.
   */
  protected boolean isPredicateContext = false;

  /**
   * Indicates whether private field accesses are allowed in the current context.
   * Private field accesses are allowed everywhere, except in postconditions of
   * overridable methods.
   */
  protected boolean isPrivateFieldAccessAllowed = true;

  protected int accessibilityLowerBound = ACC_LOW_BOUND_Private;
  // Note, "ACC_LOW_BOUND_Private" would be the same as "no lower bound"
  protected static final int ACC_LOW_BOUND_Private = 0;
  protected static final int ACC_LOW_BOUND_Package = 1;
  protected static final int ACC_LOW_BOUND_Protected = 2;
  protected static final int ACC_LOW_BOUND_Public = 3;

  /**
   * If <code>accessibilityLowerBound != ACC_LOW_BOUND_Private</code>, then
   * <code>accessibilityContext</code> is the field or routine whose modifier
   * pragma is being type checked.
   */
  /*@ invariant accessibilityContext == null ||
    @           accessibilityContext instanceof FieldDecl ||
    @           accessibilityContext instanceof RoutineDecl;
  */
  //@ readable accessibilityContext if accessibilityLowerBound != ACC_LOW_BOUND_Private;
  protected ASTNode accessibilityContext;

  /**
   * Acts as a parameter to <code>checkExpr</code>.  Its value is meaningful only
   * on entry to <code>checkExpr</code>.  It indicates whether the expression to be
   * checked is in a specification designator context.  In particular, this
   * parameter is used to disallow array index wild cards in non-spec designator
   * contexts.
   */
  protected boolean isSpecDesignatorContext = false;
  
  /**
   * Contains the loop invariant statement pragmas seen so far and not yet
   * processed.
   */
  protected ExprStmtPragmaVec loopInvariants = ExprStmtPragmaVec.make();

  /**
   * Contains the loop decreases statement pragmas seen so far and not yet
   * processed.
   */
  protected ExprStmtPragmaVec loopDecreases = ExprStmtPragmaVec.make();

  protected ExprStmtPragmaVec loopPredicates = ExprStmtPragmaVec.make();

  protected LocalVarDeclVec skolemConstants = LocalVarDeclVec.make();

  /**
   * Indicates whether we are are checking an invariant pragma.
   */
  protected boolean invariantContext = false;

  /**
   * Counts the number of accesses of free variables and fields used for checking
   * the appropriateness of invariants.
   */
  //@ readable countFreeVarsAccesses if invariantContext;
  protected int countFreeVarsAccesses = 0 ;

  /**
   * Override so that we use {@link GhostEnv} instead of {@link EnvForTypeSig}.
   */
  protected EnvForTypeSig makeEnvForTypeSig(TypeSig s,
                                            boolean staticContext) {
    return new GhostEnv(s.getEnclosingEnv(), s, staticContext);
  }
  
  public static ASTDecoration envDecoration = 
    new ASTDecoration("env");
  public static ASTDecoration staticenvDecoration = 
    new ASTDecoration("staticenv");

  public void checkTypeDeclaration(/*@ non_null */ TypeSig s) {
    super.checkTypeDeclaration(s);
    TypeDecl td = s.getTypeDecl();
    envDecoration.set(td,rootIEnv);
    staticenvDecoration.set(td,rootSEnv);
  }

  // Extensions to type declaration member checkers.

  protected void checkTypeDeclElem(TypeDeclElem e) {
    boolean savedInAnnotation = inAnnotation;
    boolean savedInModelBody = inModelBody;
    // FIXME - should this use Utils.isModel ???
    if (e instanceof ConstructorDecl &&
        null != Utils.findModifierPragma(((ConstructorDecl)e).pmodifiers,TagConstants.MODEL)) {
      inAnnotation = true;
      inModelBody = true;
    }
    if (e instanceof MethodDecl &&
        null != Utils.findModifierPragma(((MethodDecl)e).pmodifiers,TagConstants.MODEL)) {
      inAnnotation = true;
      inModelBody = true;
    }
    try {
      super.checkTypeDeclElem(e);
      if (e instanceof RoutineDecl) {
        // Desugaring presumes that typechecking has already
        // been performed
        RoutineDecl m = (RoutineDecl)e;
/*
        if ((m instanceof ConstructorDecl) && m.implicit) {
          // The desugaring of m can require the desugared
          // specs of a parent constructor, so we have to be
          // sure that the parent constructor is typechecked.
          TypeSig s = TypeSig.getSig(m.parent).superClass();
          if (s != null) checkTypeDeclElem(s.getTypeDecl());
        }
*/
/*
        if (m.originalRaises != null && 
            m.originalRaises.size() != m.raises.size()) {

            for (int i=m.originalRaises.size()+1; i < m.raises.size(); ++i) {

              TypeSig t = TypeSig.getSig(m.raises.elementAt(i));
System.out.println("FOUND " + t);
              if (!t.isSubtypeOf(Types.javaLangRuntimeException())) {
                 ErrorSet.error(m.raises.elementAt(i).getStartLoc(),
                   "The type " + t + " is not a subtype of " +
                   "RuntimeException and is not declared by the API, and " +
                   "consequently may not be declared in a specification.");
                 System.out.println("BAD");
              }
            }
        }
*/
        annotationHandler.desugar(m); 
      }
    } finally {
      inAnnotation = savedInAnnotation;
      inModelBody = savedInModelBody;
    }

    // Do a separate set of checks - purity checking
    // FIXME - perhaps these should be moved into this routine
    annotationHandler.process(e);
    
    if (e.getTag() == TagConstants.INITBLOCK) {
      InitBlock ib = (InitBlock)e;
      if (ib.pmodifiers != null) {
        checkModifierPragmaVec(ib.pmodifiers,ib,
                               Modifiers.isStatic(ib.modifiers) ? rootSEnv : rootIEnv);
        /*
          for (int i = 0; i < ib.pmodifiers.size(); i++) {
          ModifierPragma mp = (ModifierPragma)ib.pmodifiers.elementAt(i);
          ErrorSet.error(mp.getStartLoc(),
          TagConstants.toString(mp.getTag()) +
          " pragma cannot be applied to initializer block");
          }
        */
      }
    }
    
  }

  // Extensions to Expr and Stmt checkers.

  protected Env checkStmt(Env env, Stmt s) {
    boolean savedTwoStateContext = isTwoStateContext;
    isTwoStateContext = true;
    try {
      int tag = s.getTag();

      // Check for any loop invariants, loop bounds, loop predicates, or skolem
      // constants in the wrong place
      if (loopInvariants.size() != 0 || 
          loopDecreases.size() != 0 || 
          loopPredicates.size() != 0 || 
          skolemConstants.size() != 0) {
        switch (tag) {
        case TagConstants.DOSTMT: 
        case TagConstants.WHILESTMT:
        case TagConstants.FORSTMT:
        case TagConstants.LABELSTMT:
        case TagConstants.LOOP_INVARIANT:
        case TagConstants.MAINTAINING:
        case TagConstants.DECREASES:
        case TagConstants.DECREASING:
        case TagConstants.LOOP_PREDICATE:
        case TagConstants.SKOLEMCONSTANTPRAGMA:
          break;
        default:
          checkLoopInvariants(env, false);
          checkLoopDecreases(env, false);
          checkLoopPredicates(env, false);
          checkSkolemConstants(env, false);
          break;
        }
      }

      switch(tag) {
      case TagConstants.DOSTMT: 
      case TagConstants.WHILESTMT: {
        checkLoopInvariants(env, true);
        checkLoopDecreases(env, true);
        Env newEnv = checkSkolemConstants(env, true);
        checkLoopPredicates(newEnv, true);
        super.checkStmt(env, s);
        break;
      }
      case TagConstants.FORSTMT: {
        ForStmt f = (ForStmt)s;

        ExprStmtPragmaVec invs = loopInvariants;
        ExprStmtPragmaVec decrs = loopDecreases;
        ExprStmtPragmaVec preds = loopPredicates;
        LocalVarDeclVec skolemConsts = skolemConstants;

        loopInvariants = ExprStmtPragmaVec.make();
        loopDecreases = ExprStmtPragmaVec.make();
        loopPredicates = ExprStmtPragmaVec.make();
        skolemConstants = LocalVarDeclVec.make();

        Env se = checkStmtVec(env, f.forInit);

        Assert.notFalse(loopInvariants.size() == 0);
        Assert.notFalse(loopDecreases.size() == 0);
        Assert.notFalse(loopPredicates.size() == 0);
        Assert.notFalse(skolemConstants.size() == 0);

        loopInvariants = invs;
        loopDecreases = decrs;
        loopPredicates = preds;
        skolemConstants = skolemConsts;

        checkLoopInvariants(se, true);
        checkLoopDecreases(se, true);
        Env newEnv = checkSkolemConstants(se, true);
        checkLoopPredicates(newEnv, true);
        checkForLoopAfterInit(se, f);
        break;
      }
      case TagConstants.BLOCKSTMT: {
        env = super.checkStmt(env, s);
        // Check for any loop_invariant statement pragmas at the end of the
        // body.
        checkLoopInvariants(env, false);
        checkLoopDecreases(env, false);
        checkLoopPredicates(env, false);
        checkSkolemConstants(env, false);
        break;
      }
      case TagConstants.VARDECLSTMT: {
        VarDeclStmt vs = (VarDeclStmt)s;
        LocalVarDecl x = vs.decl;
        if (Utils.findModifierPragma(x.pmodifiers,
                                     TagConstants.GHOST) != null) {
          boolean savedInAnnotation = inAnnotation;
          inAnnotation = true;
          try {
            env.resolveType(sig, x.type);
            checkTypeModifiers(env, x.type);
            javafe.tc.PrepTypeDeclaration.getInst().
              checkModifiers(x.modifiers, Modifiers.ACC_FINAL,
                             x.locId, "local ghost variable");
            checkModifierPragmaVec(x.pmodifiers, x, env);

            Env newEnv = new EnvForGhostLocals(env,x);
            if (x.init != null)
              x.init = checkInit(newEnv, x.init, x.type);
            env = newEnv;
          } finally {
            inAnnotation = savedInAnnotation;
          }
          break;
        }

        env = super.checkStmt(env, s);
        break;

      }
      case TagConstants.CLASSDECLSTMT: {
        ClassDeclStmt cds = (ClassDeclStmt)s;
        ClassDecl cd = cds.decl;
        (new escjava.AnnotationHandler.NestedPragmaParser()).parseAllRoutineSpecs(cd);
        env = super.checkStmt(env, s);
        annotationHandler.desugar(cd);
        break;
      }
      default:
        env = super.checkStmt(env, s);
        break;
      }

    } finally {
      isTwoStateContext = savedTwoStateContext;
    }
    return env;
  }

  protected void checkLoopInvariants(Env env, boolean allowed) {
    for (int i = 0; i < loopInvariants.size(); i++) {
      ExprStmtPragma s = loopInvariants.elementAt(i);
      Assert.notFalse(s.getTag() == TagConstants.LOOP_INVARIANT
                      || s.getTag() == TagConstants.MAINTAINING);
      if (allowed) {
        //Assert.notFalse(!isTwoStateContext);
        Assert.notFalse(!inAnnotation || inModelBody);
        boolean savedInAnnotation = inAnnotation;
        inAnnotation = true;
        //isTwoStateContext = true;
        try {
          s.expr = checkPredicate(env, s.expr);
        } finally {
          inAnnotation = savedInAnnotation;
          //isTwoStateContext = false;
        }
      } else {
        errorExpectingLoop(s.getStartLoc(), TagConstants.LOOP_INVARIANT);
      }
    }
    loopInvariants.removeAllElements();
  }

  protected void checkLoopDecreases(Env env, boolean allowed) {
    for (int i = 0; i < loopDecreases.size(); i++) {
      ExprStmtPragma s = loopDecreases.elementAt(i);
      Assert.notFalse(s.getTag() == TagConstants.DECREASES
                      || s.getTag() == TagConstants.DECREASING);
      if (allowed) {
        //Assert.notFalse(!isTwoStateContext);
        Assert.notFalse(!inAnnotation || inModelBody);
        boolean savedInAnnotation = inAnnotation;
        inAnnotation = true;
        try {
          s.expr = checkExpr(env, s.expr, Types.bigintType);
        } finally {
          inAnnotation = savedInAnnotation;
        }
      } else {
        errorExpectingLoop(s.getStartLoc(), TagConstants.DECREASES);
      }
    }
    loopDecreases.removeAllElements();
  }

  protected void checkLoopPredicates(Env env, boolean allowed) {
    for (int i = 0; i < loopPredicates.size(); i++) {
      ExprStmtPragma s = loopPredicates.elementAt(i);
      Assert.notFalse(s.getTag() == TagConstants.LOOP_PREDICATE);
      if (allowed) {
        //Assert.notFalse(!isTwoStateContext);
        Assert.notFalse(!inAnnotation || inModelBody);
        boolean savedInAnnotation = inAnnotation;
        inAnnotation = true;
        //isTwoStateContext = true;
        try {
          s.expr = checkPredicate(env, s.expr);
        } finally {
          inAnnotation = savedInAnnotation;
          //isTwoStateContext = false;
        }
      } else {
        errorExpectingLoop(s.getStartLoc(), TagConstants.LOOP_PREDICATE);
      }
    }
    loopPredicates.removeAllElements();
  }

  protected Env checkSkolemConstants(Env env, boolean allowed) {
    for (int i = 0; i < skolemConstants.size(); i++) {
      LocalVarDecl s = skolemConstants.elementAt(i);
      if (allowed) {
        //Assert.notFalse(!isTwoStateContext);
        Assert.notFalse(!inAnnotation || inModelBody);
        boolean savedInAnnotation = inAnnotation;
        inAnnotation = true;
        //isTwoStateContext = true;
        try {
          env.resolveType(sig, s.type);
          env = new EnvForLocals(env, s);
        } finally {
          inAnnotation = savedInAnnotation;
          //isTwoStateContext = false;	
        }
      } else {
        errorExpectingLoop(s.getStartLoc(), TagConstants.SKOLEM_CONSTANT);
      }
    }
    skolemConstants.removeAllElements();
    return env;
  }

  private void errorExpectingLoop(int loc, int tag) {
    ErrorSet.error(loc, "'" + TagConstants.toString(tag) +
                   "' pragmas can occur " +
                   "only immediately prior to a loop statement.  Loop " +
                   "statement expected (and not found) here.");
  }

  protected Expr checkPredicate(Env env, Expr e) {
    boolean savedPredicateContext = isPredicateContext;
    isPredicateContext = true;
    Expr ee = checkExpr(env, e, Types.booleanType);
    isPredicateContext = savedPredicateContext;
    return ee;
  }

  protected Expr checkExpr(/*@non_null*/ Env env, /*@non_null*/ Expr e) {
    // Anticipate that the next context is probably not one suitable for
    // quantifications and labels.  "isPredicateContext" must revert to its old
    // value before return.
    boolean isCurrentlyPredicateContext = isPredicateContext;
    isPredicateContext = false;

    try {

      if (getTypeOrNull(e) != null )
        // already done
        return e;

      // No recursive call to "checkExpr" is a specification designator context, so
      // set "isSpecDesignatorContext" to "false", keeping the initial value in
      // local variable "isCurrentlySpecDesignatorContext".
      boolean isCurrentlySpecDesignatorContext = isSpecDesignatorContext;
      isSpecDesignatorContext = false;
    
      int tag = e.getTag();
      switch(tag) {

        // Handle accesses to ghost fields as well...
      case TagConstants.FIELDACCESS:
        {
          if (!inAnnotation)
            return super.checkExpr(env, e);
	
          // a field access is considerded a free variable access in an
          // invariant.
          if (invariantContext) countFreeVarsAccesses++;

          // set default result type to errorType, in case there is an error.
          setType( e, Types.errorType );
          FieldAccess fa = (FieldAccess)e;
          Type t = checkObjectDesignator(env, fa.od);
          if (t==null) {
	      //alx: dw field has not target, take universe of declaration 
	      //        (when should this happen?)
	      if (useUniverses)
		  copyUniverses(fa,fa.decl);
	      //alx-end
	      return fa;
	  }
          if (t instanceof TypeName)
            t = TypeSig.getSig( (TypeName) t );

          if (Types.isErrorType(t)) {
            setType( fa, Types.errorType );
            return fa;
          }

          try {
            fa.decl = escjava.tc.Types.lookupField( t, fa.id, sig );
	    //alx: dw
            if (useUniverses)
    			determineUniverseForFieldAccess(fa);
            //alx-end
          } catch( LookupException ex) {
            if (!Types.isErrorType(t))
              reportLookupException(ex, "field", 
                                    Types.printName(t), fa.locId);
            setType( fa, Types.errorType );
            return fa;
          }
          setType( fa, fa.decl.type );

          if (fa.od instanceof TypeObjectDesignator &&
              !GhostEnv.isStatic(fa.decl)) {
            // Is fa.decl an instance field of the same class as fa is
            // part of?
            boolean thisField = false;
            if (fa.decl.parent != null)
              thisField = (env.getEnclosingClass()
                           .isSubtypeOf(TypeSig.getSig(fa.decl.parent)));
	    
            if (thisField ||
                ((TypeObjectDesignator)fa.od).type instanceof TypeName) {

              ErrorSet.error(fa.locId,
                             "An instance field may be accessed only via "
                             + "an object and/or from a non-static"
                             + " context or an inner class enclosed"
                             + " by a type possessing that field.");

            } else
              ErrorSet.error(fa.locId,
                             "The instance fields of type "
                             + ((TypeObjectDesignator)fa.od).type
                             + " may not be accessed from type "
                             + sig );
          }

          /* FIXME -- need to clean up testing of access modifiers and to make them
             consistent with JML 
             if (!isPrivateFieldAccessAllowed &&
             Modifiers.isPrivate(fa.decl.modifiers) &&
             Utils.findModifierPragma(fa.decl,
             TagConstants.SPEC_PUBLIC) == null &&
             Utils.findModifierPragma(fa.decl,
             TagConstants.SPEC_PROTECTED) == null) {
             ErrorSet.error(fa.locId, 
             "A private field can be used in "+
             "postconditions of overridable methods only if "+
             "it is declared spec_public or spec_protected");
             }
          */

          // The following code checks that "fa" is at least as
          // spec-accessible as "accessibilityContext" is Java-accessible.
          // This is vacuously true if "accessibilityLowerBound" is
          // private.
          if (accessibilityLowerBound != ACC_LOW_BOUND_Private) {
            boolean isAccessibleEnough;
            if (Modifiers.isPublic(fa.decl.modifiers) ||
                Utils.findModifierPragma(fa.decl,
                                         TagConstants.SPEC_PUBLIC) != null) {
              // public and spec_public fields are always accessible
              isAccessibleEnough = true;
            } else if (Utils.findModifierPragma(fa.decl,
                                                TagConstants.SPEC_PROTECTED) != null) {

              // Copied from the protected case down below.
              isAccessibleEnough = false;
              if (accessibilityLowerBound == ACC_LOW_BOUND_Package) {
                isAccessibleEnough = true;
              } else if (accessibilityLowerBound == ACC_LOW_BOUND_Protected) {
                TypeDecl tdContext;
                if (accessibilityContext instanceof FieldDecl) {
                  tdContext = ((FieldDecl)accessibilityContext).parent;
                } else {
                  tdContext = ((RoutineDecl)accessibilityContext).parent;
                }
                TypeSig tsContext = TypeSig.getSig(tdContext);
                if (tsContext.isSubtypeOf(TypeSig.getSig(fa.decl.parent))) {
                  isAccessibleEnough = true;
                }
              }
	    
            } else if (Modifiers.isPrivate(fa.decl.modifiers)) {
              // Note: if "fa" type-checked so far, then "fa.decl" and
              // "accessibilityContext" are declared in the same class.
              // It would be nice to assert this fact at run-time, but
              // control may reach this point even if "fa" does not
              // type-check above.

              // "fa.decl" can be private only if
              // "accessibilityContext" is private, which was checked
              // above
              isAccessibleEnough = false;
	    
            } else if (Modifiers.isPackage(fa.decl.modifiers)) {
              // Note: if "fa" type-checked so far, then "fa.decl" and
              // "accessibilityContext" are declared in the same
              // package.  It would be nice to assert this fact at
              // run-time, but control may reach this point even if
              // "fa" does not type-check above.

              // "fa.decl" can be package (default) accessible only if
              // "accessibilityContext" is private (which was checked
              // above) or package
              isAccessibleEnough =
                (accessibilityLowerBound == ACC_LOW_BOUND_Package);
	    
            } else {
              Assert.notFalse(Modifiers.isProtected(fa.decl.modifiers));
              // Note: if "fa" type-checked so far, then either
              // "fa.decl" and "accessibilityContext" are declared in
              // the same package or the class declaring
              // "accessibilityContext" is a subtype of the class
              // declaring "fa.decl".  It would be nice to assert this
              // fact at run-time, but control may reach this point
              // even if "fa" does not type-check above.

              // "fa.decl" can be protected only if
              // "accessibilityContext" is private (which was checked
              // above) or package, or if "accessibilityContext" is
              // protected and "fa.decl" is declared in a superclass of
              // the class that declares "accessibilityContext".
              isAccessibleEnough = false;
              if (accessibilityLowerBound == ACC_LOW_BOUND_Package) {
                isAccessibleEnough = true;
              } else if (accessibilityLowerBound == ACC_LOW_BOUND_Protected) {
                TypeDecl tdContext;
                if (accessibilityContext instanceof FieldDecl) {
                  tdContext = ((FieldDecl)accessibilityContext).parent;
                } else {
                  tdContext = ((RoutineDecl)accessibilityContext).parent;
                }
                TypeSig tsContext = TypeSig.getSig(tdContext);
                if (tsContext.isSubtypeOf(TypeSig.getSig(fa.decl.parent))) {
                  isAccessibleEnough = true;
                }
              }
            }
            /* Need to fix the accessibility checking to conform with JML - FIXME
               if (!isAccessibleEnough) {
               ErrorSet.error(fa.locId, "Fields mentioned in this modifier "+
               "pragma must be at least as accessible "+
               "as the field/method being modified (perhaps "+
               "try spec_public)");
               }
            */
          }
  
          return fa;
        }

      case TagConstants.METHODINVOCATION:
        {
          countFreeVarsAccesses++;
	  MethodInvocation mi = (MethodInvocation)e;
          Type t = checkObjectDesignator(env, mi.od);
          if (mi.od instanceof ExprObjectDesignator && Types.objectsetType == t) {
// FIXME - all reach expressions are true for now
            Expr ee = LiteralExpr.make(TagConstants.BOOLEANLIT,Boolean.TRUE,
                                          Location.NULL);
            setType(ee, Types.booleanType);
	    //alx: dw could be hardcoded
            if (useUniverses)
            	determineUniverseForMethodInvocation(mi);
	    //alx-end

            return ee;
          } else {
	    Expr ee = super.checkExpr(env,e);
	    return ee;
          }
        }
      
      case TagConstants.IMPLIES:
      case TagConstants.EXPLIES:
      case TagConstants.IFF:
      case TagConstants.NIFF:
        {
          BinaryExpr be = (BinaryExpr)e;
          // each argument is allowed to contain quantifiers and labels if
          // this expression is
          isPredicateContext = isCurrentlyPredicateContext;
          be.left = checkExpr(env, be.left, Types.booleanType);
          isPredicateContext = isCurrentlyPredicateContext;
          be.right = checkExpr(env, be.right, Types.booleanType);

          // check illegal associativity of ==> and <==
          if ((tag == TagConstants.IMPLIES &&
               (be.left.getTag() == TagConstants.EXPLIES ||
                be.right.getTag() == TagConstants.EXPLIES)) ||
              (tag == TagConstants.EXPLIES &&
               (be.left.getTag() == TagConstants.IMPLIES ||
                be.right.getTag() == TagConstants.IMPLIES))) {
            // unfortunately, we don't have the location of either of the
            // operators at fault
            ErrorSet.error(be.getStartLoc(),
                           "Ambiguous association of ==> and <==.  " +
                           "Use parentheses to disambiguate.");
          }

          setType(e, Types.booleanType);
          return e;
        }

      case TagConstants.DOTDOT:
        {
          BinaryExpr be = (BinaryExpr)e;
          // each argument is allowed to contain quantifiers and labels if
          // this expression is
          isPredicateContext = false;
          be.left = checkExpr(env, be.left, Types.intType);
          be.right = checkExpr(env, be.right, Types.intType);

          // FIXME - this really needs to be a range type
          setType(e, Types.intType);
          return e;
        }

      case TagConstants.SUBTYPE:
        {
          BinaryExpr be = (BinaryExpr)e;
          Type expected = Types.booleanType;
          if (tag == TagConstants.SUBTYPE) 
            expected = Types.typecodeType;
          be.left = checkExpr(env, be.left, expected);
          be.right = checkExpr(env, be.right, expected);
          setType(e, Types.booleanType);
          return e;
        }

      case TagConstants.REACH: {
        // FIXME - just enough to get by for now
        NaryExpr ne = (NaryExpr)e;
        Expr nu = 
          checkExpr(env, ne.exprs.elementAt(0));
        ne.exprs.setElementAt(nu, 0);			
        if (ne.exprs.size() != 1) {
          ErrorSet.error(ne.sloc,
               "A \\reach expression expects just one argument");
          setType(e, Types.errorType);
        } else if (!Types.isReferenceOrNullType(getType(nu))) {
          ErrorSet.error(nu.getStartLoc(),
               "A \\reach expression expects an Object argument");
        } else {
          setType(e, Types.objectsetType);
        }
        return e;
      }

      case TagConstants.FRESH:
        {
          NaryExpr ne = (NaryExpr)e;
          if (ne.exprs.size() == 0) {
            ErrorSet.error(ne.sloc, 
		       "The function fresh must have at least one argument");
          } else if (!isTwoStateContext) {
            ErrorSet.error(ne.sloc, 
		       "The function \\fresh cannot be used in this context");
          } else if (isInsidePRE) {
            ErrorSet.error(ne.sloc, "The function \\fresh cannot be used "+
                           "inside a \\old expression");
          } else {
            for (int i = 0; i<ne.exprs.size(); ++i) {
              Expr nu = 
                checkExpr(env, ne.exprs.elementAt(i), Types.javaLangObject());
              ne.exprs.setElementAt(nu, i);			
            }
          }
          setType(e, Types.booleanType);
          return e;
        }

      case TagConstants.ELEMSNONNULL:
        {
          NaryExpr ne = (NaryExpr)e;
          if (ne.exprs.size() != 1)
            ErrorSet.error(ne.sloc, 
                           "The function \\nonnullelements takes one argument");
          else {
            Expr nu = checkExpr(env, ne.exprs.elementAt(0),
                                ArrayType.make(Types.javaLangObject(),
                                               Location.NULL));
            ne.exprs.setElementAt(nu, 0);
          }
          setType(e, Types.booleanType);
          return e;
        }

      case TagConstants.DTTFSA:
        {
          NaryExpr ne = (NaryExpr)e;
          Type resultType = Types.voidType;
          if (ne.exprs.size() < 2) {
            Assert.notFalse(1 <= ne.exprs.size());
            ErrorSet.error(ne.sloc, 
                           "The function \\dttfsa requires at least two arguments");
          } else {
            // type check each of the arguments
            for (int i = 0; i < ne.exprs.size(); i++) {
              Expr nu = checkExpr(env, ne.exprs.elementAt(i));
              ne.exprs.setElementAt(nu, i);
            }
            // the first argument should be a TypeExpr; retrieve the type
            // it denotes
            resultType = ((TypeExpr)ne.exprs.elementAt(0)).type;
            // the second argument should be a String literal
            Expr arg1 = ne.exprs.elementAt(1);
            if (arg1.getTag() != TagConstants.STRINGLIT) {
              ErrorSet.error(arg1.getStartLoc(),
                             "The second argument to \\dttfsa must be a String literal");
            } else {
              LiteralExpr lit = (LiteralExpr)arg1;
              String op = (String)lit.value;
              if (op.equals("identity") && ne.exprs.size() != 3) {
                ErrorSet.error(ne.sloc, 
                               "The function \\dttfsa requires exactly 3 arguments when the second argument is \"identity\"");
              }
            }
          }
          setType(e, resultType);
          return e;
        }

      case TagConstants.WACK_NOWARN:
      case TagConstants.NOWARN_OP:
      case TagConstants.WARN:
      case TagConstants.WARN_OP:
        {
          NaryExpr ne = (NaryExpr)e;
          Expr nu;
          if( ne.exprs.size() != 1 ) {
            ErrorSet.error(ne.sloc, 
                           "The function " + TagConstants.toString(tag) + 
                           " takes one argument");
            setType( e, Types.errorType );
          } else {
            // Get rid of this - FIXME - these are an obsolete definition
            e = checkExpr(env, ne.exprs.elementAt(0));
            //ne.exprs.setElementAt( nu, 0 );			
            //setType( e, getType(nu) );
          }
          return e;
        }

      case TagConstants.ELEMTYPE:
        {
          NaryExpr ne = (NaryExpr)e;
          if( ne.exprs.size() != 1 ) 
            ErrorSet.error(ne.sloc, 
                           "The function \\elemtype takes one argument");
          else {
            Expr nu = checkExpr(env, ne.exprs.elementAt(0));
            ne.exprs.setElementAt( nu, 0 );			
            if (!Types.isTypeType(getType(nu)))
              ErrorSet.error(nu.getStartLoc(),
                             "The argument must have TYPE type");
          }
          setType( e, Types.typecodeType );
          return e;
        }

      case TagConstants.WACK_DURATION:
      case TagConstants.WACK_WORKING_SPACE:
      case TagConstants.SPACE:
        {
          NaryExpr ne = (NaryExpr)e;
          if( ne.exprs.size() != 1 ) 
            ErrorSet.error(ne.sloc, 
                           "The function " + TagConstants.toString(tag) + 
                           " takes one argument");
          else {
            // Note: the arguments are not evaluated
            Expr nu = checkExpr(env, ne.exprs.elementAt(0));
            ne.exprs.setElementAt( nu, 0 );			
          }
          setType( e, Types.longType );
          return e;
        }

      case TagConstants.INVARIANT_FOR:
      case TagConstants.IS_INITIALIZED:
        {
          NaryExpr ne = (NaryExpr)e;
          // FIXME - Is this a one argument function ?
          if( ne.exprs.size() != 1 ) 
            ErrorSet.error( ne.sloc, 
			    "The function takes one argument");
          else {
            Expr nu = checkExpr(env, ne.exprs.elementAt(0));
            ne.exprs.setElementAt( nu, 0 );			
          }
          setType( e, Types.booleanType );
          return e;
        }

      case TagConstants.NOTMODIFIEDEXPR:
        {
          NotModifiedExpr ne = (NotModifiedExpr)e;
          ne.expr = checkExpr(env, ne.expr);
          setType( e, Types.booleanType );
          return e;
        }

      case TagConstants.MAX:
        {
          NaryExpr ne = (NaryExpr)e;
          if( ne.exprs.size() != 1 ) 
            ErrorSet.error( ne.sloc, 
                            "The function \\max takes one argument");
          else {
            Expr nu = checkExpr(env, ne.exprs.elementAt(0), Types.locksetType);
            ne.exprs.setElementAt( nu, 0 );			
          }
          setType( e, Types.javaLangObject() );
          return e;
        }

      case TagConstants.PRE:
        {
          NaryExpr ne = (NaryExpr)e;
          if( ne.exprs.size() != 1 ) {
            ErrorSet.error( ne.sloc, 
                            "The function \\old takes one argument");
            setType(e, Types.voidType);
          } else if (!isTwoStateContext) {
            ErrorSet.error(ne.sloc, 
                           "The function \\old cannot be used in this context");
            setType(e, Types.voidType);
          } else if (isInsidePRE) {
            ErrorSet.error(ne.sloc, "Nested applications of \\old not allowed");
            setType(e, Types.voidType);
          } else {
            isInsidePRE = true;
            Expr nu = checkExpr(env, ne.exprs.elementAt(0) );
            Assert.notFalse(isInsidePRE);
            isInsidePRE = false;
            ne.exprs.setElementAt( nu, 0 );			
            setType( e, getType( nu ) );
          }
          return e;
        }

      case TagConstants.TYPEOF:
        {
          NaryExpr ne = (NaryExpr)e;
          if( ne.exprs.size() != 1 ) 
            ErrorSet.error( ne.sloc, 
                            "The function \\typeof takes one argument");
          else {
            Expr ex = ne.exprs.elementAt(0);
            Expr nu = checkExpr(env, ex );
            ne.exprs.setElementAt( nu, 0 );			
            Type t = getType(nu);
            if (t instanceof PrimitiveType) {
              e = TypeExpr.make(ex.getStartLoc(),ex.getEndLoc(),t);
            }
          }
          setType( e, Types.typecodeType );
          return e;
        }

      case TagConstants.TYPEEXPR:
        {
          TypeExpr te = (TypeExpr)e;
          env.resolveType( sig, te.type );
          setType(e, Types.typecodeType );
          return e;
        }

      case TagConstants.LABELEXPR:
        {
          LabelExpr le = (LabelExpr)e;
          if (!isCurrentlyPredicateContext) {
            ErrorSet.error(le.getStartLoc(),
                           "Labeled predicates are not allowed in this context");
            setType(e, Types.booleanType);
          } else {
            isPredicateContext = true;
            le.expr = checkExpr(env, le.expr);
            setType(e, getType( le.expr ) );
          }
          return e;
        }

      case TagConstants.NUM_OF:
        {
          NumericalQuantifiedExpr qe = (NumericalQuantifiedExpr)e;
	
          // if (!isCurrentlyPredicateContext) {
          //     ErrorSet.error(qe.getStartLoc(),
          //	   "Quantified expressions are not allowed in this context");
          //   } else 
          {
            Env subenv = env;
	
            for( int i=0; i<qe.vars.size(); i++) {
              GenericVarDecl decl = qe.vars.elementAt(i);
              env.resolveType( sig, decl.type );
	    
              subenv = new EnvForLocals(subenv, decl);
            }
            isPredicateContext = true;
            qe.expr = checkExpr(subenv, qe.expr, Types.booleanType);
          }
          setType(e, Types.intType);
          return e;
        }

      case TagConstants.SUM:
      case TagConstants.PRODUCT:
      case TagConstants.MIN:
      case TagConstants.MAXQUANT:
        {
          GeneralizedQuantifiedExpr qe = (GeneralizedQuantifiedExpr)e;
	
          /*
            if (!isCurrentlyPredicateContext) {
            ErrorSet.error(qe.getStartLoc(),
            "Quantified expressions are not allowed in this context");
            } else */ 
          {
            Env subenv = env;
	
            for( int i=0; i<qe.vars.size(); i++) {
              GenericVarDecl decl = qe.vars.elementAt(i);
              env.resolveType( sig, decl.type );
	    
              subenv = new EnvForLocals(subenv, decl);
            }
            isPredicateContext = true;
            qe.rangeExpr = checkExpr(subenv, qe.rangeExpr, Types.booleanType);
            qe.expr = checkExpr(subenv, qe.expr, Types.intType);
          }
          setType(e, Types.intType);
          return e;
        }

      case TagConstants.FORALL:
      case TagConstants.EXISTS:
        {
          QuantifiedExpr qe = (QuantifiedExpr)e;
	
          if (!isCurrentlyPredicateContext) {
            ErrorSet.error(qe.getStartLoc(),
                           "Quantified expressions are not allowed in this context");
          } else {
            Env subenv = env;
	
            for( int i = 0; i < qe.vars.size(); i++) {
              GenericVarDecl decl = qe.vars.elementAt(i);
              env.resolveType(sig, decl.type);
	    
              subenv = new EnvForLocals(subenv, decl, false);
            }
            isPredicateContext = true;
            if (qe.rangeExpr != null) {
              qe.rangeExpr = checkExpr(subenv, qe.rangeExpr, Types.booleanType);
            }
            qe.expr = checkExpr(subenv, qe.expr, Types.booleanType);
          }
          if (qe.rangeExpr == null) { // skip
          } else if (tag == TagConstants.FORALL) {
            qe.expr = BinaryExpr.make(TagConstants.IMPLIES,
                                      qe.rangeExpr,qe.expr,Location.NULL);
            setType(qe.expr,Types.booleanType);
          } else {
            qe.expr = BinaryExpr.make(TagConstants.AND,
                                      qe.rangeExpr,qe.expr,Location.NULL);
            setType(qe.expr,Types.booleanType);
          }
          setType(e, Types.booleanType);
          return e;
        }

      case TagConstants.PARENEXPR:
      case TagConstants.NOT:
        {
          // the sub-expression is allowed to contain quantifiers and
          // labels if this expression is
          isPredicateContext = isCurrentlyPredicateContext;
          return super.checkExpr(env, e);
        }

      case TagConstants.ADD:
        {
          Expr ee = super.checkExpr(env, e);
          if (!Main.options().useOldStringHandling &&
              Types.isSameType(getType(ee),Types.javaLangString())) {
             boolean savedInAnnotation = inAnnotation;
             inAnnotation = true;
             BinaryExpr be = (BinaryExpr)ee;
             Expr left = be.left;
             Expr right = be.right;
             if (!Types.isSameType(getType(left),Types.javaLangString())) {
                Name nn = Name.make("java.lang.String.valueOf",left.getStartLoc());
                ExprVec a = ExprVec.make();
                a.addElement(left);
	        AmbiguousMethodInvocation mi = AmbiguousMethodInvocation.make(
                    nn,null,left.getStartLoc(),a);
                MethodInvocation mm = env.disambiguateMethodName(mi);
                left = checkExpr(env, mm);
             }
             if (!Types.isSameType(getType(right),Types.javaLangString())) {
                Name nn = Name.make("java.lang.String.valueOf",right.getStartLoc());
                ExprVec a = ExprVec.make();
                a.addElement(right);
	        AmbiguousMethodInvocation mi = AmbiguousMethodInvocation.make(
                    nn,null,right.getStartLoc(),a);
                MethodInvocation mm = env.disambiguateMethodName(mi);
                right = checkExpr(env, mm);
             }
	     int loc = be.locOp;
             Name n = Name.make(TagConstants.STRINGCATINFIX,loc);
             ExprVec args = ExprVec.make();
             args.addElement(left);
             args.addElement(right);
	     AmbiguousMethodInvocation ami = AmbiguousMethodInvocation.make(
                    n,null,loc,args);
             MethodInvocation m = env.disambiguateMethodName(ami);
             ee = checkExpr(env, m);
             inAnnotation = savedInAnnotation;
          }
          return ee;
        }

      case TagConstants.OR:
      case TagConstants.AND:
      case TagConstants.EQ:
      case TagConstants.NE:
        {
          BinaryExpr be = (BinaryExpr)e;
          // each argument is allowed to contain quantifiers and labels if
          // this expression is
          isPredicateContext = isCurrentlyPredicateContext;
          be.left = checkExpr(env, be.left);
          isPredicateContext = isCurrentlyPredicateContext;
          be.right = checkExpr(env, be.right);
          if (false && Types.isTypeType(getType(be.left)) &&
              // DO WE NEED TO check the composite expressions ??? FIXME TYPE-EQUIV
              Types.isTypeType(getType(be.right))) {
            setType( be, Types.booleanType);
          } else {
            Type t = checkBinaryExpr(be.op, be.left, be.right, be.locOp);
            setType( be, t );
          }
          return be;
        }
      
      case TagConstants.WILDREFEXPR:
        {
          // FIXME - WildRefExpr needs cleanup .  In current usage r.od is always null
          // on input.
          WildRefExpr r = (WildRefExpr)e;
          if (!isCurrentlySpecDesignatorContext) {
            setType(r, Types.errorType);
            ErrorSet.error(r.getStartLoc(),
                           "Reference wild cards allowed only as "+
                           "specification designators");
          } else {
            // Can't set the type, but need to check the type of the od
            if (r.od == null) {
              if (r.var instanceof AmbiguousVariableAccess) {
                AmbiguousVariableAccess a = (AmbiguousVariableAccess)r.var;
                Object o = env.disambiguateTypeOrFieldName(a.name);
                if (o instanceof TypeSig) {
                  r.od = TypeObjectDesignator.make(r.var.getStartLoc(),
                                                   (TypeSig)o );
                } else {
                  r.var = checkExpr(env,r.var);
                  // FIXME - really need locDot here
                  r.od = ExprObjectDesignator.make(
                                                   r.var.getEndLoc(),
                                                   r.var);
                }
                r.var = null;
              } else {
                r.var = checkExpr(env,r.var);
                // FIXME - really need locDot here
                r.od = ExprObjectDesignator.make(
                                                 r.var.getEndLoc(),
                                                 r.var);
              }
              checkObjectDesignator(env,r.od);
            } else {
              Type t = checkObjectDesignator(env,r.od);
              // FIXME
              //System.out.println("TYPE " + t);
            }
          } 
          return r;
        }      

      case TagConstants.ARRAYREFEXPR:
        {
          ArrayRefExpr r = (ArrayRefExpr)e;
	
          r.array = checkExpr(env, r.array);
          Type arrType = getType( r.array );
          r.index = checkExpr(env, r.index);
          Type t = getType( r.index );

          if( arrType instanceof ArrayType ) {
            setType( r, ((ArrayType)arrType).elemType );
            Type ndxType = 
              Types.isNumericType( t ) ? Types.unaryPromote( t ) : t;
            if( !Types.isSameType( ndxType, Types.intType ) &&
                !Types.isErrorType(ndxType) ) 
              ErrorSet.error(r.locOpenBracket, "Array index is not an integer");

          } else if( arrType.getTag() == TagConstants.LOCKSET ) {
            setType( r, Types.booleanType );
            if( !Types.isReferenceOrNullType( t ) &&
                !Types.isErrorType(t) )
              ErrorSet.error(r.locOpenBracket, 
                             "Can only index \\lockset with a reference type");
          } else {
            setType( r, Types.errorType );
            if (!Types.isErrorType(arrType) )
              ErrorSet.error( r.locOpenBracket, 
                              "Attempt to index a non-array value");
          }
          //alx: dw set universe modifier for ArrayRefExpr
          if (useUniverses)
          	determineUniverseForArrayRefExpr(r);
	  //alx-end
          return r;
        }

      case TagConstants.ARRAYRANGEREFEXPR:
        {
          ArrayRangeRefExpr r = (ArrayRangeRefExpr)e;
          if (!isCurrentlySpecDesignatorContext) {
            setType(r, Types.errorType);
            ErrorSet.error(r.getStartLoc(),
                           "Array ranges are allowed only as "+
                           "specification designators");
          } else {
	
            r.array = checkExpr(env, r.array);
            Type arrType = getType( r.array );
            if (r.lowIndex != null) r.lowIndex = checkExpr(env, r.lowIndex);
            if (r.highIndex != null) r.highIndex = checkExpr(env, r.highIndex);
            Type t = r.lowIndex == null ? null : getType( r.lowIndex );
            Type tt = r.highIndex == null ? null : getType( r.highIndex );

            if( arrType instanceof ArrayType ) {
              setType( r, ((ArrayType)arrType).elemType );
              if (t != null) {
                Type ndxType = 
                  Types.isNumericType( t ) ? Types.unaryPromote( t ) : t;
                if( !Types.isSameType( ndxType, Types.intType ) &&
                    !Types.isErrorType(ndxType) ) 
                  ErrorSet.error(r.locOpenBracket, "Array index is not an integer");
              }
              if (tt != null) {
                Type ndxType = 
                  Types.isNumericType( tt ) ? Types.unaryPromote( tt ) : tt;
                if( !Types.isSameType( ndxType, Types.intType ) &&
                    !Types.isErrorType(ndxType) ) 
                  ErrorSet.error(r.locOpenBracket, "Array index is not an integer");
              }

            } else {
              setType( r, Types.errorType );
              if (!Types.isErrorType(arrType) )
                ErrorSet.error( r.locOpenBracket, 
                                "Attempt to index a non-array value");
            }
          }

          return r;
        }
      case TagConstants.RESEXPR:
        {
          if (!isRESContext || returnType == null) {
            if (!Types.isErrorType(returnType))
              ErrorSet.error(e.getStartLoc(), 
                             "Keyword \\result is not allowed in this context");
            setType( e, Types.errorType );
          }
          else
            setType( e, returnType );

          return e;
        }

      case TagConstants.SETCOMPEXPR:
        {
          SetCompExpr s = (SetCompExpr)e;

          env.resolveType(sig, s.type);
          env.resolveType(sig, s.fp.type);
          Env subenv = new EnvForLocals(env,s.fp,false);
          boolean savedPredicateContext = isPredicateContext;
          isPredicateContext = true;
          s.expr = checkExpr(subenv, s.expr, Types.booleanType);
          isPredicateContext = savedPredicateContext;
          setType( e, s.type);
          // FIXME - CHeck that the type is only JMLObjectSet, JMLValueSet
          // Check that the predicate has the correct restricted form
          return e;
        }

      case TagConstants.NOTSPECIFIEDEXPR:
        {
          setType( e, Types.voidType);
          return e;
        }
      case TagConstants.EVERYTHINGEXPR:
        {
          if (!isCurrentlySpecDesignatorContext) {
            ErrorSet.error(e.getStartLoc(),
                           "Keyword \\everything is not allowed in this context");
            setType( e, Types.errorType);
          } else {
            setType( e, Types.voidType);
          }
          return e;
        }
      case TagConstants.NOTHINGEXPR:
        {
          if (!isCurrentlySpecDesignatorContext) {
            ErrorSet.error(e.getStartLoc(),
                           "Keyword \\nothing is not allowed in this context");
            setType( e, Types.errorType);
          } else {
            setType( e, Types.voidType);
          }
          return e;
        }
      case TagConstants.LOCKSETEXPR:
        {
          if (! isLocksetContext) {
            ErrorSet.error(e.getStartLoc(),
                           "Keyword \\lockset is not allowed in this context");
          }
          setType( e, Types.locksetType );
          return e;
        }

      case TagConstants.LE: 
      case TagConstants.LT: 
        {
          BinaryExpr be = (BinaryExpr)e;
          be.left = checkExpr(env, be.left);
          be.right = checkExpr(env, be.right);
	
          if( Types.isReferenceOrNullType( getType( be.left ) )
              && Types.isReferenceOrNullType( getType( be.right ) ) ) {
            // is lock comparison, and is ok
            setType( be, Types.booleanType );
            return be;
          } else {
            return super.checkExpr(env, e);
          }
        }

      case TagConstants.NEWINSTANCEEXPR:
      case TagConstants.NEWARRAYEXPR:
        {
          countFreeVarsAccesses++;
          /* FIXME - Yes it can, but it must be pure!
             if (inAnnotation) {
             ErrorSet.error(e.getStartLoc(),
             "new cannot be used in specification expressions");
             }
          */
          return super.checkExpr(env, e);
        }
      
      case TagConstants.ASSIGN: case TagConstants.ASGMUL:
      case TagConstants.ASGDIV: case TagConstants.ASGREM:
      case TagConstants.ASGADD: case TagConstants.ASGSUB:
      case TagConstants.ASGLSHIFT: case TagConstants.ASGRSHIFT:
      case TagConstants.ASGURSHIFT: case TagConstants.ASGBITAND:
      case TagConstants.ASGBITOR: case TagConstants.ASGBITXOR:
        {
          if (inAnnotation && !inModelBody) {
            BinaryExpr be = (BinaryExpr)e;
            ErrorSet.error(be.locOp,
			   "assignments cannot be used in specification expressions");
          }
          return super.checkExpr(env, e);
        }
      
      case TagConstants.INC: case TagConstants.DEC: 
      case TagConstants.POSTFIXINC: case TagConstants.POSTFIXDEC:
        {
          if (inAnnotation && !inModelBody) {
            UnaryExpr ue = (UnaryExpr)e;
            ErrorSet.error(ue.locOp,
			   "assignments cannot be used in specification expressions");
          }
          return super.checkExpr(env, e);
        }
      
      default:
        return super.checkExpr(env, e);
      }

    } finally {

      isPredicateContext = isCurrentlyPredicateContext;
    }
  }

  // Pragma checkers

  protected  void checkTypeDeclElemPragma(TypeDeclElemPragma e) {
    int tag = e.getTag();
    boolean savedInAnnotation = inAnnotation;
    inAnnotation = true;	// Must be reset before we exit!
    try {

      switch(tag) {
      case TagConstants.AXIOM:
      case TagConstants.INITIALLY:
      case TagConstants.INVARIANT:
      case TagConstants.CONSTRAINT: // FIXME - do we need to change the logic below to handle constraints?
        {
          ExprDeclPragma ep = (ExprDeclPragma)e;
          Env rootEnv = (tag == TagConstants.AXIOM ||
                         Modifiers.isStatic(ep.modifiers)) ? rootSEnv : rootIEnv;

          invariantContext = (tag == TagConstants.INVARIANT) ||
            tag == TagConstants.INITIALLY;
          isTwoStateContext = (tag == TagConstants.CONSTRAINT);
          boolean oldIsLocksetContext = isLocksetContext;
          isLocksetContext = false;
          if (invariantContext){
            // FIXME                       Assert.notFalse(countFreeVarsAccesses == 0);
            countFreeVarsAccesses = 0;
          }
	
          ep.expr = checkPredicate(rootEnv, ep.expr);
          isLocksetContext = oldIsLocksetContext;

          TypeSig sig = TypeSig.getSig(e.getParent());
          if (sig==javafe.tc.Types.javaLangObject() ||
              sig==javafe.tc.Types.javaLangCloneable()) {
            if (invariantContext) ErrorSet.fatal(e.getStartLoc(),
                                                 "java.lang.Object and java.lang.Cloneable may not"
                                                 + " contain invariants.");  // FIXME - Why?
          }
          /*
            FIXME - see uses of countFreeVarsAccess
            if (invariantContext && countFreeVarsAccesses == 0 &&
            // Don't print an error if the entire invariant
            // is an informal predicate
            escjava.parser.EscPragmaParser.
            informalPredicateDecoration.get(ep.expr)==null) {
            ErrorSet.error(e.getStartLoc(),
            "Class invariants must mention program variables"
            + " or fields.");
            }
          */

          if (invariantContext) {countFreeVarsAccesses = 0;}
          invariantContext = false;
          isTwoStateContext = false;
          break;
        }
      case TagConstants.REPRESENTS:
        {
          NamedExprDeclPragma ep = (NamedExprDeclPragma)e;
          boolean stat = Modifiers.isStatic(ep.modifiers);

          // What about static model vars?
          // Can the represents clause be static ? FIXME
          Env rootEnv = stat? rootSEnv : rootIEnv;
          ep.target = checkExpr(rootEnv, ep.target);

          if (ep.target instanceof FieldAccess) {
            invariantContext = false;
            isTwoStateContext = false;
            boolean oldIsLocksetContext = isLocksetContext;
            isLocksetContext = false;
            if (invariantContext){
              // FIXME                       Assert.notFalse(countFreeVarsAccesses == 0);
              countFreeVarsAccesses = 0;
            }
	    
            ep.expr = checkPredicate(rootEnv, ep.expr);
            isLocksetContext = oldIsLocksetContext;

            FieldAccess fa = (FieldAccess)ep.target;
            if (!Utils.isModel(fa.decl)) {
              ErrorSet.error(fa.getStartLoc(),
                             "A represents clause must name a model field",
                             fa.decl.locId);
            }
            if (stat && !Modifiers.isStatic(fa.decl.modifiers)) {
              ErrorSet.error(fa.getStartLoc(),
                             "A static represents clause must name a static model field");
            }
            if (!stat && Modifiers.isStatic(fa.decl.modifiers)) {
              ErrorSet.error(fa.getStartLoc(),
                             "A non-static represents clause must name a non-static model field");
            }

            /* THis is done in type prepping
               TypeDecl td = ep.parent;
               TypeDeclElemVec tv = (TypeDeclElemVec)Utils.representsDecoration.get(td);
               if (tv == null) {
               tv = TypeDeclElemVec.make(10);
               Utils.representsDecoration.set(td,tv);
               }
               tv.addElement(ep);
            */
               TypeDeclElemVec tv = (TypeDeclElemVec)Utils.representsDecoration.get(fa.decl);
               if (tv == null) {
		   tv = TypeDeclElemVec.make(10);
		   Utils.representsDecoration.set(fa.decl,tv);
               }
               tv.addElement(ep);
          } else if (!(ep.target instanceof AmbiguousVariableAccess)){
            // If the type is Ambiguous, then an Undefined variable
            // error has already been issued.  I'm not actually
            // sure that this point is reachable.
            ErrorSet.error(ep.target.getStartLoc(),
                           "Expected a field identifier here");
          }
          break;
        }
      case TagConstants.DEPENDS:
        {
          DependsPragma ep = (DependsPragma)e;
          // FIXME - perhaps use rootSEnv if the variable
          // being discussed is static ?
          Env rootEnv = rootIEnv;

          ep.target = checkExpr(rootEnv, ep.target);

          ExprVec ev = ep.exprs;
          for (int i=0; i<ev.size(); ++i) {
            ev.setElementAt(
                            checkExpr(rootEnv, ev.elementAt(i)), i);
          }

          // FIXME - Need to check that
          //	LHS is a simple model variable, a field of 
          //		this or a super class or an interface
          //	RHS consists of store-refs, no computed expressions
          //	RHS may have other model variables
          //	check access ?
          break;
        }

      case TagConstants.MODELCONSTRUCTORDECLPRAGMA: {
        ModelConstructorDeclPragma me = (ModelConstructorDeclPragma)e;
        ConstructorDecl cd = me.decl;
        Env rootEnv = Modifiers.isStatic(cd.modifiers)
          ? rootSEnv
          : rootIEnv;

        // All gets checked since the associated ConstructorDecl is
        // part of the type

        // FIXME - the body needs to allow ghost/model vars
        break;
      }


      case TagConstants.MODELMETHODDECLPRAGMA: {
        MethodDecl decl = ((ModelMethodDeclPragma)e).decl;
        Env rootEnv = Modifiers.isStatic(decl.modifiers)
          ? rootSEnv
          : rootIEnv;

        // All gets checked since the associated ConstructorDecl is
        // part of the type

        // FIXME - the body needs to allow ghost/model vars
        break;
      }

      case TagConstants.MONITORS_FOR: {
        IdExprDeclPragma emp = (IdExprDeclPragma)e;
        Identifier id = emp.id;
        TypeDeclElemVec elems = e.parent.elems;
        FieldDecl fd = null;
        for (int i=0; i<elems.size(); ++i) {
          TypeDeclElem td = elems.elementAt(i);
          if (td instanceof FieldDecl &&
              ((FieldDecl)td).id == id) {
            fd = (FieldDecl)td;
            break;
          }
        }
        boolean isStatic = false;
        if (fd == null) {
          ErrorSet.error(emp.loc, 
                         "Could not find identifier " + id + " in this class");
        } else {
          isStatic = Modifiers.isStatic(fd.modifiers);
        }
        if (Modifiers.isStatic(emp.modifiers)) isStatic = true;
        Env env = isStatic ? rootSEnv : rootIEnv;
        int oldAccessibilityLowerBound = accessibilityLowerBound;
        ASTNode oldAccessibilityContext = accessibilityContext;
        accessibilityLowerBound = getAccessibility(fd.modifiers);
        accessibilityContext = fd;
        emp.expr = checkExpr(env, emp.expr, Types.javaLangObject());
        accessibilityLowerBound = oldAccessibilityLowerBound;
        accessibilityContext = oldAccessibilityContext;
        fd.pmodifiers.addElement(
                                 ExprModifierPragma.make(
                                                         TagConstants.MONITORED_BY,
                                                         emp.expr,
                                                         emp.loc
                                                         ));
        break;
      }

      case TagConstants.WRITABLE:
      case TagConstants.READABLE: {
        NamedExprDeclPragma emp = (NamedExprDeclPragma)e;
	    
        isSpecDesignatorContext = true;
        Env newenv = rootIEnv;
        emp.target = checkDesignator(newenv, emp.target);
        isSpecDesignatorContext = false;
        emp.expr = checkPredicate(newenv, emp.expr);
        switch (emp.target.getTag()) {
        case TagConstants.FIELDACCESS: {
          FieldAccess fa = (FieldAccess)emp.target;
          if (fa.decl != null &&
              // The array "length" field has already been checked
              // insuper.checkDesignator().
              fa.decl != Types.lengthFieldDecl) {

            if (tag == TagConstants.WRITABLE &&
                Modifiers.isFinal(fa.decl.modifiers)) {
              // FIXME
            }
            fa.decl.pmodifiers.addElement(
                                          ExprModifierPragma.make(
                                                                  emp.tag == TagConstants.READABLE ?
                                                                  TagConstants.READABLE_IF:
                                                                  TagConstants.WRITABLE_IF,
                                                                  emp.expr,
                                                                  emp.loc
                                                                  )
                                          );
          }
          break;
        }
      
        case TagConstants.ARRAYREFEXPR:
        case TagConstants.ARRAYRANGEREFEXPR:
        case TagConstants.WILDREFEXPR:
        case TagConstants.EVERYTHINGEXPR:
        case TagConstants.NOTHINGEXPR:
        case TagConstants.NOTSPECIFIEDEXPR:
          // FIXME - not implemented
          break;

        default:
          if (escjava.parser.EscPragmaParser.
              informalPredicateDecoration.get(emp.target)==null) {
            // The expression is not a designator
            // but we allow an informal predicate
            if (!Types.isErrorType(getType(emp.target)))
              ErrorSet.error(emp.target.getStartLoc(),
                             "Not a specification designator expression");
          } else {
            emp.target = null;
          }
        }
        break;
      } 

      case TagConstants.MODELDECLPRAGMA: {
        FieldDecl decl = ((ModelDeclPragma)e).decl;
        Env rootEnv = Modifiers.isStatic(decl.modifiers)
          ? rootSEnv
          : rootIEnv;

        rootEnv.resolveType( sig, decl.type );
        checkModifierPragmaVec( decl.pmodifiers, decl, rootEnv );
        checkTypeModifiers(rootEnv, decl.type);

        // Check for both static and instance declarations

        if (Modifiers.isStatic(decl.modifiers)) {
          ModifierPragma inst = Utils.findModifierPragma(decl,
                                                         TagConstants.INSTANCE);
          if (inst != null) ErrorSet.error(inst.getStartLoc(),
                                           "May not specify both static and instance on a declaration");
        }

        /*
         * Check for other fields with the same name:
         */
        {
          TypeDeclElemVec elems = decl.parent.elems;
          FieldDecl fd;
          for (int i = 0; i<elems.size(); ++i) {
            TypeDeclElem tde = elems.elementAt(i);
            if (tde instanceof FieldDecl) {
              fd = (FieldDecl)tde;
            } else if (tde instanceof GhostDeclPragma) {
              fd = ((GhostDeclPragma)tde).decl;
            } else if (tde instanceof ModelDeclPragma) {
              fd = ((ModelDeclPragma)tde).decl;
            } else
              continue;
            if (fd.id ==  decl.id && fd != decl)
              ErrorSet.error(decl.locId,
                             "Another field named '"+decl.id.toString()
                             +"' already exists in this type", fd.locId);
          }
        }

        break;
      }

      case TagConstants.GHOSTDECLPRAGMA: {
        FieldDecl decl = ((GhostDeclPragma)e).decl;
        ModifierPragma inst = Utils.findModifierPragma(decl,
                                                       TagConstants.INSTANCE);
        // Check for both static and instance declarations

        if (Modifiers.isStatic(decl.modifiers)) {
          if (inst != null) ErrorSet.error(inst.getStartLoc(),
                                           "May not specify both static and instance on a declaration");
        }

        Env rootEnv = Modifiers.isStatic(decl.modifiers)
          ? rootSEnv
          : rootIEnv;

        rootEnv.resolveType( sig, decl.type );
        checkModifierPragmaVec( decl.pmodifiers, decl, rootEnv );
        checkTypeModifiers(rootEnv, decl.type);



        /*
         * Handle initializer:
         */

        if (decl.init != null) {
          leftToRight = true;
          allowedExceptions.removeAllElements();
          Assert.notFalse( allowedExceptions.size() == 0);
          decl.init = checkInit(rootEnv, decl.init, decl.type);
        }

        /*
         * Check for other fields with the same name:
         */
        {
          TypeDeclElemVec elems = decl.parent.elems;
          FieldDecl fd;
          for (int i = 0; i<elems.size(); ++i) {
            TypeDeclElem tde = elems.elementAt(i);
            if (tde instanceof FieldDecl) {
              fd = (FieldDecl)tde;
            } else if (tde instanceof GhostDeclPragma) {
              fd = ((GhostDeclPragma)tde).decl;
            } else if (tde instanceof ModelDeclPragma) {
              fd = ((ModelDeclPragma)tde).decl;
            } else
              continue;
            if (fd.id ==  decl.id && fd != decl)
              ErrorSet.error(decl.locId,
                             "Another field named '"+decl.id.toString()
                             +"' already exists in this type", fd.locId);
          }
        }


        break;
      }

      case TagConstants.STILLDEFERREDDECLPRAGMA: {
        StillDeferredDeclPragma sddp = (StillDeferredDeclPragma)e;
        if (!sig.hasField(sddp.var)) {
          ErrorSet.error(sddp.locId, "No such field");
        }

        // TBW:  To support still_deferred, one also needs to check that
        // "sddp.var" is declared as writable-deferred in the direct
        // superclass.
        break;
      }

      default:
        Assert.fail("Unexpected tag " + tag + 
                    " " + TagConstants.toString(tag));
      }
    } finally {
      inAnnotation = savedInAnnotation;
    }
  }

  protected Env checkModifierPragma(ModifierPragma p, ASTNode ctxt, Env env) {

    boolean savedInAnnotation = inAnnotation;
    inAnnotation = true;	// Must be reset before we exit!
    try {
      int tag = p.getTag();
      switch(tag) 
        {
        case TagConstants.EVERYTHINGEXPR:
        case TagConstants.EVERYTHING:
        case TagConstants.NOTHING:
        case TagConstants.NOT_SPECIFIED:
        case TagConstants.NOTHINGEXPR:
        case TagConstants.NOTSPECIFIEDEXPR:
          // FIXME - check the context???
          break;

        case TagConstants.UNINITIALIZED:
          if( ctxt.getTag() != TagConstants.LOCALVARDECL ) {
            int loc;
            if (ctxt instanceof GenericVarDecl)
              loc = ((GenericVarDecl)ctxt).locId;
            else
              loc = p.getStartLoc();
            ErrorSet.error(loc,
                           "The uninitialized annotation can occur only on "
                           +"local variable declarations");
          } else if( ((LocalVarDecl)ctxt).init == null ) {
            ErrorSet.error(((GenericVarDecl)ctxt).locId,
                           "The uninitialized annotation can occur only on "
                           +"local variable declarations "
                           +"that have an initializer");
          }
          break;
      
        case TagConstants.MONITORED:
          {
            if( ctxt.getTag() != TagConstants.FIELDDECL ) {
              int loc;
              if (ctxt instanceof GenericVarDecl)
                loc = ((GenericVarDecl)ctxt).locId;
              else
                loc = p.getStartLoc();
              ErrorSet.error(loc,
                             "The monitored annotation can occur only on "
                             +"field declarations");
              /* added functionality to have monitors on static fields 
                 } else if (env.isStaticContext()) {
                 FieldDecl fd = (FieldDecl)ctxt;
                 ErrorSet.error(fd.locId,
                 "The monitored annotation can occur only on "+
                 "instance field declarations");
              */
            }
            break;
          }

	case TagConstants.NULLABLE:
        case TagConstants.NON_NULL:
          // NOTE:  Part of the NON_NULL check is found in checkTypeDeclElem()
          // above, since there's not enough context information here to
          // determine whether a formal parameter is declared for a method
          // override.
          switch (ctxt.getTag()) {
          case TagConstants.FORMALPARADECL:
          case TagConstants.LOCALVARDECL:
          case TagConstants.FIELDDECL: {
            GenericVarDecl vd = (GenericVarDecl)ctxt;
            if (! Types.isReferenceType(vd.type)) {
              ErrorSet.error(vd.locId,
                             "The " + TagConstants.toString(tag) + 
			     " pragma can be applied only to "+
                             "variables, fields, and parameters of "+
                             "reference types");
            }
            break;
          }
          case TagConstants.METHODDECL: {
            MethodDecl md = (MethodDecl) ctxt;
            if (!Types.isReferenceType(md.returnType)) {
              ErrorSet.error(md.getStartLoc(),
                             "'" + TagConstants.toString(tag) + 
			     "' can only be used with methods whose "+
                             "result type is a reference type");
            }
            break;
          }
          default:
            ErrorSet.error(p.getStartLoc(),
                           "The " + TagConstants.toString(tag) + " pragma can be applied only to "+
                           "variables, fields, parameters, and methods");
            break;
          }
          break;
      
          // FIXME - should have a spec_protected case as well ???
        case TagConstants.SPEC_PUBLIC:
        case TagConstants.SPEC_PROTECTED:
          // JML now allows spec_public and spec_protected on declarations
          // of any java accessibiilty
          break;

        case TagConstants.WRITABLE_DEFERRED: 
          {
            if (ctxt.getTag() != TagConstants.FIELDDECL ||
                Modifiers.isStatic(((FieldDecl)ctxt).modifiers)) {
              ErrorSet.error(p.getStartLoc(),
                             "The writable_deferred annotation can occur only on "
                             +"instance field declarations");
            }
            break;
          }

        case TagConstants.INSTANCE:
          {
            int ctag = ctxt.getTag();

            if (!(ctxt instanceof GhostDeclPragma) &&
                !(ctxt instanceof ModelDeclPragma)) {

              // FIXME - should this use Utils.isModel ???
              if (ctxt instanceof FieldDecl &&
                  (Utils.findModifierPragma( ((FieldDecl)ctxt).pmodifiers, TagConstants.MODEL) != null ||
                   Utils.findModifierPragma( ((FieldDecl)ctxt).pmodifiers, TagConstants.GHOST) != null )) {
                // skip
              } else {
                ErrorSet.error(p.getStartLoc(),
                               "An instance modifier may only be applied to ghost and model fields");
              }
            }

            break;
          }

        case TagConstants.FUNCTION:
          // FIXME - do some checking
          // must be pure, static, immutable arguments, not void
          break;

        case TagConstants.PURE:
          {
            if (ctxt instanceof RoutineDecl) {
            } else if (ctxt instanceof TypeDecl) {
            } else {
              ErrorSet.error(p.getStartLoc(),
                             "Expected pure to modify a class, interface, constructor or method declaration");
            }
            break;
          }

        case TagConstants.HELPER:
          switch (ctxt.getTag()) {
          case TagConstants.CONSTRUCTORDECL:
            ((ConstructorDecl)ctxt).modifiers |= Modifiers.ACC_HELPER;
            break;
          case TagConstants.METHODDECL:
            {
              MethodDecl md = (MethodDecl) ctxt;
              md.modifiers |= Modifiers.ACC_HELPER;
              if (getOverrideStatus(md) != MSTATUS_NEW_ROUTINE) {
                ErrorSet.error(p.getStartLoc(),
                               "The helper pragma cannot be " +
                               "applied to a method override");
              } else if (isOverridable(md)) {
                ErrorSet.error(p.getStartLoc(),
                               "The helper pragma cannot be applied " +
                               "to method that can be overridden");
              } else if (md.body == null && !md.specOnly) {
                ErrorSet.error(p.getStartLoc(),
                               "The helper pragma cannot be applied " +
                               "to method without a body");
              }
              break;
            }
          default:
            ErrorSet.error(p.getStartLoc(),
                           "The helper pragma can only be applied to "+
                           "methods and constructors");
            break;
          }
          break;
	  
        case TagConstants.IMMUTABLE:
          {
            if (!(ctxt instanceof TypeDecl)) {
              ErrorSet.error(p.getStartLoc(),
                             "The immutable modifier may be applied only to type declarations");
            }
          }
          break;

        case TagConstants.WRITABLE_IF:
        case TagConstants.READABLE_IF:
          {
            ExprModifierPragma emp = (ExprModifierPragma)p;

            if( ctxt.getTag() != TagConstants.FIELDDECL
                && ctxt.getTag() != TagConstants.LOCALVARDECL ) {
              ErrorSet.error(p.getStartLoc(),
                             "The " + TagConstants.toString(tag)
                             +" annotation can occur only on "
                             +"field declarations and "
                             +"local variable declarations");
            }

            int oldAccessibilityLowerBound = accessibilityLowerBound;
            ASTNode oldAccessibilityContext = accessibilityContext;
            if (ctxt.getTag() == TagConstants.FIELDDECL) {
              FieldDecl fd = (FieldDecl)ctxt;
              accessibilityLowerBound = getAccessibility(fd.modifiers);
              accessibilityContext = fd;
            }
            emp.expr = checkPredicate(env, emp.expr);
            accessibilityLowerBound = oldAccessibilityLowerBound;
            accessibilityContext = oldAccessibilityContext;
            break;
          }

        case TagConstants.NO_WACK_FORALL:
        case TagConstants.OLD:
          {
            VarDeclModifierPragma vd = (VarDeclModifierPragma)p;
            LocalVarDecl x = vd.decl;
            env.resolveType( sig, vd.decl.type );
            checkTypeModifiers(env, x.type);
            javafe.tc.PrepTypeDeclaration.getInst().
              checkModifiers(x.modifiers, Modifiers.NONE,
                             x.locId, "local specification variable");

            boolean savedContext = isTwoStateContext;
            isTwoStateContext = true;
            // The case of x.init==null is illegal and should have
            // been caught by the parser.
            if (x.init != null) {
              //if (x.init instanceof ArrayInit) {
              x.init = checkInit(env, x.init, x.type);
              //} else {
              //checkExpr(newEnv, (Expr)x.init, x.type);
              //}
            }
            isTwoStateContext = savedContext;
            // We create the new environment after checking the
            // initializer to be sure that the initializer does not
            // reference itself.
            env = new EnvForGhostLocals(env,x);
            break;
          }

        case TagConstants.ALSO_REQUIRES:
        case TagConstants.REQUIRES:
        case TagConstants.PRECONDITION:
          {
            ExprModifierPragma emp = (ExprModifierPragma)p;

            if( ctxt instanceof InitBlock ) {
              if (emp.expr.getTag() != TagConstants.NOTSPECIFIEDEXPR)
                emp.expr = checkPredicate(env, emp.expr);
            } else if( ctxt instanceof RoutineDecl ) {
              RoutineDecl rd = (RoutineDecl)ctxt;

              int ms = getOverrideStatus(rd);

              Env newenv = env;
              /* Interpret constructor as acting on an existing object, to initialize it,
                 so this and fields are in scope.
                 if (rd instanceof ConstructorDecl) {
                 newenv = env.asStaticContext();
                 }
              */
              int oldAccessibilityLowerBound = accessibilityLowerBound;
              ASTNode oldAccessibilityContext = accessibilityContext;
              accessibilityLowerBound = getAccessibility(rd.modifiers);
              accessibilityContext = rd;
              if (emp.expr.getTag() != TagConstants.NOTSPECIFIEDEXPR)
                emp.expr = checkPredicate(newenv, emp.expr);
              accessibilityLowerBound = oldAccessibilityLowerBound;
              accessibilityContext = oldAccessibilityContext;
            } else {
              ErrorSet.error(p.getStartLoc(), TagConstants.toString(tag) +
                             " annotations can occur only on method" +
                             ((tag == TagConstants.REQUIRES || 
                               tag == TagConstants.PRECONDITION) ? " and constructor" : "") +
                             " declarations");
            }
            break;
          }

        case TagConstants.MEASURED_BY:
        case TagConstants.DURATION:
        case TagConstants.WORKING_SPACE:
          {
            CondExprModifierPragma emp = (CondExprModifierPragma)p;

            if( !(ctxt instanceof RoutineDecl ) ) {
              ErrorSet.error(p.getStartLoc(),
                             TagConstants.toString(tag)
                             +" annotations can occur only on "
                             +"method and constructor declarations");
            } else {
              RoutineDecl rd = (RoutineDecl)ctxt;
              boolean oldIsRESContext = isRESContext;
              boolean oldIsTwoStateContext = isTwoStateContext;
              boolean oldIsPrivFieldAccessAllowed = isPrivateFieldAccessAllowed;
              isRESContext = true;
              isTwoStateContext = true;
              // If "rd" is an overridable method, then every private field
              // mentioned in "emp.expr" must be spec_public.
              if (rd instanceof MethodDecl && isOverridable((MethodDecl)rd)) {
                isPrivateFieldAccessAllowed = false;
              }
              try {
                if (tag == TagConstants.MEASURED_BY) {
                  // FIXME - what type to use?
                  emp.expr = checkExpr(env, emp.expr);
                } else if (emp.expr.getTag() != TagConstants.NOTSPECIFIEDEXPR)
                  emp.expr = checkExpr(env, emp.expr, Types.longType);
                if (emp.cond != null)
                  emp.cond = checkExpr(env, emp.cond, Types.booleanType);
              } finally {
                isRESContext = oldIsRESContext;
                isTwoStateContext = oldIsTwoStateContext;
                isPrivateFieldAccessAllowed = oldIsPrivFieldAccessAllowed;
              }
            }
            break;
          }

        case TagConstants.DIVERGES:
            if (ctxt instanceof RoutineDecl) {
		RoutineDecl rd = (RoutineDecl)ctxt;
                if (Utils.isPure(rd)) {
		ExprModifierPragma emp = (ExprModifierPragma)p;
                if (emp.expr instanceof LiteralExpr &&
			((LiteralExpr)emp.expr).value == Boolean.FALSE) {
		  // OK
                } else if (p.getStartLoc() == Location.NULL) {
		  // Default clause - ignore it and deal with it during desugaring
                } else {
                  int locp = Utils.findPurePragma(rd).getStartLoc();
                  int loc = p.getStartLoc();
                  if (loc == Location.NULL) {
                    ErrorSet.error(locp,
                             "A lightweight specification case for a pure method must have a 'diverges false' clause");
                  } else {
                    ErrorSet.error(loc,
                             "A pure method may not have a diverges clause",
                             locp);
                  }
                }
              }
            }
            // fall-through
        case TagConstants.ENSURES:
        case TagConstants.ALSO_ENSURES:
        case TagConstants.POSTCONDITION:
        case TagConstants.WHEN:
          {
            ExprModifierPragma emp = (ExprModifierPragma)p;

            if( ctxt instanceof InitBlock ) {
              boolean oldIsRESContext = isRESContext;
              boolean oldIsTwoStateContext = isTwoStateContext;
              boolean oldIsPrivFieldAccessAllowed = isPrivateFieldAccessAllowed;
              try {
                isRESContext = true;
                isTwoStateContext = true;
                if (emp.expr.getTag() != TagConstants.NOTSPECIFIEDEXPR)
                  emp.expr = checkPredicate(env, emp.expr);
              } finally {
                isRESContext = oldIsRESContext;
                isTwoStateContext = oldIsTwoStateContext;
                isPrivateFieldAccessAllowed = oldIsPrivFieldAccessAllowed;
              }
            } else if( ctxt instanceof RoutineDecl ) {
              RoutineDecl rd = (RoutineDecl)ctxt;
              escjava.ast.Utils.owningDecl.set(emp,rd);
              boolean oldIsRESContext = isRESContext;
              boolean oldIsTwoStateContext = isTwoStateContext;
              boolean oldIsPrivFieldAccessAllowed = isPrivateFieldAccessAllowed;
              isRESContext = true;
              isTwoStateContext = true;
              // If "rd" is an overridable method, then every private field
              // mentioned in "emp.expr" must be spec_public.
              if (rd instanceof MethodDecl && isOverridable((MethodDecl)rd)) {
                isPrivateFieldAccessAllowed = false;
              }
              try {
                if (emp.expr.getTag() != TagConstants.NOTSPECIFIEDEXPR)
                  emp.expr = checkPredicate(env, emp.expr);
              } finally {
                isRESContext = oldIsRESContext;
                isTwoStateContext = oldIsTwoStateContext;
                isPrivateFieldAccessAllowed = oldIsPrivFieldAccessAllowed;
              }
            } else {
              ErrorSet.error(p.getStartLoc(),
                             TagConstants.toString(tag)
                             +" annotations can occur only on "
                             +"initializer, method and constructor declarations");
            }
            break;
          }


        case TagConstants.SIGNALS:
          {
            tag = p.originalTag();
            VarExprModifierPragma vemp = (VarExprModifierPragma)p;

            if( !(ctxt instanceof RoutineDecl ) ) {
              ErrorSet.error(p.getStartLoc(),
                             TagConstants.toString(tag)
                             +" annotations can occur only on "
                             +"method and constructor declarations");
            } else {
              RoutineDecl rd = (RoutineDecl)ctxt;

              // Resolve type and check that it is a subtype of Throwable
              // and comparable to some type mentioned in the throws set.
              env.resolveType(sig,vemp.arg.type);
              //TypeSig top = Main.options().useThrowable ?
	//		  Types.javaLangThrowable() : Types.javaLangException();
              TypeSig top = Types.javaLangThrowable();
                 // FIXME - JML actually requires that the var be a subtype of
		 // Exception, but the original Esc/Java did not
              if (!Types.isSubclassOf(vemp.arg.type,top)) {
                ErrorSet.error(vemp.arg.type.getStartLoc(),
                               "The type of the " +
                               TagConstants.toString(tag) +
                               " argument must be a subtype of " +
                               Types.printName(top));
              } else {
                // "vemp.arg.type" is a subclass of "Throwable", so it
                // must be a "TypeName" or a "TypeSig"
                TypeSig ssig;
                if (vemp.arg.type instanceof TypeSig) {
                  ssig = (TypeSig)vemp.arg.type;
                } else {
                  ssig = TypeSig.getSig((TypeName)vemp.arg.type);
                }
                boolean okay = false;
                for (int i = 0; i < rd.raises.size(); i++) {
                  TypeName tn = rd.raises.elementAt(i);
                  TypeSig tsig = TypeSig.getSig(tn);
                  if (Types.isSubclassOf(ssig, tsig) ||
                      Types.isSubclassOf(tsig, ssig)) {
                    okay = true;
                    break;
                  }
                }
                /* FIXME - what about Error exceptions, must they be mentioned?  */
                /* FIXME
                   if (!okay) {
                   if (!( (vemp.expr instanceof LiteralExpr) &&
                   ((LiteralExpr)vemp.expr).value.equals(Boolean.FALSE))) {
                   ErrorSet.error(vemp.arg.type.getStartLoc(),
                   "The type of the " +
                   TagConstants.toString(tag) +
                   " argument must be comparable to some type"+
                   " mentioned in the routine's throws set");
                   }
                   }
                */
              }

              if (tag == TagConstants.SIGNALS_ONLY) {
                  // Check that all Exceptions listed are either in the
                  // throws list or are RuntimeExceptions

                  // FIXME

              }

              Env subenv = new EnvForLocals(env, vemp.arg);
              // FIXME - below we say that this is a twostate context, in which case we should not set this to static???
              /*
                if (rd instanceof ConstructorDecl) {
                subenv = subenv.asStaticContext();
                }
              */

              // Check the expression to be an appropriate predicate
              boolean oldIsTwoStateContext = isTwoStateContext;
              boolean oldIsPrivFieldAccessAllowed = isPrivateFieldAccessAllowed;
              isTwoStateContext = true;
              // If "rd" is an overridable method, then every private field
              // mentioned in "emp.expr" must be spec_public.
              if (rd instanceof MethodDecl && isOverridable((MethodDecl)rd)) {
                isPrivateFieldAccessAllowed = false;
              }
              try { 
                if (vemp.expr.getTag() != TagConstants.NOTSPECIFIEDEXPR)
                  vemp.expr = checkPredicate(subenv, vemp.expr);
              } finally {
                isTwoStateContext = oldIsTwoStateContext;
                isPrivateFieldAccessAllowed = oldIsPrivFieldAccessAllowed;
              }
            }
            break;
          }


        case TagConstants.MONITORED_BY: {
          ExprModifierPragma emp = (ExprModifierPragma)p;

          if (ctxt.getTag() != TagConstants.FIELDDECL) {
            ErrorSet.error(emp.loc,
                           "The monitored_by annotation can occur only on "+
                           "field declarations");
          } else {
            FieldDecl fd = (FieldDecl)ctxt;

            int oldAccessibilityLowerBound = accessibilityLowerBound;
            ASTNode oldAccessibilityContext = accessibilityContext;
            accessibilityLowerBound = getAccessibility(fd.modifiers);
            accessibilityContext = fd;
            emp.expr = checkExpr(env, emp.expr, Types.javaLangObject());
            accessibilityLowerBound = oldAccessibilityLowerBound;
            accessibilityContext = oldAccessibilityContext;
          }
          break;
        }

        case TagConstants.MODIFIESGROUPPRAGMA: {
          ModifiesGroupPragma mg = (ModifiesGroupPragma)p;
          if (ctxt instanceof InitBlock || ctxt instanceof RoutineDecl) {
            CondExprModifierPragmaVec v = mg.items;
            for (int i=0; i<v.size(); ++i) {
              checkModifierPragma(v.elementAt(i),ctxt,env);
            }
            if (mg.precondition != null) 
              mg.precondition = checkPredicate(env,mg.precondition); // FIXME - pre environment ?
          } else {
            ErrorSet.error(mg.clauseLoc,
                           "A modifies annotation " +
                           "can occur only on " +
                           "method and constructor declarations");
          }
          break;
        }

        case TagConstants.MODIFIES:
        case TagConstants.ASSIGNABLE:
        case TagConstants.ONLY_ASSIGNED:
        case TagConstants.MODIFIABLE:
        case TagConstants.STILL_DEFERRED: {
          CondExprModifierPragma emp = (CondExprModifierPragma)p;

          if ((ctxt instanceof RoutineDecl ) ) {
            RoutineDecl rd = (RoutineDecl)ctxt;
	    
            Assert.notFalse(!isSpecDesignatorContext);
            isSpecDesignatorContext = true;
            Env newenv = env;
            /*
            // But we do need to allow the fields of this in the modifies clause, which
            // using the static context does not permit.
            if (rd instanceof ConstructorDecl) {
            // disallow "this" from constructor "modifies" clauses
            newenv = env.asStaticContext();
            }
            */
            emp.expr = checkDesignator(newenv, emp.expr);
            switch (emp.expr.getTag()) {
            case TagConstants.FIELDACCESS: {
              FieldAccess fa = (FieldAccess)emp.expr;
              if (fa.decl != null &&
                  (ctxt instanceof MethodDecl) && // FIXME - what about constructors
                  Modifiers.isFinal(fa.decl.modifiers) &&
                  // The array "length" field has already been checked
                  // insuper.checkDesignator().
                  fa.decl != Types.lengthFieldDecl) {

                // java.lang.System has fields in, out, err that are special
                // cases.  Somehow, Java allows them to be final and yet be
                // modified by public routines.  Instead of a general 
                // mechanism, we just do a special case here.
                if (fa.decl.parent != Types.javaLangSystem().getTypeDecl())
                  ErrorSet.caution(fa.locId, "a final field is not allowed as " +
                                   "the designator in a modifies clause");
              }
              break;
            }
	  
            case TagConstants.ARRAYREFEXPR:
            case TagConstants.ARRAYRANGEREFEXPR:
            case TagConstants.WILDREFEXPR:
            case TagConstants.EVERYTHINGEXPR:
            case TagConstants.NOTHINGEXPR:
            case TagConstants.NOTSPECIFIEDEXPR:
              break;

            default:
              if (escjava.parser.EscPragmaParser.
                  informalPredicateDecoration.get(emp.expr)==null) {
                // The expression is not a designator
                // but we allow an informal predicate
                if (!Types.isErrorType(getType(emp.expr)))
                  ErrorSet.error(emp.expr.getStartLoc(),
                                 "Not a specification designator expression");
              } else {
                emp.expr = null;
              }
            }
            if (rd instanceof MethodDecl && Utils.isPure(rd) &&
                emp.expr != null && emp.expr.getTag() != TagConstants.NOTHINGEXPR) {
              ErrorSet.error(p.getStartLoc(),
                             "A pure method may not have a modifies clause",
                             Utils.findPurePragma(rd).getStartLoc());
            }
            if (rd instanceof ConstructorDecl && Utils.isPure(rd) &&
                emp.expr != null && 
                !(emp.expr.getTag() == TagConstants.NOTHINGEXPR ||
                  (emp.expr.getTag() == TagConstants.WILDREFEXPR &&
                   (((WildRefExpr)emp.expr).var instanceof ThisExpr) &&
                   ((ThisExpr)((WildRefExpr)emp.expr).var).classPrefix == null)
                  ||
                  ((emp.expr instanceof FieldAccess) &&
                   Types.isSubclassOf(
                                      TypeSig.getSig(rd.parent),
                                      TypeSig.getSig(((FieldAccess)emp.expr).decl.parent)
                                      )
                   ))
                ) {
              ErrorSet.error(p.getStartLoc(),
                             "A pure constructor may not have a modifies clause",
                             Utils.findPurePragma(rd).getStartLoc());
            }
            isSpecDesignatorContext = false;
            if (emp.cond != null) emp.cond = checkExpr(newenv, emp.cond);
          }
          break;
        }

        case TagConstants.ALSO:
        case TagConstants.ALSO_REFINE:
        case TagConstants.MODEL_PROGRAM:
        case TagConstants.CODE_CONTRACT:
        case TagConstants.BEHAVIOR:
        case TagConstants.CLOSEPRAGMA:
        case TagConstants.EXAMPLE:
        case TagConstants.EXCEPTIONAL_BEHAVIOR:
        case TagConstants.EXCEPTIONAL_EXAMPLE:
        case TagConstants.FOR_EXAMPLE:
        case TagConstants.IMPLIES_THAT:
        case TagConstants.NORMAL_BEHAVIOR:
        case TagConstants.NORMAL_EXAMPLE:
        case TagConstants.OPENPRAGMA:
          // Desugaring happens before type-checking,
          // This shouldn't happen.
          break;

        case TagConstants.GHOST:
        case TagConstants.MODEL:
          break;

        case TagConstants.NESTEDMODIFIERPRAGMA:
          {
            java.util.ArrayList list = ((NestedModifierPragma)p).list;
            java.util.Iterator i = list.iterator();
            while (i.hasNext()) {
              ModifierPragmaVec mpv = (ModifierPragmaVec)i.next();
              checkModifierPragmaVec(mpv,ctxt,env);
            }
            break;
          }

        case TagConstants.PARSEDSPECS:
          {
            escjava.ParsedRoutineSpecs pp = ((ParsedSpecs)p).specs;
            java.util.Iterator i = pp.specs.iterator();
            while (i.hasNext()) {
              checkModifierPragmaVec((ModifierPragmaVec)i.next(),ctxt,env);    
            }
            i = pp.impliesThat.iterator();
            while (i.hasNext()) {
              checkModifierPragmaVec((ModifierPragmaVec)i.next(),ctxt,env);    
            }
            i = pp.examples.iterator();
            while (i.hasNext()) {
              checkModifierPragmaVec((ModifierPragmaVec)i.next(),ctxt,env);    
            }
            /* This duplicates stuff
            // The last element is the ParsedSpecs containing all the
            // clauses etc.
            ModifierPragmaVec mpv = pp.modifiers;
            ModifierPragma last = mpv.elementAt(mpv.size()-1);
            mpv.removeElementAt(mpv.size()-1);
            checkModifierPragmaVec(mpv,ctxt,env);
            mpv.addElement(last);
            */
            break;
          }

        case TagConstants.MAPS: {
          Identifier fid = ((FieldDecl)ctxt).id;
          MapsExprModifierPragma ep = (MapsExprModifierPragma)p;
          //System.out.println("FOUND " + TagConstants.toString(tag) + " for " + fid + " " + ep.id);
          if (ep.expr != null) ep.expr = checkExpr(env,ep.expr);
          isSpecDesignatorContext = true;
          if (ep.mapsexpr != null) ep.mapsexpr = checkDesignator(env,ep.mapsexpr);
          isSpecDesignatorContext = false;
          Expr e = ep.expr;
          if (e == null || TypeCheck.inst.getType(e) == Types.errorType) {
            // skip
          } else if (e instanceof FieldAccess) {
            FieldAccess fe = (FieldAccess)e; // datagroup
            FieldDecl fd = (FieldDecl)ctxt;  // field with which maps decl is associated
            if (Modifiers.isStatic( fe.decl.modifiers)
                && !Modifiers.isStatic( fd.modifiers) ){
              ErrorSet.error(((FieldDecl)ctxt).getStartLoc(), 
                     "An instance field may not be added to a static datagroup",
                     fe.decl.getStartLoc());
            } else {
              Datagroups.add(fd.parent,fe.decl,ep.mapsexpr);
            }
          } else {
            ErrorSet.error(e.getStartLoc(),
                           "Expected a field reference here, found " +
                           e.getClass());
          }
          break;
        }

        case TagConstants.IN: {
          MapsExprModifierPragma ep = (MapsExprModifierPragma)p;
          if (ep.expr != null) ep.expr = checkExpr(env,ep.expr);
          Expr e = ep.expr;
          if (e == null || TypeCheck.inst.getType(e) == Types.errorType) {
          } else if (e instanceof FieldAccess) {
            FieldAccess fe = (FieldAccess)e;  // datagroup
            FieldDecl fd = (FieldDecl)ctxt;  // field being put in the datagroup
            Expr eva = AmbiguousVariableAccess.make(
                                    SimpleName.make(fd.id,fd.getStartLoc()));
            eva = checkExpr(env,eva);
            if (Modifiers.isStatic( fe.decl.modifiers ) &&
                !Modifiers.isStatic(fd.modifiers)) {
              ErrorSet.error(fd.getStartLoc(), 
                   "An instance field may not be added to a static datagroup", 
                    fe.decl.getStartLoc());
            } else {
              Datagroups.add(fd.parent,fe.decl,eva);
            }
          } else {
            ErrorSet.error(e.getStartLoc(),
                           "Expected a field reference here, found " +
                           e.getClass());
          }
          break;
        }

	//alx: commented out not to interfere with the dw implementation
        case TagConstants.READONLY:
        case TagConstants.PEER:
        case TagConstants.REP:
	    // FIXME - need to support these
	    break;
	//alx-end

        default:
          ErrorSet.error(p.getStartLoc(),
                         "Ignored unexpected " +  
                         TagConstants.toString(tag) +
                         " tag");
          break;
        }
    } finally {
      inAnnotation = savedInAnnotation;
    }
    return env;
  }

  /**
   * @return whether or not <code>md</code> can be overridden in some possible
   * subclass.
   */

  public static boolean isOverridable(/*@ non_null */ MethodDecl md) {
    return !(Modifiers.isPrivate(md.modifiers) ||
             Modifiers.isFinal(md.modifiers) ||
             Modifiers.isFinal(md.parent.modifiers) ||
             Modifiers.isStatic(md.modifiers));
  }

  /**
   * @return a value appropriate for assignment to
   * <code>accessibilityLowerBound</code>, given member modifiers.
   */

  protected int getAccessibility(int modifiers) {
    if (Modifiers.isPrivate(modifiers)) {
      return ACC_LOW_BOUND_Private;
    } else if (Modifiers.isPackage(modifiers)) {
      return ACC_LOW_BOUND_Package;
    } else if (Modifiers.isProtected(modifiers)) {
      return ACC_LOW_BOUND_Protected;
    } else {
      Assert.notFalse(Modifiers.isPublic(modifiers));
      return ACC_LOW_BOUND_Public;
    }
  }

  protected Env checkStmtPragma(Env e, StmtPragma s) {
    boolean savedInAnnotation = inAnnotation;
    inAnnotation = true;	// Must be reset before we exit!
    boolean savedTwoStateContext = isTwoStateContext;
    isTwoStateContext = true;
    try {
      int tag = s.getTag();
      switch(tag) {
      case TagConstants.UNREACHABLE:
        break;

      case TagConstants.SETSTMTPRAGMA: {
        SetStmtPragma set = (SetStmtPragma)s;
        set.target = checkExpr(e, set.target);
        set.value = checkExpr(e, set.value);
        checkBinaryExpr(TagConstants.ASSIGN, set.target,
                        set.value, set.locOp);
        Expr t = set.target;
        int nonGhostLoc = isGhost(t);
        if (nonGhostLoc != 0) {
          ErrorSet.error(s.getStartLoc(),"May use set only with assignment targets that are declared ghost",nonGhostLoc);
        }
        /*
          if (t instanceof FieldAccess &&
          ((FieldAccess)t).decl == Types.lengthFieldDecl) {
          ErrorSet.error(s.getStartLoc(),"The length field of an array may not be set");
          }
        */
        break;
      }

      case TagConstants.HENCE_BY:
      case TagConstants.ASSUME:
      case TagConstants.ASSERT:
        {
          ExprStmtPragma es = (ExprStmtPragma)s;
          es.expr = checkPredicate(e, es.expr);
          if (es.label != null) 
            es.label = checkExpr(e, es.label);
          break;
        }
      
      case TagConstants.LOOP_INVARIANT:
      case TagConstants.MAINTAINING:
        {
          ExprStmtPragma lis = (ExprStmtPragma)s;
          loopInvariants.addElement(lis);
          break;
        }

      case TagConstants.DECREASES:
      case TagConstants.DECREASING:
        {
          ExprStmtPragma lis = (ExprStmtPragma)s;
          loopDecreases.addElement(lis);
          break;
        }

      case TagConstants.LOOP_PREDICATE:
        {
          ExprStmtPragma lis = (ExprStmtPragma)s;
          loopPredicates.addElement(lis);
          break;
        }

      case TagConstants.SKOLEMCONSTANTPRAGMA:
        {
          SkolemConstantPragma p = (SkolemConstantPragma)s;
          skolemConstants.append(p.decls);
          break;
        }

      default:
        Assert.fail("Unexpected tag " + tag +" "+s +
                    " " + Location.toString(s.getStartLoc()));
      }
    } finally {
      inAnnotation = savedInAnnotation;
      isTwoStateContext = savedTwoStateContext;
    }
    return e;
  }


  // Utility routines

  /**
   * Copy the Type associated with an expression by the typechecking pass to
   * another Expr.  This is used by Substitute to ensure it returns an Expr of the
   * same Type.
   */
  public static void copyType(VarInit from, VarInit to) {
    Type t = getTypeOrNull(from);
    if (t != null)
      setType(to, t);
  }

  /**
   * @return a set of *all* the methods a given method overrides. This includes
   * transitivity.
   */
  //@ ensures \result.elementType == \type(MethodDecl);
  public static /*@non_null*/Set getAllOverrides(/*@non_null*/MethodDecl md) {
    Set direct = javafe.tc.PrepTypeDeclaration.getInst().getOverrides(md.parent, md);
    Set result = new Set();

    Enumeration e = direct.elements();
    while (e.hasMoreElements()) {
      MethodDecl directMD = (MethodDecl)(e.nextElement());
      result.add(directMD);
      result.addEnumeration(getAllOverrides(directMD).elements());
    }

    return result;
  }

  public static javafe.util.Set getDirectOverrides(MethodDecl md) {
    return javafe.tc.PrepTypeDeclaration.getInst().getOverrides(md.parent, md);
  }

  /**
   * @return If <code>md</code> is a method that overrides a method in a
   * superclass, the overridden method is returned.  Otherwise, if <code>md</code>
   * overrides a method in an interface, such a method is returned.  Otherwise,
   * <code>null</code> is returned.
   */

  public static MethodDecl getSuperClassOverride(MethodDecl md) {
    MethodDecl classOverride = null;
    MethodDecl interfaceOverride = null;
    Set direct = javafe.tc.PrepTypeDeclaration.getInst().getOverrides(md.parent, md);
    Enumeration e = direct.elements();
    while (e.hasMoreElements()) {
      MethodDecl directMD = (MethodDecl)(e.nextElement());
      if (directMD.parent instanceof ClassDecl) {
        if (classOverride == null) {
          classOverride = directMD;
        } else {
          Assert.fail("we think this can no longer happen!");
          // This suggests that the method is inherited from TWO classes!
          // This can actually happen, if the method is one that is
          // declared in Object, because every interface has the methods of
          // Object.  In this case, pick the method override that does not
          // reside in Object.
          Type javaLangObject = Types.javaLangObject();
          Type t0 = TypeSig.getSig(classOverride.parent);
          Type t1 = TypeSig.getSig(directMD.parent);
          Assert.notFalse(Types.isSameType(t0, javaLangObject) ||
                          Types.isSameType(t1, javaLangObject));
          Assert.notFalse(!Types.isSameType(t0, javaLangObject) ||
                          !Types.isSameType(t1, javaLangObject));
          if (!Types.isSameType(t1, javaLangObject)) {
            classOverride = directMD;
          }
        }
      } else {
        interfaceOverride = directMD;
      }
    }
    if (classOverride != null) {
      return classOverride;
    } else {
      return interfaceOverride;
    }
  }

  /**
   * @return whether or not <code>rd</code> is a method override declaration, that
   * is, whether or not <code>rd</code> overrides a method declared in a superclass
   * or superinterface.
   */

  public static boolean isMethodOverride(RoutineDecl rd) {
    return getOverrideStatus(rd) != MSTATUS_NEW_ROUTINE;
  }	

  static public final int MSTATUS_NEW_ROUTINE = 0;
  static public final int MSTATUS_CLASS_NEW_METHOD = 1;
  static public final int MSTATUS_OVERRIDE = 2;

  /**
   * @return <code>MSTATUS_NEW_ROUTINE</code> if <code>rd</code> is a routine that
   * doesn't override any other method.  This includes the case where
   * <code>rd</code> is a constructor or a static method.
   *
   * <p> Returns <code>MSTATUS_CLASS_NEW_METHOD</code> if <code>rd</code> is a
   * method declared in a class, doesn't override any method in any superclass, but
   * overrides a method in an interface.
   *
   * <p> Otherwise, returns <code>MSTATUS_OVERRIDE</code>.
   */

  /*@ ensures \result == MSTATUS_NEW_ROUTINE ||
    @         \result == MSTATUS_CLASS_NEW_METHOD ||
    @         \result == MSTATUS_OVERRIDE; 
  */
  public static int getOverrideStatus(/*@ non_null */ RoutineDecl rd) {
    if (!(rd instanceof MethodDecl) || Modifiers.isStatic(rd.modifiers)) {
      return MSTATUS_NEW_ROUTINE;
    }
    MethodDecl md = (MethodDecl)rd;

    Set direct = javafe.tc.PrepTypeDeclaration.getInst().getOverrides(md.parent, md);
    if (direct.size() == 0) {
      return MSTATUS_NEW_ROUTINE;
    }
    if (!(md.parent instanceof ClassDecl)) {
      return MSTATUS_OVERRIDE;
    }

    Enumeration e = direct.elements();
    while (e.hasMoreElements()) {
      MethodDecl directMD = (MethodDecl)(e.nextElement());
      if (directMD.parent instanceof ClassDecl) {
        return MSTATUS_OVERRIDE;
      }
    }
    return MSTATUS_CLASS_NEW_METHOD;
  }

  /**
   * @return null if method md is allowed to declare its jth (counting from 0)
   * formal parameter as non_null.  That is the case if the method does not
   * override anything, or if in everything that it does override that parameter is
   * declared non_null.  Otherwise returns the MethodDecl corresponding to the
   * overridden method with which the argument rd is in conflict.
   *
   * @todo kiniry - implement nullable versions of this or make two new methods
   */
  static public MethodDecl getSuperNonNullStatus(RoutineDecl rd, int j) {
    if (!(rd instanceof MethodDecl) || Modifiers.isStatic(rd.modifiers)) {
      return null;
    }
    MethodDecl md = (MethodDecl)rd;

    Set direct = javafe.tc.PrepTypeDeclaration.getInst().getOverrides(md.parent, md);
    if (direct.size() == 0) {
      return null;
    }
    return getSuperMethodDeclIfParamIsNullable(j,direct);
  }

  static public MethodDecl getSuperMethodDeclIfParamIsNullable(int j, Set directOverrides) {
    Enumeration e = directOverrides.elements();
    while (e.hasMoreElements()) {
      MethodDecl directMD = (MethodDecl)(e.nextElement());
      FormalParaDecl f = directMD.args.elementAt(j);
      if (!methodArgIsNonNull(f))
        return directMD;
    }
    return null;
  }

  public static boolean methodArgIsNonNull(FormalParaDecl arg) {
		ModifierPragma non_null_pragma = Utils.findModifierPragma(arg, TagConstants.NON_NULL);
		if (non_null_pragma != null)
			return true;
		ModifierPragma nullable_pragma = Utils.findModifierPragma(arg, TagConstants.NULLABLE);
		if (nullable_pragma != null)
			return false;
		return Main.options().nonNullByDefault;
	}

  /** Returns non-zero if the expression is a ghost expression - that is, it
      would not exist if all ghost declarations were removed.  Otherwise
      returns a Location value of a relevant non-ghost declaration.
  */
  public int isGhost(Expr t) {
    if (t instanceof ArrayRefExpr) {
      t = ((ArrayRefExpr)t).array;
    }
    if (t instanceof FieldAccess) {
      FieldAccess fa = (FieldAccess)t;
      if (fa.decl == null || GhostEnv.isGhostField(fa.decl))
        return 0;
      int gl = isGhost(fa.od);
      if (gl == 0) return 0;
      if (gl == -1) return fa.decl.getStartLoc();
      return gl;
    } else if (t instanceof VariableAccess) {
      VariableAccess va = (VariableAccess)t;
      GenericVarDecl gd = va.decl;
      if ( Utils.findModifierPragma(
                                    gd.pmodifiers,TagConstants.GHOST) == null)
        return gd.getStartLoc();
    } else if (t instanceof ParenExpr) {
      return isGhost( ((ParenExpr)t).expr );
    } else if (t instanceof CastExpr) {
      return isGhost( ((CastExpr)t).expr );
    }
    return 0;
    // FIXME - need some test that the expression in advance of
    // a . in a field reference or the [] in an array reference
    // only designates ghost fields/variables
    // e.g. what about method calls, operator expressions?

  }

  public int isGhost(ObjectDesignator od) {
    if (od instanceof ExprObjectDesignator) {
      Expr e = ((ExprObjectDesignator)od).expr;
      if (e == null || e instanceof ThisExpr) return -1;
      return isGhost(e);
    }
    return -1; // OK for TypeObjectDesignator and SuperObjectDesignator
  }

  protected boolean assignmentConvertable(Expr e, Type t) {
    if (super.assignmentConvertable(e,t)) return true;
    return Types.isTypeType(t) && Types.isTypeType(getType(e));
  }

  //alx: dw overridden, so javafe handels pure methods correctly
  //TODO: what other specs?
  protected boolean isPure( RoutineDecl rd) {
  	return Utils.isPure(rd);
  }
  //alx-end

} // end of class FlowInsensitiveChecks

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
