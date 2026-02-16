/* Copyright 2000, 2001, Compaq Computer Corporation */
/* Modified 2007, Systems Research Group, University College Dublin */

package escjava.translate;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Hashtable;
import java.util.LinkedList;
import java.util.ListIterator;

import javafe.ast.ASTNode;
import javafe.ast.AmbiguousVariableAccess;
import javafe.ast.ArrayRefExpr;
import javafe.ast.BinaryExpr;
import javafe.ast.CastExpr;
import javafe.ast.ClassLiteral;
import javafe.ast.CondExpr;
import javafe.ast.ConstructorDecl;
import javafe.ast.Expr;
import javafe.ast.ExprObjectDesignator;
import javafe.ast.ExprVec;
import javafe.ast.FieldAccess;
import javafe.ast.FieldDecl;
import javafe.ast.FormalParaDecl;
import javafe.ast.FormalParaDeclVec;
import javafe.ast.GenericVarDecl;
import javafe.ast.GenericVarDeclVec;
import javafe.ast.Identifier;
import javafe.ast.InstanceOfExpr;
import javafe.ast.LiteralExpr;
import javafe.ast.LocalVarDecl;
import javafe.ast.MethodDecl;
import javafe.ast.MethodInvocation;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.NewArrayExpr;
import javafe.ast.NewInstanceExpr;
import javafe.ast.ObjectDesignator;
import javafe.ast.ParenExpr;
import javafe.ast.RoutineDecl;
import javafe.ast.SuperObjectDesignator;
import javafe.ast.ThisExpr;
import javafe.ast.Type;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.UnaryExpr;
import javafe.ast.VarInit;
import javafe.ast.VariableAccess;
import javafe.tc.TypeCheck;
import javafe.tc.TypeSig;
import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.util.Location;
import escjava.AnnotationHandler;
import escjava.Main;
import escjava.ast.ExprModifierPragma;
import escjava.ast.GeneralizedQuantifiedExpr;
import escjava.ast.GuardExpr;
import escjava.ast.LabelExpr;
import escjava.ast.Modifiers;
import escjava.ast.NamedExprDeclPragma;
import escjava.ast.NaryExpr;
import escjava.ast.NotModifiedExpr;
import escjava.ast.QuantifiedExpr;
import escjava.ast.SimpleModifierPragma;
import escjava.ast.TagConstants;
import escjava.ast.TypeExpr;
import escjava.ast.Utils;
import escjava.ast.VarExprModifierPragma;
import escjava.parser.OldVarDecl;
import escjava.tc.Types;

/** Translates Annotation Expressions to GCExpr. */

public class TrAnExpr {
  
  public static final String UNABLE_TO_PARSE = "Unable to parse this expression - might not be valid syntax";

/** This hashtable keeps track of cautions issued, with respect to
   * using variables in \old pragmas that were not mentioned in modifies
   * pragmas.  The purpose of this hashtable is to prevent issuing
   * duplicate cautions.
   **/
  
  // NOT USED private static Set issuedPRECautions = new Set();  
  
  private static int newStringCount = 0;
  
  public static Translate translate = null;
  
  /** This is the abbreviated form of function <code>TrSpecExpr</code>
   * described in ESCJ 16.  It is a shorthand for the full form in which
   * <code>sp</code> and <code>st</code> are passed in as empty maps.
   **/
  
  public static Expr trSpecExpr(Expr e) {
    return trSpecExpr(e, null, null);
  }
  
  public static Expr trSpecExpr(Expr e, Expr thisExpr) {
    try {
      specialThisExpr = thisExpr;
      return trSpecExpr(e);
    } finally {
      specialThisExpr = null;
    }
  }
  
  // inits if not already inited
  public static void initForClause(boolean b) {
    if (!extraSpecs) initForClause();
  }
  public static void initForClause() {
    extraSpecs = true;
    trSpecExprAuxConditions = ExprVec.make();
    trSpecExprAuxAssumptions = ExprVec.make();
    trSpecAuxAxiomsNeeded = new java.util.HashSet();
    trSpecModelVarsUsed = new java.util.HashMap();
    boundStack.clear();
  }
  
  public static void closeForClause() {
    extraSpecs = false;
    trSpecExprAuxConditions = null;
    boundStack.clear();
  }
  
  public static void initForRoutine() {
    extraSpecs = false;
    trSpecExprAuxConditions = null;
    tempn = 100;
    declStack = new LinkedList();
    currentAlloc = GC.allocvar;
    currentState = GC.statevar;
    boundStack = new LinkedList();
    maxLevel = Main.options().rewriteDepth;
  }
  
  public static boolean doRewrites() {
    return extraSpecs;
  }
  
  public static int level = 0;
  public static int maxLevel = 3; // FIXME if this is much bigger the JML specs file java.math.BigInteger.parse runs out of memory
  public static boolean extraSpecs = false;
  public static ExprVec trSpecExprAuxConditions = null;
  public static ExprVec trSpecExprAuxAssumptions = null;
  public static java.util.Map trSpecModelVarsUsed = null;
  public static java.util.Set trSpecAuxAxiomsNeeded = null;
  public static int tempn = 100;
  public static LinkedList declStack = new LinkedList();
  public static VariableAccess currentAlloc = GC.allocvar;
  public static VariableAccess currentState = GC.statevar;
  public static LinkedList boundStack;
  
  /** This is the full form of function <code>TrSpecExpr</code>
   * described in ESCJ 16.  Each of the parameters <code>sp</code>
   * and <code>st</code> is allowed to be null, which is interpreted
   * as the empty map.
   **/
  
  protected static Expr specialResultExpr = null;
  protected static Expr specialThisExpr = null;
  
  protected static TrAnExpr inst = new TrAnExpr();
  
  public static Expr trSpecExpr(Expr e, Hashtable sp, Hashtable st, Expr thisExpr) {
    try {
      specialThisExpr = thisExpr;
      return trSpecExpr(e,sp,st);
    } finally {
      specialThisExpr = null;
    }
  }
  public static Expr trSpecExpr(Expr e, /*@ nullable */ Hashtable sp, /*@ nullable */ Hashtable st) {
    return inst.trSpecExprI(e,sp,st);
  }
  
  public Expr trSpecExprI(Expr e, /*@ nullable */ Hashtable sp, /*@ nullable */ Hashtable st) {
    int tag = e.getTag();
    switch (tag) {
      case TagConstants.THISEXPR: {
        ThisExpr t = (ThisExpr)e;
        Expr a;
        if (t.classPrefix != null) {
          a = trSpecExprI(Inner.unfoldThis(t), sp, st);
        } else if (specialThisExpr != null) {
          a = specialThisExpr;
        } else {
          a = apply(sp, makeVarAccess(GC.thisvar.decl, e.getStartLoc()));
        }
        return a;
        }
      
      // Literals (which are already GCExpr's
      case TagConstants.BOOLEANLIT: 
      case TagConstants.CHARLIT:
      case TagConstants.DOUBLELIT: 
      case TagConstants.FLOATLIT:
      case TagConstants.INTLIT:
      case TagConstants.LONGLIT:
      case TagConstants.NULLLIT:
        return e;

      case TagConstants.STRINGLIT:
        { String s = ((LiteralExpr)e).value.toString();
          Expr ee = GC.nary(TagConstants.INTERN,
                            GC.symlit(Translate.Strings.intern(s).toString()),
                            GC.symlit(Integer.toString(s.length())) );
	  return ee;
        }
      
      case TagConstants.RESEXPR:
        if (specialResultExpr != null) return specialResultExpr;
      return apply(sp, makeVarAccess(GC.resultvar.decl, e.getStartLoc()));
      
      case TagConstants.LOCKSETEXPR:
        return apply(sp, makeVarAccess(GC.LSvar.decl, e.getStartLoc()));
      
      case TagConstants.VARIABLEACCESS: {
        VariableAccess va = (VariableAccess)e;
        if (va.decl instanceof OldVarDecl) {
          // variable declared in an old pragma (not the \old fcn.)
          VarInit vi = ((OldVarDecl)va.decl).init;
          if (vi instanceof Expr) {
            return trSpecExpr( (Expr)vi, sp, st);
          } else {
            ErrorSet.fatal(e.getStartLoc(),
            "Have not implemented array initializers in old annotations");
          }
        } else {
          // local variable, parameter, or bound variable
          return apply(sp, va);
        }
      }
      
      case TagConstants.FIELDACCESS: {
        FieldAccess fa = (FieldAccess)e;
        VariableAccess va = makeVarAccess(fa.decl, fa.locId);
        boolean treatLikeAField = false;
        // va accesses the field
        if (Utils.isModel(va.decl.pmodifiers)) {
          // Treat a model variable like a field if (a) it has no
          // represents clauses and (b) it is not modified.
          
          ObjectDesignator od = fa.od;
          TypeSig ts = null;
          if (od == null) {
            System.out.println("OD NULL"); // FIXME
            // SHould use the TypeSIg of the class being translated
          } else {
            ts = (TypeSig)od.type();
          }
          TypeDeclElemVec reps = ts == null ? null : // FIXME
            GetSpec.getRepresentsClauses(
              null /*ts.getTypeDecl()*/, fa.decl);
          if (reps == null || reps.size() == 0) {
            //boolean b = translate.frameHandler.isDefinitelyNotAssignable(
            //    (od instanceof ExprObjectDesignator) ?
            //        ((ExprObjectDesignator)od).expr : null ,fa.decl);
            treatLikeAField = true;
          }
          //System.out.println("TREATLIKEAFIELD " + treatLikeAField + " " + doRewrites() + " " + fa + " " + Location.toString(fa.getStartLoc()) );
          //System.out.println("MODEL VAR " + fa.decl.id + " " + Location.toString(fa.locId));
          if (!treatLikeAField && doRewrites()) {
            trSpecAuxAxiomsNeeded.add(new RepHelper(fa));
            Identifier id = representsMethodName(va);
            if (Modifiers.isStatic(fa.decl.modifiers)) {
              return GC.nary(id,stateVar(sp)); // FIXME or should this be a function of the class object
            }
          } else { // Treat like a field - only good if the model
            // variable is not modified, since modifications
            // to model vars are not by assignment.
            
            /*
             java.util.List reps = escjava.AnnotationHandler.findRepresents(fa.decl);
             java.util.Iterator it = reps.iterator();
             while (it.hasNext()) {
             Expr ex = (Expr)it.next();
             
             if (doRewrites() && !declStack.contains(fa.decl)) {
             declStack.addFirst(fa.decl);
             if (trSpecModelVarsUsed != null &&
             !trSpecModelVarsUsed.containsKey(fa.decl) ) trSpecModelVarsUsed.put(fa.decl,va);
             
             Hashtable h = new Hashtable();
             if (fa.od instanceof ExprObjectDesignator) {
             if (!(((ExprObjectDesignator)fa.od).expr instanceof ThisExpr)) {
             h.put(Substitute.thisexpr, ((ExprObjectDesignator)fa.od).expr);
             }
             } else if (fa.od instanceof SuperObjectDesignator) {
             // FIXME
              System.out.println("NOT SUPPORTED-A " + fa.od.getClass());
              } // fall-through for TypeObjectDesignator
              
              ex = Substitute.doSubst(h,ex);
              Expr ee = trSpecExpr(ex,sp,st);
              trSpecExprAuxConditions.addElement(ee);
              declStack.removeFirst();
              }
              }
              */
          }
        }
        
        if (Modifiers.isStatic(fa.decl.modifiers)) {
          //fa.od.getTag() == TagConstants.TYPEOBJECTDESIGNATOR) 
          VariableAccess nva = apply(sp, va);
          return nva;
        } else {
          // select expression whose rhs is an instance variable
          Expr lhs;
          switch (fa.od.getTag()) {
            case TagConstants.EXPROBJECTDESIGNATOR: {
              ExprObjectDesignator eod = (ExprObjectDesignator)fa.od;
              lhs = trSpecExpr(eod.expr, sp, st);
              break;
            }
            
            case TagConstants.SUPEROBJECTDESIGNATOR:
              lhs = apply(sp, makeVarAccess(GC.thisvar.decl,
                  fa.od.getStartLoc()));
            break;
            
            default:
              //@ unreachable;
              Assert.fail("Fall thru; tag = "
                  + escjava.ast.TagConstants.toString(fa.od.getTag()));
            lhs = null; // dummy assignment
          }
          
          if (fa.decl == Types.lengthFieldDecl) {
            return GC.nary(fa.getStartLoc(), fa.getEndLoc(),
                TagConstants.ARRAYLENGTH, lhs);
          } else if (Utils.isModel(va.decl.pmodifiers) && !treatLikeAField && doRewrites()) {
            Identifier id = representsMethodName(va);
            ExprVec arg = ExprVec.make(2);
            arg.addElement(stateVar(sp));
            arg.addElement(lhs);
            return GC.nary(id, va.decl, arg);
          } else {
            return GC.nary(fa.getStartLoc(), fa.getEndLoc(),
                TagConstants.SELECT, apply(sp, va), lhs);
          }
        }
      }
      
      case TagConstants.ARRAYREFEXPR: {
        ArrayRefExpr r = (ArrayRefExpr)e;
        
        if (TypeCheck.inst.getType(r.array).getTag() == TagConstants.LOCKSET) {
          return GC.nary(r.array.getStartLoc(), r.locCloseBracket,
              TagConstants.SELECT,
              trSpecExpr(r.array, sp, st),
              trSpecExpr(r.index, sp, st));
        } else {
          VariableAccess elems = apply(sp, makeVarAccess(GC.elemsvar.decl,
              e.getStartLoc()));
          Expr e0 = trSpecExpr(r.array, sp, st);
          Expr e1 = trSpecExpr(r.index, sp, st);
          Expr a = GC.nary(TagConstants.SELECT, elems, e0);
          return GC.nary(r.array.getStartLoc(), r.locCloseBracket,
              TagConstants.SELECT, a, e1);
        }
      }
      
      case TagConstants.ARRAYRANGEREFEXPR:
      case TagConstants.WILDREFEXPR:
        return null;
      
      case TagConstants.PARENEXPR: {
        // drop parens
        ParenExpr pe = (ParenExpr)e;
        return trSpecExpr(pe.expr, sp, st);
      }
      
      // Unary operator expressions
      
      case TagConstants.UNARYSUB: 
      case TagConstants.NOT: 
      case TagConstants.BITNOT: {
        UnaryExpr ue = (UnaryExpr)e;
        int newtag = getGCTagForUnary(ue);
        return GC.nary(ue.getStartLoc(), ue.getEndLoc(), newtag, 
            trSpecExpr(ue.expr, sp, st));
      }
      
      case TagConstants.UNARYADD: {
        // drop UnaryADD
        UnaryExpr ue = (UnaryExpr)e;
        return trSpecExpr(ue.expr, sp, st);
      }
      
      case TagConstants.TYPEOF:
      case TagConstants.ELEMTYPE:
      case TagConstants.MAX:
      {
        Assert.notFalse(((NaryExpr)e).exprs.size() == 1);
        NaryExpr ne = (NaryExpr)e;
        int n = ne.exprs.size();
        ExprVec exprs = ExprVec.make(n);
        for (int i = 0; i < n; i++) {
          exprs.addElement(trSpecExpr(ne.exprs.elementAt(i), sp, st));
        }
        return GC.nary(ne.getStartLoc(), ne.getEndLoc(), ne.getTag(), exprs);
      }

      case TagConstants.DTTFSA: {
        NaryExpr ne = (NaryExpr)e;
        int n = ne.exprs.size();
        ExprVec exprs = ExprVec.make(n);
        if (n>0) exprs.addElement(trSpecExpr(ne.exprs.elementAt(0), sp, st));
        if (n>1) exprs.addElement(ne.exprs.elementAt(1)); // This is a String - don't want to intern it
        for (int i = 2; i < n; i++) {
          exprs.addElement(trSpecExpr(ne.exprs.elementAt(i), sp, st));
        }
        return GC.nary(ne.getStartLoc(), ne.getEndLoc(), ne.getTag(), exprs);
      }
      
      case TagConstants.ELEMSNONNULL: {
        NaryExpr ne = (NaryExpr)e;
        VariableAccess elems = apply(sp, makeVarAccess(GC.elemsvar.decl,
            e.getStartLoc()));
        
        return GC.nary(ne.getStartLoc(), ne.getEndLoc(), ne.getTag(),
            trSpecExpr(ne.exprs.elementAt(0), sp, st), elems);
      }
      
      // Binary operator expressions
      
      case TagConstants.OR:
      case TagConstants.AND:
      case TagConstants.IMPLIES:
      case TagConstants.IFF:
      case TagConstants.NIFF:
      case TagConstants.BITOR:
      case TagConstants.BITAND:
      case TagConstants.BITXOR:
      {
        BinaryExpr be = (BinaryExpr)e;
        int newtag = getGCTagForBinary(be);
        return GC.nary(be.getStartLoc(), be.getEndLoc(),
            newtag,
            trSpecExpr(be.left, sp, st),
            trSpecExpr(be.right, sp, st));
      }
      
      case TagConstants.EQ:
      case TagConstants.NE:
      case TagConstants.LSHIFT:
      case TagConstants.RSHIFT:
      case TagConstants.URSHIFT:
      { // FIXME - do these have any type matching to do?
        BinaryExpr be = (BinaryExpr)e;
        int newtag = getGCTagForBinary(be);
        return GC.nary(be.getStartLoc(), be.getEndLoc(),
            newtag,
            trSpecExpr(be.left, sp, st),
            trSpecExpr(be.right, sp, st));
      }
      
      case TagConstants.GE:
      case TagConstants.GT:
      case TagConstants.LE:
      case TagConstants.LT:
      case TagConstants.ADD:
      case TagConstants.SUB:
      case TagConstants.DIV:
      case TagConstants.MOD:
      case TagConstants.STAR:
      {
        // Must do appropriate primitive type casting to make arguments the same type
        // FIXME - what about + on strings?
        BinaryExpr be = (BinaryExpr)e;
        //EscPrettyPrint.inst.print(System.out,0,be);
        int newtag = getGCTagForBinary(be);
        Expr left = trSpecExpr(be.left, sp, st);
        Expr right = trSpecExpr(be.right, sp, st);
        /*
         int leftType = ((PrimitiveType)TypeCheck.inst.getType(be.left)).getTag();
         int rightType = ((PrimitiveType)TypeCheck.inst.getType(be.right)).getTag();
         // FIXME - do we need to do any promotion ?
          if (leftType != rightType && false) {
          System.out.println("TYPES " + TagConstants.toString(newtag) + " " + TagConstants.toString(leftType) + " " + TagConstants.toString(rightType));
          }
          */
        if (newtag == TagConstants.STRINGCAT) {
          ExprVec ev = ExprVec.make(3);
          ev.addElement(left);
          ev.addElement(right);
          ev.addElement(GC.nary(Identifier.intern("next"),
              GC.allocvar, LiteralExpr.make(TagConstants.INTLIT,new Integer(++newStringCount),Location.NULL)));
          return GC.nary(be.getStartLoc(), be.getEndLoc(),
              newtag, ev);
        } else {
          return GC.nary(be.getStartLoc(), be.getEndLoc(),
              newtag, left, right);
        }
      }
      
      case TagConstants.NEWINSTANCEEXPR: {
        NewInstanceExpr me = (NewInstanceExpr)e;
        Type type = TypeCheck.inst.getType(me);
        Expr v;
        VariableAccess vv = tempName(e.getStartLoc(),"tempNewObject",type);
        if (boundStack.size() == 0) {
          v = vv;
        } else {
          ExprVec ev = ExprVec.make();
          java.util.Iterator ii = boundStack.iterator();
          while (ii.hasNext()) {
            Object o = ((QuantifiedExpr)ii.next()).vars;
            if (o instanceof GenericVarDecl) {
              GenericVarDecl g = (GenericVarDecl)o;
              ev.addElement( VariableAccess.make(g.id, g.getStartLoc(), g) );
            } else if (o instanceof GenericVarDeclVec) {
              GenericVarDeclVec gi = (GenericVarDeclVec)o;
              for (int kk = 0; kk<gi.size(); ++kk) {
                GenericVarDecl g = gi.elementAt(kk);
                ev.addElement( VariableAccess.make(g.id, g.getStartLoc(), g) );
              }
            } else System.out.print("[[" + o.getClass() + "]]");
          }
          v = GC.nary(vv.id, vv.decl, ev);
          //v = MethodInvocation.make(ExprObjectDesignator.make(Location.NULL,ThisExpr.make(null,Location.NULL)),vv.id,null, Location.NULL, Location.NULL, ev);
        }
        boolean genSimpleVar = false;
        boolean genFunctionCallAndAxioms = false;
        boolean genVarAndConditions = false;
        boolean isFunction = Utils.isFunction(me.decl);
        if ((isFunction||true) && doRewrites()) {
          genFunctionCallAndAxioms = true;
        } else if ( !doRewrites()
            || level > maxLevel 
            || declStack.contains(me.decl)
        ) {
          genSimpleVar = true;
        } else {
          genVarAndConditions = true;
        }
        
        if (genSimpleVar) {
          return v;
        } else if (genVarAndConditions) {
          ++level;
          declStack.addFirst(me.decl);
          // v is the variable that is the result of the constructor call 
          // Adds v != null
          Expr ee = GC.nary(TagConstants.REFNE, v, 
              LiteralExpr.make(TagConstants.NULLLIT, null, Location.NULL));
          trSpecExprAuxConditions.addElement(ee);
          // Adds \typeof(v) == \type(...)
          ee = GC.nary(TagConstants.TYPEEQ,
              GC.nary(TagConstants.TYPEOF, v),
              TypeExpr.make(Location.NULL, Location.NULL, type));
          trSpecExprAuxConditions.addElement(ee);
          // Adds alloc < newalloc
          VariableAccess newAlloc =
            apply(sp,GC.makeVar(GC.allocvar.id,e.getStartLoc())); // alloc
          trSpecExprAuxAssumptions.addElement(
              GC.nary(TagConstants.ALLOCLT, currentAlloc, newAlloc));
          // Adds ! isAllocated(v, alloc)
          trSpecExprAuxConditions.addElement(
              GC.nary(TagConstants.BOOLNOT,
                  GC.nary(TagConstants.ISALLOCATED, v, currentAlloc)));
          // Adds isAllocated(v, newalloc)
          trSpecExprAuxConditions.addElement(
              GC.nary(TagConstants.ISALLOCATED, v, newAlloc));
          currentAlloc = newAlloc;
          // Go add all the specs generated by substituting v for \result
          // in the specs of the constructor
          getCalledSpecs(me.decl,null,me.args,v,sp,st); // adds to trSpecExprAuxConditions
          //System.out.println("END-NEWINST " + Location.toString(me.decl.getStartLoc()) + " " + declStack.contains(me.decl));
          declStack.removeFirst();
          --level;
          return v;
        } else if (genFunctionCallAndAxioms) {
          
          trSpecAuxAxiomsNeeded.add(new RepHelper(me.decl.parent,me.decl));
          Identifier constructorname = fullName(me.decl,false);
          VariableAccess newAlloc =
            apply(sp,GC.makeVar(GC.allocvar.id,e.getStartLoc())); // alloc
          ExprVec ev = ExprVec.make(me.args.size()+4);
          // FIXME - enclosingInstance ???
          if (!isFunction) {
            ev.addElement( stateVar(sp) );
          }
          ev.addElement(currentAlloc);
          ev.addElement(newAlloc);
          for (int i=0; i<me.args.size(); ++i) {
            ev.addElement( trSpecExpr( me.args.elementAt(i), sp, st));
          }
          Expr ne = GC.nary(me.getStartLoc(), me.getEndLoc(),
              constructorname,ev);
          ((NaryExpr)ne).symbol = me.decl;
          // Adds alloc < newalloc
          trSpecExprAuxAssumptions.addElement(
              GC.nary(TagConstants.ALLOCLT, currentAlloc, newAlloc));
          currentAlloc = newAlloc;
          return ne;
        }
      }
      
      case TagConstants.METHODINVOCATION: {
        /* We can handle a method invocation in a spec expression in two ways.
         a) We can turn the method invocation into a function call within the target
         logic.  The we add axioms for that function call corresponding to the 
         postconditions of the method in the Java program.  We add the implicit this
         parameter as an argument of the function if the method is not static.
         The difficulty is that not all methods are functions; functions have equal
         results if their arguments are equal.  Methods don't necessarily satisfy this
         because their arguments have structure that might change without the object
         identity changing.  Having immutable objects helps.
         b) Alternatively we define a new constant corresponding to the result of the
         method invocation. [If the method invocation is within the scope of 
         quantifiers, then we have to define a new function with the appropriate 
         arguments.] Then we add an assumption that the constant satisfies the 
         postconditions of the method (with \result replaced by the new constant).
         The difficulties are that we need a new constant for each method invocations
         and that we have to limit the depth since method postconditions can contain
         more method calls.
         */
        
        MethodInvocation me = (MethodInvocation)e;
        
        // FIXME - when is me.decl null ? When it is a built-in method like
	// \reach().has()

        boolean isFunction = me.decl == null ? true : Utils.isFunction(me.decl);
        // The above result will be true if the method is declared to be a function
        // or if it has only immutable arguments.
        
        Expr v;
        VariableAccess vv = tempName(e.getStartLoc(),"tempMethodReturn",
            TypeCheck.inst.getType(me));
        if (boundStack.size() == 0) {
          v = vv;
        } else {
          ExprVec ev = ExprVec.make();
          java.util.Iterator ii = boundStack.iterator();
          while (ii.hasNext()) {
            Object o = ((QuantifiedExpr)ii.next()).vars;
            if (o instanceof GenericVarDecl) {
              GenericVarDecl g = (GenericVarDecl)o;
              ev.addElement( VariableAccess.make(g.id, g.getStartLoc(), g) );
            } else if (o instanceof GenericVarDeclVec) {
              GenericVarDeclVec gi = (GenericVarDeclVec)o;
              for (int kk = 0; kk<gi.size(); ++kk) {
                GenericVarDecl g = gi.elementAt(kk);
                ev.addElement( VariableAccess.make(g.id, g.getStartLoc(), g) );
              }
            } else System.out.print("[[" + o.getClass() + "]]");  // FIXME
          }
          v = GC.nary(vv.id, vv.decl, ev);
        }
        /*
         System.out.print("METHOD " + me.decl.parent.id + " " + me.decl.id + " " );
         EscPrettyPrint.inst.print(System.out,0,me.od);
         System.out.println("");
         */
        
        boolean genSimpleVar = false;
        boolean genFunctionCallAndAxioms = false;
        boolean genVarAndConditions = false;
        if (isFunction || true) {
          genFunctionCallAndAxioms = true;
        } else if (!doRewrites()
            || level > maxLevel 
            || declStack.contains(me.decl)
        ) {
          genSimpleVar = true;
        } else {
          genVarAndConditions = true;
        }
        
        if (genSimpleVar) {
          // Just replace the method invocation with a simple new variable
          // We won't be able to reason about it because it will be unique.
          return v;
        } else if (genVarAndConditions) {
          ++level;
          declStack.addFirst(me.decl);
          ExprVec ev = ExprVec.make(me.args.size());
          if (!Modifiers.isStatic(me.decl.modifiers)) {
            if (me.od instanceof ExprObjectDesignator) {
              Expr ex = ((ExprObjectDesignator)me.od).expr;
              ev.addElement( trSpecExpr( ex, sp, st));
            } else {
              // FIXME
              System.out.println("NOT SUPPORTED-B " + me.od.getClass());
            } 
          }
          
          getCalledSpecs(me.decl,me.od,me.args,v,sp,st); // adds to trSpecExprAuxConditions
          --level;
          declStack.removeFirst();
          return v;
        } else if (genFunctionCallAndAxioms) {
          if (doRewrites()) {
            trSpecAuxAxiomsNeeded.add(new RepHelper(me.decl.parent,me.decl));
          }
          ExprVec ev = ExprVec.make(me.args.size()+1);
          boolean useSuper = false;
          if (!isFunction) {
            ev.addElement( stateVar(sp) );
          }
          if (!Modifiers.isStatic(me.decl.modifiers)) {
            if (me.od instanceof ExprObjectDesignator) {
              Expr ex = ((ExprObjectDesignator)me.od).expr;
              ev.addElement( trSpecExpr( ex, sp, st));
            } else {
              useSuper = true;
              ev.addElement( trSpecExpr(javafe.ast.ThisExpr.make(null,me.od.getStartLoc()),sp,st));
              // FIXME
              //System.out.println("NOT SUPPORTED-C " + me.od.getClass());
            } 
          }
          for (int i=0; i<me.args.size(); ++i) {
            ev.addElement( trSpecExpr( me.args.elementAt(i), sp, st));
          }
          Expr ne = GC.nary(me.getStartLoc(), me.getEndLoc(),
              TagConstants.METHODCALL,ev);
          ((NaryExpr)ne).symbol = me.decl;
          ((NaryExpr)ne).methodName = fullName(me.decl,useSuper); 
          return ne;
        }
      }
      
      case TagConstants.NEWARRAYEXPR: {
        NewArrayExpr nae = (NewArrayExpr)e;
        if (doRewrites() ) {
          ExprVec ev = ExprVec.make(5);
          ev.addElement(apply(sp,currentAlloc) ); // current alloc
          VariableAccess newAlloc = GC.makeVar(GC.allocvar.id,nae.getStartLoc());
          ev.addElement(apply(sp,newAlloc) ); // new alloc value
          trSpecExprAuxAssumptions.addElement(
              GC.nary(TagConstants.ALLOCLT, currentAlloc, newAlloc));
          currentAlloc = newAlloc;
          Expr edims = GC.nary( TagConstants.ARRAYSHAPEONE, 
              trSpecExpr( nae.dims.elementAt(0), sp, st));
          for (int kk = 1; kk < nae.dims.size(); ++kk) {
            edims = GC.nary( TagConstants.ARRAYSHAPEMORE,
                trSpecExpr( nae.dims.elementAt(kk), sp, st), edims);
          }
          ev.addElement(edims ); // arrayShape
          Type t = TypeCheck.inst.getType(nae);
          ev.addElement( TypeExpr.make(Location.NULL, Location.NULL, t) );
          ev.addElement(Types.zeroEquivalent(Types.baseType(t))); // initial value
          return GC.nary(TagConstants.ARRAYMAKE,ev);
        } else {
          ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,
              e.getStartLoc(),"Not checking predicates containing new array expressions");
          return null;
        }
      }
      
      case TagConstants.EXPLIES: {
        // handle as implies, but with arguments reversed
        BinaryExpr be = (BinaryExpr)e;
        return GC.nary(be.getStartLoc(), be.getEndLoc(),
            TagConstants.BOOLIMPLIES,
            trSpecExpr(be.right, sp, st),
            trSpecExpr(be.left, sp, st));
      }
      
      case TagConstants.SUBTYPE: {
        BinaryExpr be = (BinaryExpr)e;
        return GC.nary(be.getStartLoc(), be.getEndLoc(),
            TagConstants.TYPELE,
            trSpecExpr(be.left, sp, st),
            trSpecExpr(be.right, sp, st));
      }
      
      // Other expressions
      
      case TagConstants.CONDEXPR: {
        CondExpr ce = (CondExpr)e;
        return GC.nary(ce.test.getStartLoc(), ce.test.getEndLoc(),
            TagConstants.CONDITIONAL,
            trSpecExpr(ce.test, sp, st),
            trSpecExpr(ce.thn, sp, st),
            trSpecExpr(ce.els, sp, st));
      }
      
      case TagConstants.INSTANCEOFEXPR: {
        InstanceOfExpr ie = (InstanceOfExpr)e;
        Expr isOfType = GC.nary(ie.getStartLoc(), ie.getEndLoc(), 
            TagConstants.IS,
            trSpecExpr(ie.expr, sp, st),
            trType(ie.type));
        Expr notNull = GC.nary(ie.getStartLoc(), ie.getEndLoc(), 
            TagConstants.REFNE,
            trSpecExpr(ie.expr, sp, st),
            GC.nulllit);
        
        return GC.and(ie.getStartLoc(), ie.getEndLoc(),
            isOfType,
            notNull);
      }
      
      case TagConstants.CASTEXPR: {
        CastExpr ce = (CastExpr)e;
        return GC.nary(ce.getStartLoc(), ce.getEndLoc(), TagConstants.CAST,
            trSpecExpr(ce.expr, sp, st),
            trType(ce.type));
      }
      
      case TagConstants.CLASSLITERAL: {
        ClassLiteral cl = (ClassLiteral)e;
        return GC.nary(cl.getStartLoc(), cl.getEndLoc(),
            TagConstants.CLASSLITERALFUNC,
            trType(cl.type));
      }
      
      case TagConstants.TYPEEXPR:
        return e;
      
      case TagConstants.REACH: {
        ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,
            e.getStartLoc(),"Not checking predicates containing reach expressions");
        // FIXME - ignore these till we can figure out how to reason
        return null;
      }
      
      case TagConstants.NUM_OF:
      case TagConstants.SUM:
      case TagConstants.PRODUCT:
      {
        //ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,
        //e.getStartLoc(),"Not checking predicates containing generalized quantifier expressions");
        // FIXME - ignore these till we can figure out how to reason
        // Not sure this is the correct type ??? FIXME
        Type t = TypeCheck.inst.getType(e);
        VariableAccess va = tempName(e.getStartLoc(),"quantvalue",t);
        return va;
      }
      
      case TagConstants.MIN:
      case TagConstants.MAXQUANT:
      {
        Type t = TypeCheck.inst.getType(e);
        VariableAccess va = tempName(e.getStartLoc(),"quantvalue",t);
	ExprVec args = ExprVec.make();
        args.addElement(GC.statevar);
        if (boundStack.size() > 0) {
          ListIterator iter = boundStack.listIterator(boundStack.size());
          while (iter.hasPrevious()) {
            QuantifiedExpr qqe = (QuantifiedExpr)iter.previous();
            Object o = qqe.vars;
            if (o instanceof GenericVarDecl) {
              GenericVarDecl g = (GenericVarDecl)o;
              args.addElement(VariableAccess.make(g.id,Location.NULL,g));
            } else if (o instanceof GenericVarDeclVec) {
              GenericVarDeclVec gi = (GenericVarDeclVec)o;
              for (int i=0; i<gi.size(); ++i) {
                GenericVarDecl g = gi.elementAt(i);
                args.addElement(VariableAccess.make(g.id,Location.NULL,g));
              }
            }
          }
        }
        Expr vaf = GC.nary(va.id, va.decl, args);
        GeneralizedQuantifiedExpr qe = (GeneralizedQuantifiedExpr)e;
        Expr tex = trSpecExpr(qe.expr,sp,st);
        Expr rex = trSpecExpr(qe.rangeExpr,sp,st);
          GenericVarDeclVec dummyDecls = GenericVarDeclVec.make();
          Expr goodTypes = rex;
            for (int k=0; k<qe.vars.size(); ++k) {
              GenericVarDecl decl = qe.vars.elementAt(k);
              Assert.notFalse(sp == null || ! sp.contains(decl));
              Assert.notFalse(st == null || ! st.contains(decl));
              dummyDecls.addElement(decl);
              
              goodTypes = GC.and(goodTypes, quantTypeCorrect(decl, sp));
            }
        Expr ne = GC.implies(
            GC.quantifiedExpr(Location.NULL, Location.NULL,
                TagConstants.EXISTS, qe.vars, null, GC.and(goodTypes, rex), null, null),
                GC.quantifiedExpr(Location.NULL, Location.NULL,
                    TagConstants.EXISTS, qe.vars, null,
                    GC.and(goodTypes, GC.nary(TagConstants.INTEGRALEQ, vaf, tex)),
                    null, null));
        ExprVec pats = ExprVec.make();
        pats.addElement(vaf);
        if (boundStack.size() > 0) {
          // FIXME - the following only works for one level of bound variable
          // because of the pats - need to combine them all or maybe just put
          // the pats at the innermost level
          ListIterator iter = boundStack.listIterator(boundStack.size());
          ExprVec ppats = pats;
          while (iter.hasPrevious()) {
            QuantifiedExpr qqe = (QuantifiedExpr)iter.previous();
            Expr rrex = qqe.rangeExpr == null ? GC.truelit :
                               trSpecExpr(qqe.rangeExpr, sp, st);
            Object o = qqe.vars;
            if (o instanceof GenericVarDecl) {
              GenericVarDecl g = (GenericVarDecl)o;
              ne = GC.forallwithpats(g,GC.implies(rrex,ne),ppats);
              ppats = null;
            } else if (o instanceof GenericVarDeclVec) {
              GenericVarDeclVec gi = (GenericVarDeclVec)o;
               int kk = gi.size();
               while (--kk >= 0) {
                 GenericVarDecl g = gi.elementAt(kk);
                 ne = GC.forallwithpats(g,GC.implies(rrex,ne),ppats);
                 ppats = null;
               }
            } else System.out.print("[[" + o.getClass() + "]]");
          }
        }
        trSpecExprAuxAssumptions.addElement(ne);
        if (tag == TagConstants.MIN) {
          // (\min vars; rangeexpr; expr) generates the axioms
          // (\forall vars; rangeexpr ==> va <= expr) and
          // (\exists vars; rangeexpr) ==> (\exists vars; rangeexpr && va == expr)
          ne = QuantifiedExpr.make(Location.NULL, Location.NULL, 
              TagConstants.FORALL, qe.vars, null,
              GC.implies(goodTypes, GC.nary(TagConstants.INTEGRALLE, vaf, tex)),
              null,null);  // Use pats? FIXME
        } else {
          // (\max vars; rangeexpr; expr) generates the axioms
          // (\forall vars; rangeexpr ==> va >= expr) and
          // (\exists vars; rangeexpr) ==> (\exists vars; rangeexpr && va == expr)
          ne = QuantifiedExpr.make(Location.NULL, Location.NULL, 
              TagConstants.FORALL, qe.vars, null,
              GC.implies(goodTypes, GC.nary(TagConstants.INTEGRALGE, vaf, tex)),
              null, null); // FIXME Use Pats?
        }
        if (boundStack.size() > 0) {
          ListIterator iter = boundStack.listIterator(boundStack.size());
          while (iter.hasPrevious()) {
            QuantifiedExpr qqe = (QuantifiedExpr)iter.previous();
            Expr rrex = qqe.rangeExpr == null ? GC.truelit :
                              trSpecExpr(qqe.rangeExpr, sp, st);
            Object o = qqe.vars;
            if (o instanceof GenericVarDecl) {
              GenericVarDecl g = (GenericVarDecl)o;
              ne = GC.forall(g,rrex,GC.implies(rrex,ne));
            } else if (o instanceof GenericVarDeclVec) {
              GenericVarDeclVec gi = (GenericVarDeclVec)o;
              ne = GC.forall(gi,rrex,GC.implies(rrex,ne));
              /*
               int kk = gi.size();
               while (--kk >= 0) {
               GenericVarDecl g = gi.elementAt(kk);
               ne = GC.forall(g,rrex,GC.implies(rrex,ne));
               }
               */
            } else System.out.print("[[" + o.getClass() + "]]");
          }
        }
        trSpecExprAuxAssumptions.addElement(ne);
        return vaf;
      }
      
      case TagConstants.FORALL:
      case TagConstants.EXISTS: {
        QuantifiedExpr qe = (QuantifiedExpr)e;
          int op;
          if (qe.getTag() == TagConstants.FORALL)
            op = TagConstants.BOOLIMPLIES;
          else
            op = TagConstants.BOOLAND;
          
        if (doRewrites()) boundStack.add(qe);
        //if (doRewrites()) System.out.println("FORALL " + Location.toString(e.getStartLoc()));
        //if (doRewrites()) ErrorSet.dump(null);
	// FIXME - this needs to be integrated with the following two cases???
        if (qe.vars.size() != 1) {
          int locStart = e.getStartLoc();
          int locEnd = e.getEndLoc();
          
          GenericVarDeclVec dummyDecls = GenericVarDeclVec.make();
          Expr goodTypes = GC.truelit;
            for (int k=0; k<qe.vars.size(); ++k) {
              GenericVarDecl decl = qe.vars.elementAt(k);
              Assert.notFalse(sp == null || ! sp.contains(decl));
              Assert.notFalse(st == null || ! st.contains(decl));
              dummyDecls.addElement(decl);
              
              goodTypes = GC.and(goodTypes, quantTypeCorrect(decl, sp));
            }
          int pos = 0;
          if (doRewrites()) pos = trSpecExprAuxConditions.size();
          Expr body = trSpecExpr(qe.expr, sp, st);
          //if (doRewrites()) System.out.println("FORALL " + pos + " " + trSpecExprAuxConditions.size());
          if (doRewrites() && pos != trSpecExprAuxConditions.size() ) {
            ExprVec ev;
            if (pos == 0) {
              ev = trSpecExprAuxConditions;
              trSpecExprAuxConditions = ExprVec.make();
            } else {
              int sz = trSpecExprAuxConditions.size();
              ev = ExprVec.make(sz - pos);
              for (int i=pos; i < sz; ++i) {
                ev.addElement(trSpecExprAuxConditions.elementAt(i));
              }
              for (int i=pos; i<sz; ++i) {
                trSpecExprAuxConditions.removeElementAt(sz+pos-i-1);
              }
            }
            body = GC.andx( GC.nary(TagConstants.BOOLAND,ev), body);
          }
          if (doRewrites()) boundStack.removeLast();
          /*
           if (doRewrites()) {
           System.out.println("FORALL-ENDA " + Location.toString(e.getStartLoc()));
           EscPrettyPrint.inst.print(System.out,0,body);
           System.out.println("");
           }
           */
	  Expr qbody = GC.nary(locStart, locEnd, op, goodTypes, body);
	// FIXME - here and nearby - the rangeExpr in the make call is not
        // used later - is it ever not null coming into here?
          return GC.quantifiedExpr(locStart, locEnd, tag,
              qe.vars, 
              qe.rangeExpr == null ? goodTypes :
                trSpecExpr(qe.rangeExpr, sp, st),
                qbody, null, null);
        } else if (Main.options().nestQuantifiers) { // default is false
          GenericVarDecl decl = qe.vars.elementAt(0);
          
          Assert.notFalse(sp == null || ! sp.contains(decl));
          Assert.notFalse(st == null || ! st.contains(decl));
          Assert.notFalse( qe.vars.size() == 1 );
          
          Expr body = GC.nary(qe.getStartLoc(), qe.getEndLoc(), op,
              quantTypeCorrect(decl, sp),
              trSpecExpr(qe.expr, sp, st));
          if (doRewrites()) boundStack.removeLast();
          //if (doRewrites()) System.out.println("FORALL-ENDB " + Location.toString(e.getStartLoc()));
          return GC.quantifiedExpr(qe.getStartLoc(), qe.getEndLoc(),
              qe.getTag(),
              decl, 
              qe.rangeExpr == null ? null :
                trSpecExpr(qe.rangeExpr, sp, st),
                body, null, null);
        } else {
          // FIXME - need to handle AuxConditions in here
          int locStart = e.getStartLoc();
          int locEnd = e.getEndLoc();
          
          GenericVarDeclVec dummyDecls = GenericVarDeclVec.make();
          Expr goodTypes = GC.truelit;
          while (true) {
            for (int k=0; k<qe.vars.size(); ++k) {
              GenericVarDecl decl = qe.vars.elementAt(k);
              Assert.notFalse(sp == null || ! sp.contains(decl));
              Assert.notFalse(st == null || ! st.contains(decl));
              dummyDecls.addElement(decl);
              
              goodTypes = GC.and(goodTypes, quantTypeCorrect(decl, sp));
            }
            if (qe.expr.getTag() == tag) {
              qe = (QuantifiedExpr)qe.expr;
            } else {
              Expr body = trSpecExpr(qe.expr, sp, st);
              Expr qbody = GC.nary(locStart, locEnd, op, goodTypes, body);
              if (doRewrites()) boundStack.removeLast();
              //if (doRewrites()) System.out.println("FORALL-ENDC " + Location.toString(e.getStartLoc()));
              return GC.quantifiedExpr(locStart, locEnd, tag,
                  dummyDecls, 
                  qe.rangeExpr == null ? null :
                    trSpecExpr(qe.rangeExpr, sp, st),
                    qbody, null, null);
            }
          }
        }
        // unreachable;
      }
      
      case TagConstants.SETCOMPEXPR: {
        ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,
            e.getStartLoc(),"Not checking predicates containing set comprehension expressions");
        return null;
      }
      
      case TagConstants.LABELEXPR: {
        LabelExpr le = (LabelExpr)e;
        return LabelExpr.make(le.getStartLoc(), le.getEndLoc(),
            le.positive, le.label,
            trSpecExpr(le.expr, sp, st));
      }
      
      case TagConstants.PRE: {
        //
        // Original code altered to generate a caution if the variables in
        // the PRE clause are not mentioned in an appropriate "modifies"
        // clause.  Caroline Tice, 1/11/2000
        // Note: Current implementation assumes that if ANY substitution 
        //       takes place, then ALL appropriate substitutions took place.
        
        NaryExpr ne = (NaryExpr)e;
        
        /*  Compare the number of substituted variables before and
         *  after translating the PRE expression, using only the st
         *  list (i.e. the "modifies" list) for the translations.  If
         *  no substitutions took place, generate a caution.
         *  Then translate the expression previously generated, using
         *  the union of sp and st.
         */
        
        /* FIXME - disable this for now.  We use \old in AnnotationHandler when we
         are desugaring annotations, to wrap around a requires predicate when it is
         being combined with an ensures predicate.  This error would have us only
         wrap those variables being modified and not everything.
         int cReplacementsBefore = getReplacementCount();
         Expr tmpExpr = trSpecExpr(ne.exprs.elementAt(0), st, null);
         if (cReplacementsBefore == getReplacementCount()) {
         int loc = ne.getStartLoc();
         String locStr = Location.toString(loc).intern();
         if (!(issuedPRECautions.contains(locStr))) {
         ErrorSet.caution (loc, 
         "Variables in \\old not mentioned in modifies pragma.");
         issuedPRECautions.add(locStr);
         }
         }
         //return trSpecExpr(ne.exprs.elementAt(0), union(sp, st), null);
          */
        if (st != null) {
          return trSpecExpr(ne.exprs.elementAt(0), st, null);
        } else {
          return trSpecExpr(ne.exprs.elementAt(0), sp, st);
        }
      }
      
      case TagConstants.FRESH: {
        NaryExpr ne = (NaryExpr)e;
        int sloc = ne.getStartLoc();
        int eloc = ne.getEndLoc();
        int n = ne.exprs.size();
        ExprVec ev = ExprVec.make(n);
        for (int i=0; i<ne.exprs.size(); ++i) {
          Expr arg = trSpecExpr(ne.exprs.elementAt(i), sp, st);
          // arg != null
          Expr nonnull = GC.nary(sloc, eloc,
            TagConstants.REFNE, arg, GC.nulllit);
          // isAllocated(arg, alloc@pre)
          Expr isAlloced = GC.nary(sloc, eloc, TagConstants.ISALLOCATED,
            arg, apply(st, GC.allocvar));
          // !isAllocated(arg, alloc@pre)
          Expr newlyallocated = GC.not(sloc, eloc, isAlloced);
          ev.addElement(GC.and(sloc, eloc, nonnull, newlyallocated));
        }
        return GC.and(ev);
      }
      
      case TagConstants.DOTDOT:
        BinaryExpr be = (BinaryExpr)e;
      // FIXME
      return be.left;
      
      case TagConstants.NOWARN_OP:
      case TagConstants.WACK_NOWARN:
      case TagConstants.WARN_OP:
      case TagConstants.WARN:
      {
        // FIXME - set these as a pass through for now
        NaryExpr ne = (NaryExpr)e;
        return trSpecExpr(ne.exprs.elementAt(0), sp, st);
      }
      
      case TagConstants.IS_INITIALIZED:
      case TagConstants.INVARIANT_FOR:{
        // We return a fresh temporary variable, unused elsewhere, until
        // we know how to determine some conditions that these functions
        // satisfy.  FIXME
        Identifier n = Identifier.intern("tempInit"+(++tempn));
        VariableAccess v =  VariableAccess.make(n, e.getStartLoc(), 
            LocalVarDecl.make(Modifiers.NONE, null,n,
                Types.booleanType,
                UniqName.temporaryVariable,
                null, Location.NULL));
        return v;
      } 
      
      case TagConstants.SPACE:
      case TagConstants.WACK_WORKING_SPACE:
      case TagConstants.WACK_DURATION: {
        // We return a fresh temporary variable, unused elsewhere, until
        // we know how to determine some conditions that these functions
        // satisfy.  FIXME
        return tempName(e.getStartLoc(),"tempResources",Types.longType);
      }
      
      case TagConstants.NOTHINGEXPR:
      case TagConstants.EVERYTHINGEXPR:
        return null;
      
      case TagConstants.NOTMODIFIEDEXPR: {
        Expr ee = ((NotModifiedExpr)e).expr;
        Expr post = trSpecExpr(ee,sp,st);
        Expr pre;
        if (st == null) {
          pre = trSpecExpr(ee,sp,st);
        } else {
          pre = trSpecExpr(ee,st,null);
        }
        
        Type t = TypeCheck.inst.getType(ee);
        int ftag = TagConstants.ANYEQ;
        if (Types.isBooleanType(t)) ftag = TagConstants.BOOLEQ;
        
        return LabelExpr.make(ee.getStartLoc(),ee.getEndLoc(),
            false,GC.makeLabel("AdditionalInfo",ee.getStartLoc(),Location.NULL),GC.nary(ftag,post,pre));
      }
      
      default:
        Assert.fail("UnknownTag<"+e.getTag()+","+
            TagConstants.toString(e.getTag())+"> on "+e+ " " +
            Location.toString(e.getStartLoc()));
      return null; // dummy return
    }
  }
  
  static VariableAccess tempName(int loc, String prefix, Type type) {
    Identifier n = Identifier.intern(prefix + "#" + (++tempn));
    VariableAccess v =  VariableAccess.make(n, loc, 
        LocalVarDecl.make(Modifiers.NONE, null,n,
            type,
            UniqName.temporaryVariable,
            null, Location.NULL));
    return v;
  }
  
  static public Hashtable union(Hashtable h0, Hashtable h1) {
    if (h0 == null)
      return h1;
    if (h1 == null)
      return h0;
    Hashtable h2 = new Hashtable();
    for (Enumeration keys = h0.keys(); keys.hasMoreElements(); ) {
      Object key = keys.nextElement();
      h2.put(key, h0.get(key));
    }
    for (Enumeration keys = h1.keys(); keys.hasMoreElements(); ) {
      Object key = keys.nextElement();
      h2.put(key, h1.get(key));
    }
    return h2;
  }
  
  /** This method implements the ESCJ 16 function
   * <code>TargetTypeCorrect</code>,
   *
   *      except for the <code>init$</code> case!
   *
   **/
  
  /*@ requires vd == GC.allocvar.decl ==> wt != null; */
  public static Expr targetTypeCorrect(/*@ non_null */ GenericVarDecl vd,
      Hashtable wt) {
    if (vd == GC.elemsvar.decl) {
      // ElemsTypeCorrect[[ vd ]]
      return elemsTypeCorrect(vd);
    } else if (vd == GC.allocvar.decl) {
      // wt[[ alloc ]] < alloc
      VariableAccess allocPre = (VariableAccess)wt.get(GC.allocvar.decl);
      return GC.nary(TagConstants.ALLOCLT, allocPre, GC.allocvar);
    } else if (vd instanceof FieldDecl && !Modifiers.isStatic(vd.modifiers)) {
      // FieldTypeCorrect[[ vd ]]
      return fieldTypeCorrect((FieldDecl)vd);
    } else {
      // TBW:  If we ever implement safe loops, we need a case for
      // "init$" here, see ESCJ 16.
      
      // TypeCorrect[[ vd ]]
      return typeCorrect(vd);
    }
  }
  
  /** This method implements the ESCJ 16 function <code>TypeCorrect</code>.
   **/
  
  public static Expr typeCorrect(GenericVarDecl vd) {
    return typeAndNonNullCorrectAs(vd, vd.type,
        GetSpec.NonNullPragma(vd), false,
        null);
  }
  
  public static Expr typeCorrect(GenericVarDecl vd, Hashtable sp) {
    return typeAndNonNullCorrectAs(vd, vd.type,
        GetSpec.NonNullPragma(vd), false,
        sp);
  }
  
  // "vd" is a quantified variable
  public static Expr quantTypeCorrect(GenericVarDecl vd, Hashtable sp) {
    Assert.notFalse(GetSpec.NonNullPragma(vd) == null);
    if ((Types.isIntType(vd.type) || Types.isLongType(vd.type)) &&
        !Main.options().useIntQuantAntecedents) {
      return GC.truelit;
    } else {
      return typeAndNonNullCorrectAs(vd, vd.type, null, true, sp);
    }
  }
  
  public static Expr resultEqualsCall(GenericVarDecl vd, RoutineDecl rd, Hashtable sp) {
    VariableAccess v = makeVarAccess(vd, Location.NULL);
    boolean isConstructor = rd instanceof ConstructorDecl;
    
    ExprVec ev = ExprVec.make( rd.args.size()+4 );
    if (!Utils.isFunction(rd)) {
      ev.addElement( stateVar(sp) ); 
    }
    
    ArrayList bounds = new ArrayList(rd.args.size()+4);
    if (!Modifiers.isStatic(rd.modifiers)) {
      if (!isConstructor) {
        ev.addElement( apply(sp,GC.thisvar));
      }
    }
    LocalVarDecl alloc1=null, alloc2=null;
    if (isConstructor) {
      // FIXME - do we need anything for constructors? is this right?
      alloc1 = UniqName.newBoundVariable("alloc_");
      ev.addElement( makeVarAccess( (GenericVarDecl)alloc1, Location.NULL));
      alloc2 = UniqName.newBoundVariable("allocNew_");
      ev.addElement( makeVarAccess( (GenericVarDecl)alloc2, Location.NULL));
    }
    
    for (int k=0; k<rd.args.size(); ++k) {
      FormalParaDecl a = rd.args.elementAt(k);
      VariableAccess vn = makeVarAccess( a, Location.NULL);
      ev.addElement(vn);
    }    
    Expr fcall = GC.nary(fullName(rd,false), rd, ev);
    return GC.nary(TagConstants.ANYEQ, v, fcall);
  }
  
  public static Expr typeCorrectAs(/*@ non_null */ GenericVarDecl vd, /*@ non_null */ Type type) {
    return typeAndNonNullCorrectAs(vd, type, null, false, null);
  }
  
  public static Expr typeAndNonNullCorrectAs(/*@ non_null */ GenericVarDecl vd,
      /*@ non_null */ Type type,
      /*@ nullable */ SimpleModifierPragma nonNullPragma,
      boolean forceNonNull,
      /*@ nullable */ Hashtable sp) {
    return GC.and(typeAndNonNullAllocCorrectAs(vd, type,
        nonNullPragma, forceNonNull,
        sp, true));
  }
  
  /** Returns a vector of conjuncts.  This vector is "virgin" and can be modified
   * by the caller.  It contains at least 2 empty slots, allows clients to
   * append a couple of items without incurring another allocation.
   */
  
  public static ExprVec typeAndNonNullAllocCorrectAs(/*@ non_null */ GenericVarDecl vd,
      /*@ non_null */ Type type,
      /*@ nullable */ SimpleModifierPragma nonNullPragma,
      boolean forceNonNull,
      /*@ nullable */ Hashtable sp,
      boolean mentionAllocated) {
    VariableAccess v = makeVarAccess(vd, Location.NULL);
    ExprVec conjuncts = ExprVec.make(5);
    
    // is(v, type)
    Expr e = GC.nary(TagConstants.IS, v, trType(type));
    conjuncts.addElement(e);
    if (! Types.isReferenceType(type))
      return conjuncts;
    
    if (mentionAllocated) {
      // isAllocated(v, alloc)
      e = GC.nary(TagConstants.ISALLOCATED,
          v,
          apply(sp, GC.allocvar));
      conjuncts.addElement(e);
    }
    
    if (nonNullPragma != null || forceNonNull) {
      e = GC.nary(TagConstants.REFNE, v, GC.nulllit);
      if (nonNullPragma != null) {
        int locPragmaDecl = nonNullPragma.getStartLoc();
        if (Main.options().guardedVC && locPragmaDecl != Location.NULL) {
          e = GuardExpr.make(e, locPragmaDecl);
        }
        LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
      }
      conjuncts.addElement(e);
    }
    
    return conjuncts;
  }
  
  public static Expr fieldTypeCorrect(FieldDecl fdecl) {
    VariableAccess f = makeVarAccess(fdecl, Location.NULL);
    
    // f == asField(f, T)
    Expr asField = GC.nary(TagConstants.ANYEQ,
        f,
        GC.nary(TagConstants.ASFIELD,
            f,
            trType(fdecl.type)));
    if (! Types.isReferenceType(fdecl.type))
      return asField;
    
    ExprVec conjuncts = ExprVec.make(3);
    conjuncts.addElement(asField);
    
    // fClosedTime(f) < alloc
    {
      Expr c0 = GC.nary(TagConstants.ALLOCLT,
          GC.nary(TagConstants.FCLOSEDTIME, f),
          GC.allocvar);
      conjuncts.addElement(c0);
    }
    
    SimpleModifierPragma nonNullPragma = GetSpec.NonNullPragma(fdecl);
    if (nonNullPragma != null) {
      // (ALL s :: s != null ==> ...constraints on f[s]...)
      LocalVarDecl sDecl = UniqName.newBoundVariable('s');
      VariableAccess s = makeVarAccess(sDecl, Location.NULL);
      Expr antecedent = GC.nary(TagConstants.REFNE, s, GC.nulllit);
      Expr consequent;
    	if(Main.options().nne &&
    			AnnotationHandler.numOfArrayDimOfReferenceType(fdecl.type) > 0) 
    	{
    		// FIXME: Chalin: handle arrays of arbitrary dimensions.
    		// (ALL s :: s != null ==> nonnullelements(f[s],elems))
    		consequent = GC.nary(TagConstants.ELEMSNONNULL, GC.select(f, s), GC.elemsvar);
    	} else {
    		// (ALL s :: s != null ==> f[s] != null)
    		consequent = GC.nary(TagConstants.REFNE, GC.select(f, s), GC.nulllit);
    	}
      Expr quant = GC.forall(sDecl, GC.implies(antecedent, consequent));
      int locPragmaDecl = nonNullPragma.getStartLoc();
      LabelInfoToString.recordAnnotationAssumption(locPragmaDecl);
      if (Main.options().guardedVC && locPragmaDecl != Location.NULL) {
        quant = GuardExpr.make(quant, locPragmaDecl);
      }
      conjuncts.addElement(quant);
    }
    
    return GC.and(conjuncts);
  }
  
  public static Expr elemsTypeCorrect(GenericVarDecl edecl) {
    VariableAccess e = makeVarAccess(edecl, Location.NULL);
    // c0:  e == asElems(e)
    Expr c0 = GC.nary(TagConstants.ANYEQ, e, GC.nary(TagConstants.ASELEMS, e));
    // c1:  eClosedTime(e) < alloc
    Expr c1 = GC.nary(TagConstants.ALLOCLT,
        GC.nary(TagConstants.ECLOSEDTIME, e),
        GC.allocvar);
    // c0 && c1
    return GC.and(c0, c1);
  }
  
  /* getGCTagForBinary uses a 2-dim lookup table.  Each entry is:
   
   *<PRE>
   * { Binary tag,
   *     nary-tag-for-bool, nary-tag-for-integral, nary-tag-for-long-integral,
   *     nary-tag-for-floats, nary-tag-for-refs, nary-tag-for-typecode
   * }.
   * In the nary-tag-for-long-integral column, a -1 means unused.
   * In other columns, -1 means invalid combination.
   * </PRE>
   * */
  
  private static final int binary_table[][] 
                                          = { { TagConstants.OR,
                                            TagConstants.BOOLOR, -1, -1,
                                            -1, -1, -1 },
                                            { TagConstants.AND,
                                              TagConstants.BOOLAND, -1, -1,
                                              -1, -1, -1 },
                                              { TagConstants.IMPLIES,
                                                TagConstants.BOOLIMPLIES, -1, -1,
                                                -1, -1, -1 },
                                                { TagConstants.IFF,
                                                  TagConstants.BOOLEQ, -1, -1,
                                                  -1, -1, -1 },
                                                  { TagConstants.NIFF,
                                                    TagConstants.BOOLNE, -1, -1,
                                                    -1, -1, -1 },
                                                    { TagConstants.BITOR,
                                                      TagConstants.BOOLOR, TagConstants.INTEGRALOR, -1,
                                                      -1, -1, -1 },
                                                      { TagConstants.ASGBITOR,
                                                        TagConstants.BOOLOR, TagConstants.INTEGRALOR, -1,
                                                        -1, -1, -1 },
                                                        { TagConstants.BITAND,
                                                          TagConstants.BOOLAND, TagConstants.INTEGRALAND, -1,
                                                          -1, -1, -1 },
                                                          { TagConstants.ASGBITAND,
                                                            TagConstants.BOOLAND, TagConstants.INTEGRALAND, -1,
                                                            -1, -1, -1 },
                                                            { TagConstants.BITXOR,
                                                              TagConstants.BOOLNE, TagConstants.INTEGRALXOR, -1,
                                                              -1, -1, -1 },
                                                              { TagConstants.ASGBITXOR,
                                                                TagConstants.BOOLNE, TagConstants.INTEGRALXOR, -1,
                                                                -1, -1, -1 },
                                                                { TagConstants.EQ,
                                                                  TagConstants.BOOLEQ, TagConstants.INTEGRALEQ, -1,
                                                                  TagConstants.FLOATINGEQ, TagConstants.REFEQ, TagConstants.TYPEEQ },
                                                                  { TagConstants.NE,
                                                                    TagConstants.BOOLNE, TagConstants.INTEGRALNE, -1,
                                                                    TagConstants.FLOATINGNE, TagConstants.REFNE,  TagConstants.TYPENE },
                                                                    { TagConstants.GE,
                                                                      -1, TagConstants.INTEGRALGE, -1,
                                                                      TagConstants.FLOATINGGE, -1, -1 },
                                                                      { TagConstants.GT,
                                                                        -1, TagConstants.INTEGRALGT, -1,
                                                                        TagConstants.FLOATINGGT, -1, -1 },
                                                                        { TagConstants.LE,
                                                                          -1, TagConstants.INTEGRALLE, -1,
                                                                          TagConstants.FLOATINGLE, TagConstants.LOCKLE, -1 },
                                                                          { TagConstants.LT,
                                                                            -1, TagConstants.INTEGRALLT, -1,
                                                                            TagConstants.FLOATINGLT, TagConstants.LOCKLT, -1 },
                                                                            { TagConstants.LSHIFT,
                                                                              -1, TagConstants.INTSHIFTL, TagConstants.LONGSHIFTL,
                                                                              -1, -1, -1 },
                                                                              { TagConstants.ASGLSHIFT,
                                                                                -1, TagConstants.INTSHIFTL, TagConstants.LONGSHIFTL,
                                                                                -1, -1, -1 },
                                                                                { TagConstants.RSHIFT,
                                                                                  -1, TagConstants.INTSHIFTR, TagConstants.LONGSHIFTR,
                                                                                  -1, -1, -1 },
                                                                                  { TagConstants.ASGRSHIFT,
                                                                                    -1, TagConstants.INTSHIFTR, TagConstants.LONGSHIFTR,
                                                                                    -1, -1, -1 },
                                                                                    { TagConstants.URSHIFT,
                                                                                      -1, TagConstants.INTSHIFTRU, TagConstants.LONGSHIFTRU,
                                                                                      -1, -1, -1 },
                                                                                      { TagConstants.ASGURSHIFT,
                                                                                        -1, TagConstants.INTSHIFTRU, TagConstants.LONGSHIFTRU,
                                                                                        -1, -1, -1 },
                                                                                        { TagConstants.ADD,
                                                                                          -1, TagConstants.INTEGRALADD, -1,
                                                                                          TagConstants.FLOATINGADD, -1, -1 },  // see below
                                                                                          { TagConstants.ASGADD,
                                                                                            -1, TagConstants.INTEGRALADD, -1,
                                                                                            TagConstants.FLOATINGADD, -1, -1 },
                                                                                            { TagConstants.SUB,
                                                                                              -1, TagConstants.INTEGRALSUB, -1,
                                                                                              TagConstants.FLOATINGSUB, -1, -1 },
                                                                                              { TagConstants.ASGSUB,
                                                                                                -1, TagConstants.INTEGRALSUB, -1,
                                                                                                TagConstants.FLOATINGSUB, -1, -1 },
                                                                                                { TagConstants.DIV,
                                                                                                  -1, TagConstants.INTEGRALDIV, -1,
                                                                                                  TagConstants.FLOATINGDIV, -1, -1 },
                                                                                                  { TagConstants.ASGDIV,
                                                                                                    -1, TagConstants.INTEGRALDIV, -1,
                                                                                                    TagConstants.FLOATINGDIV, -1, -1 },
                                                                                                    { TagConstants.MOD,
                                                                                                      -1, TagConstants.INTEGRALMOD, -1,
                                                                                                      TagConstants.FLOATINGMOD, -1, -1 },
                                                                                                      { TagConstants.ASGREM,
                                                                                                        -1, TagConstants.INTEGRALMOD, -1,
                                                                                                        TagConstants.FLOATINGMOD, -1, -1 },
                                                                                                        { TagConstants.STAR,
                                                                                                          -1, TagConstants.INTEGRALMUL, -1,
                                                                                                          TagConstants.FLOATINGMUL, -1, -1 },
                                                                                                          { TagConstants.ASGMUL,
                                                                                                            -1, TagConstants.INTEGRALMUL, -1,
                                                                                                            TagConstants.FLOATINGMUL, -1, -1 } };
  
  static {
    for (int i = 0; i < binary_table.length; i++) {
      if (binary_table[i].length != 7)
        Assert.fail("bad length, binary_table row " + i);
    }
  }
  
  static public int getGCTagForBinary(BinaryExpr be) {
    
    int tag = be.getTag();
    
    Type leftType = TypeCheck.inst.getType( be.left );
    Type rightType = TypeCheck.inst.getType( be.right );
    
    // before consulting the table, handle "+" & "+=" on strings:
    if ((tag == TagConstants.ADD || tag == TagConstants.ASGADD) &&
        (Types.isReferenceOrNullType(leftType) ||
            Types.isReferenceOrNullType(rightType))) {
      // this "+" or "+=" denotes string concatenation
      return TagConstants.STRINGCAT;
    }
    
    // find tag in table
    int i;
    
    for( i=0; i<binary_table.length; i++ ) {
      if (binary_table[i][0] == tag)
        break;
    }
    if( i==binary_table.length )
      Assert.fail("Bad tag "+TagConstants.toString(tag));
    
    // have index of correct line in i.
    int naryTag;
    if (Types.isBooleanType( leftType ) 
        && Types.isBooleanType( rightType )) {
      naryTag = binary_table[i][1];
    } else if (Types.isLongType( leftType ) 
        && Types.isIntegralType( rightType )) {
      if (binary_table[i][3] != -1) {
        naryTag = binary_table[i][3];
      } else {
        naryTag = binary_table[i][2];
      }
    } else if (Types.isIntegralType( leftType ) 
        && Types.isIntegralType( rightType )) {
      naryTag = binary_table[i][2];
    } else if (Types.isNumericType( leftType ) 
        && Types.isNumericType( rightType )) {
      // ## should convert integral args to floating point
      naryTag = binary_table[i][4];
    } else if (Types.isReferenceOrNullType( leftType ) 
        && Types.isReferenceOrNullType( rightType )) {
      naryTag = binary_table[i][5];
    } else if (Types.isTypeType(leftType)
        && Types.isTypeType(rightType)) {
      naryTag = binary_table[i][6];
    } else if (Types.isErrorType(leftType)) {
      ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,be.left.getStartLoc(),
      "Failed to translate some unimplemented construct");
      naryTag = -1; // dummy assignment
    } else if (Types.isErrorType(rightType)) {
      if (be.right instanceof AmbiguousVariableAccess)
        ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,be.right.getStartLoc(),
        "Unknown variable");
      else
        
        ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,be.right.getStartLoc(),
        "Failed to translate some unimplemented construct");
      naryTag = -1; // dummy assignment
    } else {
      System.out.println("LTYPE " + leftType);
      System.out.println("RTYPE " + rightType);
      Assert.fail("Bad types on tag "+TagConstants.toString(tag) );
      naryTag = -1; // dummy assignment
    }
    
    // have naryTag
    if( naryTag == -1 ) {
      Assert.fail("Bad tag/types combination at "+TagConstants.toString(tag)
          +" left type "+leftType
          +" right type "+rightType);
    }
    
    return naryTag;
  }
  
  /* getGCTagForUnary uses a 2-dim lookup table.  Each entry is:
   
   * <pre>
   * { Unary tag,
   *   nary-tag-for-bool-args,
   *   nary-tag-for-integral,
   *   nary-tag-for-floats
   * }.
   * Use -1 if invalid combination.
   * </pre>
   */
  
  private static final int unary_table[][] 
                                         = { { TagConstants.UNARYSUB, -1, TagConstants.INTEGRALNEG,
                                           TagConstants.FLOATINGNEG },
                                           { TagConstants.NOT, TagConstants.BOOLNOT, -1, -1 },
                                           { TagConstants.BITNOT, -1, TagConstants.INTEGRALNOT, -1 },
                                           { TagConstants.INC, -1, TagConstants.INTEGRALADD,
                                             TagConstants.FLOATINGADD },
                                             { TagConstants.POSTFIXINC, -1, TagConstants.INTEGRALADD,
                                               TagConstants.FLOATINGADD },
                                               { TagConstants.DEC, -1, TagConstants.INTEGRALSUB,
                                                 TagConstants.FLOATINGSUB },
                                                 { TagConstants.POSTFIXDEC, -1, TagConstants.INTEGRALSUB,
                                                   TagConstants.FLOATINGSUB } };
  
  static {
    for (int i = 0; i < unary_table.length; i++) {
      if (unary_table[i].length != 4)
        Assert.fail("bad length, unary table row " + i);
    }
  }
  
  /** TBW.  Requires e.getTag() in { UNARYSUB, NOT, BITNOT, INC,
   POSTFIXINC, DEC, POSTFIXDEC } */
  
  static public int getGCTagForUnary(UnaryExpr e) {
    // find correct row in table
    int tag = e.getTag();
    int row;
    for (row = 0; row < unary_table.length; row++) {
      if (unary_table[row][0] == tag)
        break;
    }
    Assert.notFalse(row < unary_table.length, "Bad tag");
    
    // find correct column in table
    /*-@ uninitialized */ int col = 0;
    Type argType = TypeCheck.inst.getType(e.expr);
    if (Types.isBooleanType(argType)) col = 1;
    else if (Types.isIntegralType(argType)) col = 2;
    else if (Types.isNumericType(argType)) col = 3;
    else if (Types.isErrorType(argType)) {
      ErrorSet.notImplemented(!Main.options().noNotCheckedWarnings,
          e.expr.getStartLoc(),
      "Failed to translate some unimplemented construct");
    } else Assert.fail("Bad type");
    
    int result = unary_table[row][col];
    Assert.notFalse(result != -1, "Bad type combination");
    return result;
  }
  
  public static VariableAccess makeVarAccess(/*@ non_null */ GenericVarDecl decl, int loc) {
    return VariableAccess.make(decl.id, loc, decl);
  }
  
  /** Returns the number of variable substitutions that calls to
   * <code>trSpecExpr</code> have caused.  To find out how many times a
   * substitution was made, a caller can surround the call to
   * <code>trSpecExpr</code> by two calls to this method and compute the
   * difference.
   **/
  
  static public int getReplacementCount() {
    return cSubstReplacements;
  }
  
  private static int cSubstReplacements = 0;
  
  protected static VariableAccess apply(Hashtable map, VariableAccess va) {
    if (map == null) {
      return va;
    }
    VariableAccess v = (VariableAccess)map.get(va.decl);
    if (v != null) {
      if (va.decl == GC.thisvar.decl) cSubstReplacements++;
      return v;
    } else {
      return va;
    }
  }
  
  public static /*@ non_null */ TypeExpr trType(/*@ non_null */ Type type) {
    // Other than unique-ification of type names, what distinguishes
    // Java types from the way they are represented in the verification
    // condition is that a Java array type "T[]" is represented as
    // "(array T)".  The definition of "TrType" in ESCJ 16 spells out
    // the details of this conversion.  However, unlike what ESCJ 16
    // says, the ESC/Java implementation represents types in
    // guarded-command expressions as "TypeExpr" nodes.  The conversion
    // between "[]" and "array" is done when the "TypeExpr" is converted
    // to a verification condition string.
    return TypeExpr.make(type.getStartLoc(), type.getEndLoc(), type);
  }
  
  public static void getCalledSpecs( 
      RoutineDecl decl,
      ObjectDesignator od, ExprVec ev, 
      Expr resultVar,
      Hashtable sp, Hashtable st) {
    //System.out.println("GCS " + decl.parent.id + " " + (decl instanceof MethodDecl ? (((MethodDecl)decl).id.toString()) : "" ) + " " + Location.toString(decl.getStartLoc()) + " " + (declStack.contains(decl)) );
    ModifierPragmaVec md = decl.pmodifiers;
    if (md == null) return;
    for (int i=0; i<md.size(); ++i) {
      ModifierPragma m = md.elementAt(i);
      switch (m.getTag()) {
        case TagConstants.ENSURES:
        case TagConstants.POSTCONDITION:
        {
          if (Utils.ensuresDecoration.isTrue(m)) {
            //System.out.println("SKIPPING-A");
            continue;
          }
          Expr e = ((ExprModifierPragma)m).expr;
          /*
           System.out.println("GCS-INSERTING FOR " + decl.parent.id + " " + (decl instanceof MethodDecl ? ((MethodDecl)decl).id.toString() : "") + " " + Location.toString(m.getStartLoc()) );
           EscPrettyPrint.inst.print(System.out,0,m);
           System.out.println("");
           */
          //System.out.println("ENSURES " + e);
          Hashtable h = new Hashtable();
          Expr oldResultExpr = specialResultExpr;
          specialResultExpr = resultVar;
          //h.put(Substitute.resexpr,resultVar);
          if (od instanceof ExprObjectDesignator) {
            if (!(((ExprObjectDesignator)od).expr instanceof ThisExpr)) {
              h.put(Substitute.thisexpr, ((ExprObjectDesignator)od).expr);
            }
          } else if (od instanceof SuperObjectDesignator) {
            // FIXME
            System.out.println("NOT SUPPORTED-D " + od.getClass());
          } // fall-through for TypeObjectDesignator or null
          
          FormalParaDeclVec args = decl.args;
          for (int j=0; j<args.size(); ++j) {
            h.put(args.elementAt(j), ev.elementAt(j));
          }
          e = Substitute.doSimpleSubst(h,e);
          e = trSpecExpr(e,sp,st);
          trSpecExprAuxConditions.addElement(e);
          specialResultExpr = oldResultExpr;
          
          // System.out.println("INSERTING-END " + Location.toString(m.getStartLoc()) );
          // EscPrettyPrint.inst.print(System.out,0,e);
          // System.out.println("");
           
        }
        
        default:
          break;
      }
    }
    
    // FIXME - What about constraint clauses
    /*
     
     TypeDeclElemVec tdev = decl.parent.elems;
     for (int j=0; j<tdev.size(); ++j) {
     TypeDeclElem e = tdev.elementAt(j);
     if (!(e instanceof TypeDeclElemPragma)) continue;
     TypeDeclElemPragma p = (TypeDeclElemPragma)e;
     if (p.getTag() == TagConstants.INVARIANT) {
     Expr ee = ((ExprDeclPragma)p).expr;
     Hashtable h = new Hashtable();
     h.put(Substitute.resexpr,resultVar);
     if (od instanceof ExprObjectDesignator) {
     if (!(((ExprObjectDesignator)od).expr instanceof ThisExpr)) {
     h.put(Substitute.thisexpr, ((ExprObjectDesignator)od).expr);
     }
     } else if (od instanceof SuperObjectDesignator) {
     // FIXME
      System.out.println("NOT SUPPORTED-E " + od.getClass());
      } else if (od == null) { // Constructor case
      h.put(Substitute.thisexpr, resultVar);
      } // fall-through for TypeObjectDesignator 
      ee = Substitute.doSimpleSubst(h,ee);
      ee = trSpecExpr(ee,sp,st);
      trSpecExprAuxConditions.addElement(ee);
      }
      }
      */
  }
  
  // This creates a unique name for a function call
  // The same name gets used for overriding methods
  static public Identifier fullName(RoutineDecl rd,boolean useSuper) {
    String fullname;
    if (Modifiers.isStatic(rd.getModifiers()) || rd instanceof ConstructorDecl) {
      int loc = rd.getStartLoc();
      TypeSig sig = TypeSig.getSig(rd.parent);
      if (useSuper) sig = sig.superClass();
      fullname = sig.getExternalName() + "." 
      + rd.id().toString() + "." ;
      int line = Location.toLineNumber(loc);
      if (line == 1) {
        // If the reference is to a binary file, there is no unique 
        // declaration location, so we append a hash code
        fullname = fullname + rd.hashCode();
      } else {
        fullname = fullname + line + "." + Location.toColumn(loc);
      }
    } else {
      fullname = rd.id().toString() + "#";
      if (!Utils.isFunction(rd)) fullname = fullname + "#state";
      for (int i=0; i<rd.args.size(); ++i) {
        Type t = rd.args.elementAt(i).type;
        fullname = fullname + "#" + Types.printName(t);
      }
    }
    return Identifier.intern(fullname); 
  }
  
  
  static public Expr getEquivalentAxioms(RepHelper o, Hashtable sp) {
    Expr ax = null;
    ASTNode astn = o.a;
    TypeDecl td = o.td;
    initForClause();
    if (astn instanceof RoutineDecl) {
      boolean isConstructor = astn instanceof ConstructorDecl;
      RoutineDecl rd = (RoutineDecl)astn;
      GenericVarDecl newThis = UniqName.newBoundThis();


      // NOTE: We wrap each postcondition in a separate forall quantification
      // because the variable names might be different.  In addition, 
      // Esc/java tacks on location information to make the names somewhat
      // unique, and these differ for different postconditions.  The 
      // postconditions themselves already have the variable usages within
      // them resolved to point to the original formal parameter declarations.
      // Thus it is easiest to simply use the formal parameters for the 
      // quantification variables.  Once could combine all the postconditions
      // that are associated with a single syntactic declaration; 
      // constructors and static methods have just one relevant declaration;
      // but we haven't done this - in part because the output is more readable
      // with the conjunct broken up.

      ModifierPragmaVec v = Utils.getAllSpecs(rd);
      ExprVec conjuncts = ExprVec.make(v.size());


      // FIXME - what about ensures clauses with \old in them
      // Note - if there is an ensures clause with object fields, then it is
      // not a great candidate for a function call
      
      Expr ex = escjava.AnnotationHandler.T;
      javafe.tc.FlowInsensitiveChecks.setType(ex, Types.booleanType);
      for (int i=0; i<v.size(); ++i) {
        ModifierPragma p = v.elementAt(i);
        if (p.getTag() == TagConstants.SIGNALS) { // includes SIGNALS_ONLY
          //Expr e = trSpecExpr(((VarExprModifierPragma)p).expr,null,null); // FIXME - no substitution?
          Expr e = ((VarExprModifierPragma)p).expr;
          ex = BinaryExpr.make(TagConstants.AND,ex,e,Location.NULL);
          javafe.tc.FlowInsensitiveChecks.setType(ex, Types.booleanType);
        }
      }
      Expr notExceptionalReturn = UnaryExpr.make(TagConstants.NOT,ex,
        Location.NULL);
      javafe.tc.FlowInsensitiveChecks.setType(notExceptionalReturn, Types.booleanType);
      
      ex = escjava.AnnotationHandler.T;
      javafe.tc.FlowInsensitiveChecks.setType(ex, Types.booleanType);
      for (int i=0; i<v.size(); ++i) {
        ModifierPragma p = v.elementAt(i);
        if (p.getTag() == TagConstants.DIVERGES) {
          //Expr e = trSpecExpr(((ExprModifierPragma)p).expr,null,null); // FIXME - no substitution?
          Expr e = ((ExprModifierPragma)p).expr;
          ex = BinaryExpr.make(TagConstants.AND,ex,e,Location.NULL);
          javafe.tc.FlowInsensitiveChecks.setType(ex, Types.booleanType);
        }
      }
      Expr notDivergentReturn = UnaryExpr.make(TagConstants.NOT,ex,
                Location.NULL);
      javafe.tc.FlowInsensitiveChecks.setType(notDivergentReturn, Types.booleanType);
      
      Expr guard = ex == escjava.AnnotationHandler.T ? notExceptionalReturn :
              BinaryExpr.make(TagConstants.AND,notExceptionalReturn,notDivergentReturn,Location.NULL);
      javafe.tc.FlowInsensitiveChecks.setType(guard, Types.booleanType);
      
      // find ensures pragmas and translate them into axioms
      for (int i=0; i<v.size(); ++i) {
        ModifierPragma p = v.elementAt(i);
        if (p.getTag() != TagConstants.ENSURES &&
            p.getTag() != TagConstants.POSTCONDITION) continue;
        if (Utils.ensuresDecoration.isTrue(p)) {
          //System.out.println("SKIPPING-B");
          continue;
        }

        //System.out.println("IN GETEQUIVALENTAXIOMS");
        //escjava.AnnotationHandler.printSpec(p);
      RoutineDecl trd = (RoutineDecl) Utils.owningDecl.get(p);

      ArrayList bounds = makeBounds(trd, newThis);
      
      Expr fcall = createFunctionCall(trd,bounds,sp);

      specialResultExpr = fcall;
      if (astn instanceof MethodDecl) {
        specialThisExpr = makeVarAccess( newThis, Location.NULL);
      } else {
        specialThisExpr = fcall;
      }

      Expr e = ((ExprModifierPragma)p).expr;

      Expr translatedGuard = trSpecExpr(guard,null,null);
      Expr translatedPostcondition = trSpecExpr(e,null,null); // FIXME - no subst?
      translatedPostcondition = GC.implies(translatedGuard,translatedPostcondition);  

        //System.out.println("NEW POST");
        //escjava.ast.EscPrettyPrint.inst.print(System.out,0,translatedPostcondition);
        //System.out.println();
        
        // Wrap the expression in a forall

        Expr ee = createForall(translatedPostcondition,fcall,bounds);
        //escjava.ast.EscPrettyPrint.inst.print(System.out,0,ee);
        //System.out.println();
        conjuncts.addElement(ee);
      }

      specialResultExpr = null;
      specialThisExpr = null;

      // There are a few more special axioms to create

      ArrayList bounds = makeBounds(rd, newThis);
      Expr fcall = createFunctionCall(rd,bounds,sp);
      
      Expr expr2;
      if (isConstructor) {
	ExprVec conjuncts2 = ExprVec.make(4);
        // If this is a constructor, then we are generating
        // axioms for a new instance expression.  There are
        // a couple more implicit axioms
        
        // result != null
        Expr ee = GC.nary(TagConstants.REFNE, fcall, GC.nulllit);
        conjuncts2.addElement(ee);
        
        // Adds \typeof(v) == \type(...)
        Type type = TypeSig.getSig(rd.parent);
        ee = GC.nary(TagConstants.TYPEEQ,
            GC.nary(TagConstants.TYPEOF, fcall),
            trType(type));
        //GC.nary(TagConstants.CLASSLITERALFUNC, trType(type)));
        conjuncts2.addElement(ee);
        // Adds ! isAllocated(v, alloc)
        ee = GC.nary(TagConstants.BOOLNOT,
            GC.nary(TagConstants.ISALLOCATED, fcall, 
                makeVarAccess( (LocalVarDecl)bounds.get(0), Location.NULL)));
        conjuncts2.addElement(ee);
        // Adds isAllocated(v, newalloc)
        ee = GC.nary(TagConstants.ISALLOCATED, fcall,
            makeVarAccess( (LocalVarDecl)bounds.get(1), Location.NULL));
        conjuncts2.addElement(ee);
        expr2 = GC.and(conjuncts2);
      } else {
        // add an axiom about the type of the method result
        Type type = ((MethodDecl)rd).returnType;
        // Is allows the result to be null for reference types
        // but is equivalent to \typeof() == . for primitive types
        expr2 = GC.nary(TagConstants.IS,
            fcall,
            trType(type));
      }
      
      
      conjuncts.addElement(createForall(expr2,fcall,bounds));

      // create a composite AND of all the foralls
      ax = GC.and(conjuncts);

    } else if (astn instanceof FieldDecl) {
      FieldDecl fd = (FieldDecl)astn;
      Expr ee = GC.and(getModelVarAxioms(td,fd,sp));
      ax = ee;
      
    } else {
      //System.out.println("NOTHING FOR TYPE YET");
      // FIXME are these already included by virtue of the FindContributors mechanism
      ax = GC.truelit;
    }
    
    closeForClause();
    return ax;
  }

  static private ArrayList makeBounds(RoutineDecl rd, GenericVarDecl newThis) {
      // Make the list of bound parameters
      ArrayList bounds = new ArrayList(rd.args.size()+4);
      Hashtable h = new Hashtable();
      LocalVarDecl alloc1=null, alloc2=null;
      if (rd instanceof ConstructorDecl) {
        alloc1 = UniqName.newBoundVariable("alloc_");
        bounds.add(alloc1);
        alloc2 = UniqName.newBoundVariable("allocNew_");
        bounds.add(alloc2);
      }
      if (!Modifiers.isStatic(rd.modifiers)) {
        if (rd instanceof MethodDecl) bounds.add( newThis );
      }
      for (int k=0; k<rd.args.size(); ++k) {
        FormalParaDecl a = rd.args.elementAt(k);
        //LocalVarDecl n = UniqName.newBoundVariable(a.id.toString());
        VariableAccess vn = makeVarAccess( a, Location.NULL);
        bounds.add(a);
        h.put(a,vn);
      }
      return bounds;
  }
  
  static private Expr createFunctionCall(RoutineDecl rd, ArrayList bounds, Hashtable sp) {

      // create a function call
      ExprVec ev = ExprVec.make( bounds.size()+1 );
      if (!Utils.isFunction(rd)) {
        ev.addElement( stateVar(sp) );
      }
      for (int k=0; k<bounds.size(); ++k) {
        ev.addElement( makeVarAccess( (GenericVarDecl)bounds.get(k), Location.NULL));
      }
      return GC.nary(fullName(rd,false), rd, ev);      
  }


    //TODO: make the freshner somewhat nicer   
    private static int fcounter = 0;
 
   /**
    *  Make new vars that should be fresh.
    *
    * @param bounds original vars, <code>ArrayList</code> of <code>GenericVarDecl</code>
    * @param newBounds to this <code>ArrayList</code> fresh  <code>GenericVarDecl</code> will be added; one for each old var
    * @return a substitution that will replace old vars with the new ones.
    */
    //@ ensures \fresh(\result);
    /*@pure*/static private Hashtable makeFreshner(/*@non_null*/ArrayList bounds, /*@non_null*/ArrayList newBounds) {
        Hashtable subst = new Hashtable();
        java.util.Set boundsSet = new java.util.HashSet(bounds);
        for (java.util.Iterator it = boundsSet.iterator(); it.hasNext(); ) {
            GenericVarDecl oldVar = (GenericVarDecl) it.next();
            String newIdStr = oldVar.id + "_freshness" + fcounter++;
            Identifier newId = Identifier.intern(newIdStr);
            GenericVarDecl newVar = LocalVarDecl.make(oldVar.modifiers, null, newId, oldVar.type, oldVar.locId, null, Location.NULL);

            /*debug System.out.println("old var: " + oldVar); */
            /*debug System.out.println("new var: " + newVar); */

            VariableAccess newVarAccess = VariableAccess.make(newVar.id, Location.NULL, newVar);
            subst.put(oldVar, newVarAccess);
            newBounds.add(newVar);
        }
        return subst;
    }


  static private Expr createForall(Expr expr, Expr fcall, ArrayList bounds)  {
      // rename and substitue vars to obtain unique
      ArrayList newBounds = new ArrayList(bounds.size());
      Hashtable subst = makeFreshner(bounds, newBounds);
      // substitute old vars with fresh ones
      expr = escjava.translate.Substitute.doSubst(subst, expr);
      fcall = escjava.translate.Substitute.doSubst(subst, fcall);
      bounds = newBounds;

      // create a forall expr for each bound var
      // TODO: why both the expression with pats and with no pats?--mikolas
      Expr ee = expr;
      Expr ee2 = ee;
      ExprVec pats = ExprVec.make(1);
      pats.addElement(fcall);
      for (int k=bounds.size()-1; k>=0; --k) {
        GenericVarDecl oo = (GenericVarDecl)bounds.get(k);
        ee = GC.forall(oo,ee);
        ee2 = GC.forallwithpats(oo,ee2,pats);
      }
      Expr retv = GC.and(ee,ee2);
      return retv;
  }
  
  /** Translates an individual represents clause into a class-level axiom. */
  static public Expr getRepresentsAxiom(NamedExprDeclPragma p, Hashtable sp) {
    boolean isStatic;
    ASTNode symbol = null;
    if (p.target instanceof FieldAccess) {
      isStatic = Modifiers.isStatic(((FieldAccess)p.target).decl.modifiers);
      symbol = ((FieldAccess)p.target).decl;
    } else {
      System.out.println("UNSUPPORTED OPTION-GRA " + p.target.getClass());  // FIXME - array access ??
      isStatic = false;
    }
    GenericVarDecl newThis = UniqName.newBoundThis();
    specialThisExpr = makeVarAccess( newThis, Location.NULL);
    ExprVec args = ExprVec.make(2);
    args.addElement(stateVar(sp));
    if (!isStatic) args.addElement(specialThisExpr);
    ExprVec pats = ExprVec.make(2);
    Expr fcall = GC.nary(representsMethodName(p.target), symbol, args);
    pats.addElement(fcall);
    Expr e = TrAnExpr.trSpecExpr(p.expr, null, null);
    //e = GC.forallwithpats(newThis,e,pats);
    e = GC.forall(newThis,e);
    specialThisExpr = null;
    return e;
  }
  
  static public Identifier representsMethodName(Object pt) {
    String id = "ZZZZZZZZZZZZZZ";
    FieldDecl d;
    
    if (pt instanceof FieldDecl) {
      d = (FieldDecl)pt;
    } else if (pt instanceof FieldAccess) {
      FieldAccess fa = (FieldAccess)pt;
      d = fa.decl;
    } else if (pt instanceof VariableAccess) {
      GenericVarDecl dd = ((VariableAccess)pt).decl;
      if (dd instanceof FieldDecl) d = (FieldDecl)dd;
      else {
        System.out.println("RMN " + dd.getClass() + " " + dd.toString() );
        return Identifier.intern(id);
      }
    } else {
      System.out.println("RMN " + pt.getClass());
      return Identifier.intern(id);
    }
    id = TypeSig.getSig(d.parent).getExternalName() + "#"  + d.id.toString();
    return Identifier.intern(id);
  }
  
  public static ExprVec getModelVarAxioms(TypeDecl td, FieldDecl fd, Hashtable sp) {
    //TypeDeclElemVec tv = GetSpec.getRepresentsClauses(td,fd);
    TypeDeclElemVec tv = GetSpec.getRepresentsClauses(null,fd);
    
    ExprVec ev = ExprVec.make();
    if (tv != null) for (int i=0; i<tv.size(); ++i) {
      NamedExprDeclPragma p = (NamedExprDeclPragma)tv.elementAt(i);
      ev.addElement(getRepresentsAxiom(p,sp));
    }
    return ev;
  }
  
  public static VariableAccess stateVar(Hashtable sp) {
    if (sp == null) { // current context
      return currentState;
    } else { // possible old context
      return apply(sp,GC.statevar);
    }
  }
}
