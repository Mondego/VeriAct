/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.ast;

import javafe.ast.ASTNode;
import javafe.ast.ASTDecoration;
import javafe.ast.ModifierPragma;
import javafe.ast.ModifierPragmaVec;
import javafe.ast.GenericVarDecl;
import javafe.ast.IdentifierNode;
import javafe.ast.ASTDecoration;
import javafe.ast.RoutineDecl;
import javafe.ast.MethodDecl;
import javafe.ast.ClassDecl;
import javafe.ast.TypeDecl;
import javafe.tc.TypeSig;
import javafe.ast.Type;
import javafe.ast.TypeName;
import javafe.ast.ArrayType;
import javafe.ast.PrimitiveType;
import javafe.ast.FormalParaDecl;
import javafe.ast.FormalParaDeclVec;
import javafe.ast.*;
import escjava.Main;

import javafe.util.*;
import java.util.Enumeration;


public final class Utils
{
  /**
   * Finds and returns the first modifier pragma of <code>vdecl</code>
   * that has the tag <code>tag</code>, if any.  If none, returns
   * <code>null</code>.<p>
   *
   * Note, if you want to know whether a variable is
   * <code>non_null</code> or <code>nullable</code> (outside of called
   * within AnnotationHandler, of course), use method 
   * {@link escjava.AnnotationHandler.getNonNullAndNullable(javafe.ast.RoutineDecl)}
   * instead, for it properly handles inheritance of
   * <code>non_null</code> and <code>nullable</code> pragmas.
   **/
  static public /*@nullable*/ ModifierPragma findModifierPragma(/*@ non_null */ GenericVarDecl vdecl,
                                                  int tag) {
    return findModifierPragma(vdecl.pmodifiers,tag);
  }

  static public /*@nullable*/ ModifierPragma findModifierPragma(ModifierPragmaVec mp,
                                                  int tag) {
    if (mp != null) {
      for (int j = 0; j < mp.size(); j++) {
        ModifierPragma prag= mp.elementAt(j);
        if (prag.getTag() == tag)
          return prag;
      }
    }
    return null;  // not present
  }

  static public void removeModifierPragma(/*@ non_null */ GenericVarDecl vdecl, int tag) {
    removeModifierPragma(vdecl.pmodifiers,tag);
  }

  static public void removeModifierPragma(ModifierPragmaVec p, int tag) {
    if (p != null) {
      for (int j = 0; j < p.size(); j++) {
        ModifierPragma prag= p.elementAt(j);
        if (prag.getTag() == tag) {
          p.removeElementAt(j);
          --j;
        }
      }
    }
  }

  static public boolean isModel(javafe.ast.FieldDecl fd) {
    return isModel(fd.pmodifiers);
  }

  static public boolean isModel(ModifierPragmaVec m) {
    if (m == null) return false;
    return findModifierPragma(m,TagConstants.MODEL) != null;
  }

  // Used for designator expressions, as in a modifies clause.
  static public boolean isModel(Expr e) {
    if (e instanceof VariableAccess) {
      VariableAccess va = (VariableAccess)e;
      if (va.decl instanceof FieldDecl) {
        return isModel( (FieldDecl)va.decl );
      }
      //System.out.println("ISMODEL-VA " + va.decl.getClass());
    } else if (e instanceof FieldAccess) {
      return isModel( ((FieldAccess)e).decl );
    } else if (e instanceof NothingExpr) {
      return true; 
    } else if (e instanceof EverythingExpr) {
      return false;
    } else if (e instanceof ArrayRefExpr) {
      return isModel( ((ArrayRefExpr)e).array );
    } else if (e instanceof ArrayRangeRefExpr) {
      return isModel( ((ArrayRangeRefExpr)e).array );
    } else if (e instanceof WildRefExpr) {
      return false;
    } else if (e instanceof NaryExpr) {
      // This can occur if \dttfsa is used in a modifies clause
      return false;
    } else {
      //System.out.println(EscPrettyPrint.inst.toString(e));
      //System.out.println("ISMODEL " + e.getClass());
      //ErrorSet.dump(null);
    }
    return false; // default
  }

  static public abstract class BooleanDecoration extends ASTDecoration {
    private static final Object decFALSE = new Object();
    private static final Object decTRUE = new Object();
    public BooleanDecoration(String s) {
      super(s);
    }
    public void set(ASTNode o, boolean b) {
      set(o, b?decTRUE:decFALSE);
    }
    public boolean isTrue(ASTNode o) {
      Object res = get(o);
      if (res == null) {
        boolean b = calculate(o);
        set(o,b);
        return b;
      }
      return res == decTRUE;
    }
    public abstract boolean calculate(ASTNode o);
  }

  static private BooleanDecoration pureDecoration = new BooleanDecoration("pure") {
      public boolean calculate(ASTNode o) {
        RoutineDecl rd = (RoutineDecl)o;
        return findPurePragma(rd) != null;
      }
    };
  static public boolean isPure(RoutineDecl rd) {
    return pureDecoration.isTrue(rd);
  }
  static public void setPure(RoutineDecl rd) {
    pureDecoration.set(rd,true);
  }
  static public ModifierPragma findPurePragma(RoutineDecl rd) {
    ModifierPragma m = null;
    m = findModifierPragma(rd.pmodifiers,TagConstants.PURE);
    if (m!=null) return m;
    m = findModifierPragma(rd.parent.pmodifiers,TagConstants.PURE);
    if (m != null) return m;
    if (rd instanceof MethodDecl) {
      MethodDecl md = (MethodDecl)rd;
      Set direct = javafe.tc.PrepTypeDeclaration.getInst().getOverrides(md.parent, md);
      Enumeration e = direct.elements();
      while (e.hasMoreElements()) {
        MethodDecl directMD = (MethodDecl)e.nextElement();
        m = findPurePragma(directMD);
        if (m != null) return m;
      }
    } 
    return null;
  }
  private static final BooleanDecoration immutableDecoration = new BooleanDecoration("immutable") {
      public boolean calculate(ASTNode o) {
        return findModifierPragma(((TypeDecl)o).pmodifiers, TagConstants.IMMUTABLE)
          != null ;
        //|| findModifierPragma(((TypeDecl)o).pmodifiers, TagConstants.PURE) != null;
      }
    };
  public static boolean isImmutable(TypeDecl cd) {
    return immutableDecoration.isTrue(cd);
  }

  public static final BooleanDecoration ensuresDecoration = new BooleanDecoration("ensuresFromException") {
    public boolean calculate(ASTNode o) {
      return false;
    }
  };

  private static final BooleanDecoration allocatesDecoration = new BooleanDecoration("allocates") {
    public boolean calculate(ASTNode o) {
      if (!(o instanceof MethodDecl)) {
         //System.out.println("CALLING ALLOCATES ON NOT METHOD " + Location.toString(o.getStartLoc()));
         return true;
      }
      MethodDecl rd = (MethodDecl)o;
        javafe.util.Set ov = escjava.tc.FlowInsensitiveChecks.getDirectOverrides((MethodDecl)rd);
        Enumeration k = ov.elements();
        while (k.hasMoreElements()) {
           if (isAllocates( (MethodDecl)k.nextElement() )) {
//System.out.println("PARENT ALLOCATES " + rd.id() + " " + Location.toString(rd.getStartLoc()));
               return true;
           }
        }
        ModifierPragmaVec mpv = rd.pmodifiers;
        for (int i=0; i<mpv.size(); ++i) {
          ModifierPragma m = mpv.elementAt(i);
          if (m instanceof ExprModifierPragma) {
            if (checkForFresh( ((ExprModifierPragma)m).expr )) {
//System.out.println("ENSURES ALLOCATES " + rd.id() + " " + Location.toString(m.getStartLoc()));
              return true;
            }
          } else if (m instanceof VarExprModifierPragma) {
            if (checkForFresh( ((VarExprModifierPragma)m).expr )) {
//System.out.println("SIGNALS ALLOCATES " + rd.id() + " " + Location.toString(m.getStartLoc()));
              return true;
            }
          }
        }
        return false;
    }

    private boolean checkForFresh(ExprVec ev) {
        for (int i=0; i<ev.size(); ++i) {
          if (checkForFresh(ev.elementAt(i))) return true;
        }
        return false;
    }
    private boolean checkForFresh(Expr e) {
      if (e == null) return false;
      else if (e instanceof BinaryExpr) {
        BinaryExpr be = (BinaryExpr)e;
        return checkForFresh(be.left) || checkForFresh(be.right);
      } else if (e instanceof UnaryExpr) {
        UnaryExpr ue = (UnaryExpr)e;
        return checkForFresh(ue.expr);
      } else if (e instanceof NaryExpr) {
        NaryExpr ne = (NaryExpr)e;
        if (ne.op == TagConstants.FRESH) return true;
        return checkForFresh(ne.exprs);
      } else if (e instanceof MethodInvocation) {
        MethodInvocation mi = (MethodInvocation)e;
        if (mi.od instanceof ExprObjectDesignator) {
          if ( checkForFresh( ((ExprObjectDesignator)mi.od).expr )) return true;
        }
        return checkForFresh(mi.args);
      } else if (e instanceof ArrayRefExpr) {
        return checkForFresh( ((ArrayRefExpr)e).index) ||
               checkForFresh( ((ArrayRefExpr)e).array) ;
      } else if (e instanceof CondExpr) {
        return checkForFresh( ((CondExpr)e).test ) ||
               checkForFresh( ((CondExpr)e).thn  ) ||
               checkForFresh( ((CondExpr)e).els  );
      } else if (e instanceof CastExpr) {
        return checkForFresh( ((CastExpr)e).expr );
      } else if (e instanceof ParenExpr) {
        return checkForFresh( ((ParenExpr)e).expr );
      } else if (e instanceof LabelExpr) {
        return checkForFresh( ((LabelExpr)e).expr );
      } else if (e instanceof FieldAccess) {
        FieldAccess fa = (FieldAccess)e;
        if (fa.od instanceof ExprObjectDesignator) {
          if ( checkForFresh( ((ExprObjectDesignator)fa.od).expr )) return true;
        }
        return false;
      } else if (e instanceof NewArrayExpr) {
        NewArrayExpr nae = (NewArrayExpr)e;
        return checkForFresh( nae.dims );
         // FIXME - what about array initializers?
      } else if (e instanceof NewInstanceExpr) {
        NewInstanceExpr nie = (NewInstanceExpr)e;
        return checkForFresh( nie.args );
      } else if (e instanceof SetCompExpr) {
        return checkForFresh( ((SetCompExpr)e).expr );
      } else if (e instanceof LiteralExpr) {
        return false;
      } else if (e instanceof VariableAccess) {
        return false;
      } else if (e instanceof ThisExpr) {
        return false;
      } else if (e instanceof ResExpr) {
        return false;
      } else if (e instanceof escjava.ast.TypeExpr) {
        return false;
      } else if (e instanceof LockSetExpr) {
        return false;
      } else if (e instanceof NotModifiedExpr) {
        return false;
      } else if (e instanceof InstanceOfExpr) {
        return false;
      } else if (e instanceof ClassLiteral) {
        return false;
      } else if (e instanceof QuantifiedExpr) {
        return checkForFresh( ((QuantifiedExpr)e).expr) ||
               checkForFresh( ((QuantifiedExpr)e).rangeExpr) ;
      } else if (e instanceof GeneralizedQuantifiedExpr ) {
        return checkForFresh( ((GeneralizedQuantifiedExpr)e).expr) ||
               checkForFresh( ((GeneralizedQuantifiedExpr)e).rangeExpr) ;
      } else if (e instanceof NumericalQuantifiedExpr) {
        return checkForFresh( ((NumericalQuantifiedExpr)e).expr) ||
               checkForFresh( ((NumericalQuantifiedExpr)e).rangeExpr) ;
      } else {
        System.out.println("CLASS " + e.getClass());
        return true;
      }
    }
  };
  public static boolean isAllocates(RoutineDecl rd) {
    return allocatesDecoration.isTrue(rd);
  }

  private static final BooleanDecoration functionDecoration = new BooleanDecoration("function") {
    public boolean calculate(ASTNode o) {
      RoutineDecl rd = (RoutineDecl)o;
      if (findModifierPragma(rd.pmodifiers,TagConstants.FUNCTION)
          != null) return true;
      if (!isPure(rd)) return false;
      // If non-static, the owning class must be immutable
      // But it can't override non-function methods
      // Constructors don't depend on the owning class
      if (!Modifiers.isStatic(rd.modifiers) && (rd instanceof MethodDecl)) {
        if ( ! isImmutable(rd.parent) ) return false;
        if (isAllocates(rd)) return false;
        javafe.util.Set ov = escjava.tc.FlowInsensitiveChecks.getAllOverrides((MethodDecl)rd);
        Enumeration i = ov.elements();
        while (i.hasMoreElements()) {
           if (!isFunction( (MethodDecl)i.nextElement() )) return false;
        }
      }
      // All argument types must be primitive or immutable
      FormalParaDeclVec args = rd.args;
      for (int i=0; i<args.size(); ++i) {
        FormalParaDecl f = args.elementAt(i);
        Type t = f.type;
        if (t instanceof TypeName) t = TypeSig.getSig((TypeName)t);
        if (t instanceof PrimitiveType) continue;
        if (t instanceof ArrayType) return false;
        if (t instanceof TypeSig) {
          if (! isImmutable(((TypeSig)t).getTypeDecl())) return false;
        }
      }
      return true;
    }
  };
public static boolean isFunction(RoutineDecl rd) {
  return functionDecoration.isTrue(rd);
}


  public static final ASTDecoration exceptionDecoration = 
                                     new ASTDecoration("originalExceptions");

  public static final ASTDecoration axiomDecoration = new ASTDecoration("axioms");

  public static final ASTDecoration representsDecoration =
    new ASTDecoration("representsClauses");

  public static final ASTDecoration owningDecl = new ASTDecoration("owningDecl");
  public static final ASTDecoration allSpecs = new ASTDecoration("allSpecs");

  static public ModifierPragmaVec getAllSpecs(RoutineDecl rd) {
    Object o = allSpecs.get(rd);
    if (o != null) return (ModifierPragmaVec)o;
    ModifierPragmaVec result = ModifierPragmaVec.make();
    allSpecs.set(rd,result);
    result.append(rd.pmodifiers);
    if (rd instanceof ConstructorDecl) return result;
    MethodDecl rrd = (MethodDecl)rd;
    Type[] argtypes = rd.argTypes();
    TypeSig td = TypeSig.getSig(rd.parent);
    java.util.Iterator i = td.superInterfaces().iterator();
    td = td.superClass();
    while (td != null) {
      MethodDecl md = td.hasMethod(rrd.id, argtypes);
      if (md != null) result.append(md.pmodifiers);
      td = td.superClass();
    }

    while (i.hasNext()) {
      td = (TypeSig)i.next();
      MethodDecl md = td.hasMethod(rrd.id, argtypes);
      if (md != null) result.append(md.pmodifiers);
    }
	
    return result;
  }
  
  public static final ASTDecoration inheritedSpecs = new ASTDecoration("inheritedSpecs");

  static public ModifierPragmaVec getInheritedSpecs(RoutineDecl rd) {
    IdentifierNode fullName = IdentifierNode.make(escjava.translate.TrAnExpr.fullName(rd,false));
    Object o = inheritedSpecs.get(fullName);
    if (o != null) return (ModifierPragmaVec)o;
    ModifierPragmaVec result = ModifierPragmaVec.make();
    inheritedSpecs.set(fullName,result);
    return result;
  }
  
  static public ModifierPragmaVec addInheritedSpecs(RoutineDecl rd, ModifierPragmaVec mp) {
    ModifierPragmaVec mpv = Utils.getInheritedSpecs(rd);
    mpv.append(mp);
    ExprModifierPragma req = null;
    int index = 0;
    for (int i=0; i<mpv.size(); ++i) {
      ModifierPragma m = mpv.elementAt(i);
      if (m.getTag() != TagConstants.REQUIRES) continue;
      if (req == null) { req = (ExprModifierPragma)m; index = i; continue; }
      req = escjava.AnnotationHandler.or(req,(ExprModifierPragma)m);
      mpv.setElementAt(req,index);
      mpv.removeElementAt(i);
      --i;
    }
    return mpv;
  }

}

