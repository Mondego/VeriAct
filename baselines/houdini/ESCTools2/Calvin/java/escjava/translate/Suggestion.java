/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

import java.util.Enumeration;
import javafe.util.Assert;
import javafe.util.Location;
import javafe.util.Set;
import javafe.util.Location;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.tc.FlowInsensitiveChecks;
import escjava.prover.*;

/** This class generates possible ways to annotate a program to eliminate
  * a given warning.<p>
  *
  * NOTE:  The syntax of the strings produced in this class must be kept
  * in sync with what is expected by the script wizfilter.perl.
  **/

class Suggestion {
  static String generate(int warningTag, Object auxInfo,
			 /*@ non_null */ RoutineDecl rd,
			 /*@ non_null */ Set directTargets,
			 /*@ non_null */ SList cc, int locDecl) {
      switch (warningTag) {
      case TagConstants.CHKNULLPOINTER:
      case TagConstants.CHKNONNULL:
	  {
	      VarInit E = (VarInit)auxInfo;
	      // CF: bug workaround
	      if (E==null) return null;
	      Assert.notNull(E);
	      return gNull(E, rd);
	  }
	  
      case TagConstants.CHKNONNULLINIT:
	  return "add an initializer or initialize the field in constructor";
	  
      case TagConstants.CHKINDEXNEGATIVE:
      case TagConstants.CHKNEGATIVEARRAYSIZE:
	  {
	      VarInit E = (VarInit)auxInfo;
	      Assert.notNull(E);
	      return gNeg(E, rd, directTargets);
	  }
	  
      case TagConstants.CHKARRAYSTORE:
	  {
	      VarInit E = (VarInit)auxInfo;
	      Assert.notNull(E);
	      return gArrStore(E, rd);
	  }
      case TagConstants.CHKOBJECTINVARIANT:
	  {
	      if (auxInfo == null)
		  return null;
	      else {
		  Expr E = (Expr)auxInfo;
		  return gInvariant(E, rd, cc, locDecl);
	      }
	  }
      default:
	  break;
      }
      return null;  // no suggestion
  }

  private static String gNull(/*@ non_null */ VarInit E,
			      /*@ non_null */ RoutineDecl rd) {
    switch (E.getTag()) {
      case TagConstants.FIELDACCESS:
	{
	  FieldDecl fd = ((FieldAccess)E).decl;
	  String description;
	  if (Modifiers.isStatic(fd.modifiers)) {
	    description = "static field ";
	  } else {
	    if (rd instanceof ConstructorDecl && fd.parent == rd.parent) {
	      return "none <instance field in constructor>";
	    }
	    description = "instance field ";
	  }
	  return "perhaps declare " + description + name(fd.id, fd.locId) +
	    " with 'non_null'";
	}

      case TagConstants.VARIABLEACCESS:
	{
	  GenericVarDecl vd = ((VariableAccess)E).decl;
	  if (vd.getTag() == TagConstants.FORMALPARADECL) {
	    if (FlowInsensitiveChecks.isMethodOverride(rd)) {
	      MethodDecl md = (MethodDecl)rd;

	      // Find index of parameter in method's signature
	      int pi = 0;
	      while (md.args.elementAt(pi) != vd) {
		pi++;
	      }
	      // Find the corresponding parameter in the original method decl
	      MethodDecl mdOrig = getOriginalMethod(md);
	      GenericVarDecl vdSuper = mdOrig.args.elementAt(pi);

	      String s = "perhaps declare parameter " +
		name(vdSuper.id, vdSuper.locId) +
		" with 'non_null'";
	      String n0 = vdSuper.id.toString();
	      String n1 = vd.id.toString();
	      if (!n0.equals(n1)) {
		s += " (the parameter is known as '" + n1 +
		  "' in the method override in class " + md.parent.id + ")";
	      }
	      return s;
	    } else {
	      return "perhaps declare parameter " + name(vd.id, vd.locId) +
		" with 'non_null'";
	    }
	  } else if (vd.getTag() == TagConstants.LOCALVARDECL) {
	    return "perhaps declare local variable " + name(vd.id, vd.locId) +
	      " with 'non_null'";
	  } else {
	    return "none <unknown variable>";
	  }
	}

      case TagConstants.METHODINVOCATION:
	{
	  MethodDecl md =((MethodInvocation)E).decl;
	  if (FlowInsensitiveChecks.isMethodOverride(md)) {
	    return "perhaps declare method override " + name(md.id, md.locId) +
	      " with 'also_ensures \\result != null;' " +
	      "(or the overridden method with 'ensures \\result != null;')";
	  } else if (md instanceof MethodDecl) {
	    return "perhaps declare method " + name(md.id, md.locId) +
	      " with 'ensures \\result != null;'";
	  } else {
	    Assert.fail("unexpected routine returns possibly-null value");
	    return null;
	  }
	}

      case TagConstants.NULLLIT:
	return "none <null>";

      default:
	return "none <intimidating expression>";
    }
  }

  private static String gNeg(/*@ non_null */ VarInit E,
			     /*@ non_null */ RoutineDecl rd,
			     /*@ non_null */ Set directTargets) {
    switch (E.getTag()) {
      case TagConstants.FIELDACCESS:
	{
	  FieldDecl fd = ((FieldAccess)E).decl;
	  String description;
	  if (Modifiers.isStatic(fd.modifiers)) {
	    description = "static invariant ";
	  } else {
	    description = "instance invariant ";
	  }
	  if (directTargets.contains(fd)) {
	    return "none <" +
	      (Modifiers.isStatic(fd.modifiers) ? "static" : "instance") +
	      " field is direct target>";
	  } else {
	    return "perhaps declare " + description + "'invariant 0 <= " +
	      fd.id + ";' in class " + fd.parent.id +
	      " (near " + name(fd.id, fd.locId) + ")";
	  }
	}

      case TagConstants.VARIABLEACCESS:
	{
	  GenericVarDecl vd = ((VariableAccess)E).decl;
	  if (vd.getTag() == TagConstants.FORMALPARADECL) {
	    if (directTargets.contains(vd)) {
	      return "none <parameter is direct target>";
	    } else if (FlowInsensitiveChecks.isMethodOverride(rd)) {
	      MethodDecl md = (MethodDecl)rd;
	      MethodDecl mdOrig = getOriginalMethod(md);
	      return "perhaps declare overridden method " +
		name(mdOrig.id, mdOrig.locId) +
		" with 'requires 0 <= " + vd.id + ";'";
	    } else if (rd instanceof MethodDecl) {
	      MethodDecl md = (MethodDecl)rd;
	      return "perhaps declare method " + name(md.id, md.locId) +
		" with 'requires 0 <= " + vd.id + ";'";
	    } else {
	      return "perhaps declare constructor " +
		name(rd.parent.id, rd.locId) +
		" with 'requires 0 <= " + vd.id + ";'";
	    }
	  } else if (vd.getTag() == TagConstants.LOCALVARDECL) {
	    return "none <local variable>";
	  } else {
	    return "none <unknown variable type>";
	  }
	}

      case TagConstants.METHODINVOCATION:
	{
	  MethodDecl md =((MethodInvocation)E).decl;
	  if (FlowInsensitiveChecks.isMethodOverride(md)) {
	    return "perhaps declare method override " + name(md.id, md.locId) +
	      " with 'also_ensures 0 <= \\result;' " +
	      "(or the overridden method with 'ensures 0 <= \\result;')";
	  } else if (md instanceof MethodDecl) {
	    return "perhaps declare method " + name(md.id, md.locId) +
	      " with 'ensures 0 <= \\result;'";
	  } else {
	    Assert.fail("unexpected routine returns possibly-negative value");
	    return null;
	  }
	}
      
      default:
	return "none <big expression>";
    }
  }


  private static String gArrStore(/*@ non_null */ VarInit E,
				  /*@ non_null */ RoutineDecl rd) {
      switch (E.getTag()) {
      case TagConstants.FIELDACCESS:
	  {
	      FieldDecl fd = ((FieldAccess)E).decl;
	      String description;
	      if (Modifiers.isStatic(fd.modifiers)) {
		  description = "static invariant ";
	      } else {
		  description = "instance invariant ";
	      }
	      Assert.notFalse(fd.type instanceof ArrayType);
	      return "perhaps declare " + description + 
		  "'invariant \\elemtype(\\typeof(" + fd.id + 
		  ")) == \\type(" + typeName(((ArrayType) fd.type).elemType) 
		  + ");' in class " + fd.parent.id +
		  " (near " + name(fd.id, fd.locId) + ")";      
	  }
	  
      case TagConstants.VARIABLEACCESS:
	  {
	      GenericVarDecl vd = ((VariableAccess)E).decl;
	      if (vd.getTag() == TagConstants.FORMALPARADECL) {
		  MethodDecl md = (MethodDecl)rd;
		  if (FlowInsensitiveChecks.isMethodOverride(rd)) {
		      
		      // Find index of parameter in method's signature
		      int pi = 0;
		      while (md.args.elementAt(pi) != vd) {
			  pi++;
		      }
		      /* Find the corresponding parameter in the 
			 original method decl */
		      MethodDecl mdOrig = getOriginalMethod(md);
		      GenericVarDecl vdSuper = mdOrig.args.elementAt(pi);

		      String n0 = vdSuper.id.toString();
		      String n1 = vd.id.toString();
		      
		      Assert.notFalse(vdSuper.type instanceof ArrayType);
		      String s = "perhaps declare overridden method " +
			  name(mdOrig.id, mdOrig.locId) +
			  " with 'requires \\elemtype(\\typeof(" + n0 +
			  ")) == \\type(" + 
			  typeName(((ArrayType) vdSuper.type).elemType) +
			  ");'";
		      if (!n0.equals(n1)) {
			  s += " (the parameter is known as '" + n1 +
			      "' in the method override in class " + 
			      md.parent.id + ")";
		      }
		      return s;
		  } else {
		      Assert.notFalse(vd.type instanceof ArrayType);
		      return "perhaps declare method " + 
			  name(md.id, md.locId) +
			  " with 'requires \\elemtype(\\typeof(" + vd.id 
			  + ")) == \\type(" 
			  + typeName(((ArrayType) vd.type).elemType)
			  + ");'";
			  
		  }
	      } else if (vd.getTag() == TagConstants.LOCALVARDECL) {
		  return "none <local variable>";
	      } else {
		  return "none <unknown variable>";
	      }
	  }
	  
      case TagConstants.METHODINVOCATION:
	  {
	      MethodDecl md =((MethodInvocation)E).decl;
	      if (FlowInsensitiveChecks.isMethodOverride(md)) {
		  Assert.notFalse(md.returnType instanceof ArrayType);
		  String returnType = typeName(((ArrayType) md.returnType).elemType);
		  return "perhaps declare method override " + name(md.id, md.locId) +
		      " with 'also_ensures \\elemtype(\\typeof(\\result)) == " +
		      "\\type(" + returnType + ");' " +
		      "(or the overridden method with " +
		      "'ensures \\elemtype(\\typeof(\\result)) == \\type(" + 
		      returnType + ");')";
	      } else if (md instanceof MethodDecl) {
		  Assert.notFalse(md.returnType instanceof ArrayType);
		  return "perhaps declare method " + 
		      name(md.id, md.locId) +
		      " with 'ensures \\elemtype(\\typeof(\\result)) == \\type(" +
		      typeName(((ArrayType) md.returnType).elemType) + 
		      ");'";
	      } else {
		  Assert.fail("unexpected routine returns possibly-null value");
		  return null;
	      }
	  }
	  
      case TagConstants.NULLLIT:
	  return "none <null>";
	  
      default:
	  return "none <intimidating expression>";
      } 
  } 


  private static String gInvariant(/*@ non_null */ Expr E,
				   /*@ non_null */ RoutineDecl rd,
				   /*@ non_null */ SList cc,
						   int locDecl) {
      if (brokenObjIsMadeUp(cc)) {
	  Set inj = possiblyInjectiveFields(E);
	  if (inj == null)
	      return null;
	  else if (inj.size() != 1)
	      return null;
	  else {
	      Enumeration els = inj.elements();
	      String injField = ((Identifier) els.nextElement()).toString();
	      return "perhaps declare instance invariant 'invariant this." + 
		  injField + ".owner == this;' in class " + rd.parent.id + 
		  " (near associated declaration at " + 
		  Location.toString(locDecl) + ")";
	  }
      }
      else
	  return null;
  }



  /** Returns a method that <code>md</code> overrides.  If <code>md</code>
    * overrides a method in a class, then that method is returned.  Otherwise,
    * any one of the overrides is returned.
    **/

  static MethodDecl getOriginalMethod(/*@ non_null */ MethodDecl md) {
    MethodDecl orig = md;
    while (true) {
      MethodDecl mdSuper = FlowInsensitiveChecks.getSuperClassOverride(orig);
      if (mdSuper == null) {
	return orig;
      }
      orig = mdSuper;
    }
  }


  //@ ensures RES != null;
  private static String name(/*@ non_null */ Identifier id, int loc) {
    String s = "'" + id + "'";
    if (!Location.isWholeFileLoc(loc)) {
      s += " at " + Location.toLineNumber(loc) + "," + Location.toColumn(loc);
    }
    s += " in " + Location.toFileName(loc);
    return s;
  }


  //@ ensures RES != null;
  private static String typeName(/*@ non_null */ Type type) {
      String name;
      if (type instanceof PrimitiveType) {
	  PrimitiveType pt = (PrimitiveType) type;
	  name = javafe.ast.TagConstants.toString(pt.tag);
	  name = name.substring(name.length() - 4).toLowerCase();
      }
      else if (type instanceof TypeName) {
	  TypeName tn = (TypeName) type;
	  name = tn.name.printName();
      }
      else if (type instanceof ArrayType) {
	  ArrayType at = (ArrayType) type;
	  String elemName = typeName(at.elemType);
	  name = elemName + "[]";
      }
      else {
	  javafe.util.Assert.fail("Unknown kind of Type");
	  name = "";
      }
      return name;
  }


    /* Returns true if the counterexample context does not say that
       brokenObj is equal to some program variable, and false otherwise.
       This tells us whether or not Simplify had to dream up the broken
       object, which is a strong indicator of an injectivity problem.
    */
    private static boolean brokenObjIsMadeUp(/*@ non_null */ SList cc) {
	int n = cc.length();
	try {
	    for (int i = 0; i < n; i++) {
		SExp cur = cc.at(i);
		if (cur.isList()) {
		    SList curL = (SList) cur;
		    if (curL.length() == 3) {
			if (curL.at(0).toString().equals("EQ") &&
			    (curL.at(1).toString().startsWith("brokenObj") 
			     ||
			     curL.at(2).toString().startsWith("brokenObj")))
			    return false;
		    }
		}
	    }
	}
	catch (SExpTypeError e) {
	    Assert.fail("Out of bounds SList access");
	}
	return true;	
    }

    
    /* Finds fields that potentially need to be declared injective.  This
       routine simply searches for fields f in an expression of the form
       brokenObj.f.f' or brokenObj.f[i]. */
    private static Set possiblyInjectiveFields(/*@ non_null */ Expr e) {
	if (e instanceof LabelExpr)
	    return possiblyInjectiveFields(((LabelExpr) e).expr);
	else if (e instanceof QuantifiedExpr)
	    return possiblyInjectiveFields(((QuantifiedExpr) e).expr);
	else if (e.getTag() == TagConstants.SELECT) {
	    ExprVec exprs = ((NaryExpr) e).exprs;
	    Expr first = exprs.elementAt(0);
	    Expr second = exprs.elementAt(1);
	    // check for brokenObj.f.f' or this.f.f' (in case the brokenObj
	    // substitution hasn't already been performed for some reason)
	    if (first.getTag() == TagConstants.VARIABLEACCESS &&
		second.getTag() == TagConstants.SELECT) {
		ExprVec sexprs = ((NaryExpr) second).exprs;
		Expr sec0 = sexprs.elementAt(0);
		Expr sec1 = sexprs.elementAt(1);
		if (sec0.getTag() == TagConstants.VARIABLEACCESS &&
		    sec1.getTag() == TagConstants.VARIABLEACCESS) {
		    String name = ((VariableAccess)sec1).id.toString();
		    if (name.startsWith("brokenObj") || name.equals("this")) {
			Set set = new Set();
			set.add(((VariableAccess)sec0).id);
			return set;
		    }
		    else
			return null;
		}
		else
		    return null;
	    }
	    // check for brokenObj.f[i] or this.f[i] (in case the brokenObj
	    // substitution hasn't already been performed for some reason)
	    else if (first.getTag() == TagConstants.SELECT) {
		ExprVec fexprs = ((NaryExpr) first).exprs;
		Expr fir0 = fexprs.elementAt(0);
		Expr fir1 = fexprs.elementAt(1);
		Set set = new Set();
		if (fir0.getTag() == TagConstants.VARIABLEACCESS &&
		    fir1.getTag() == TagConstants.SELECT) {
		    Assert.notFalse(((VariableAccess)fir0).id.toString().equals("elems"));
		    ExprVec f1exprs = ((NaryExpr) fir1).exprs;
		    Expr fir10 = f1exprs.elementAt(0);
		    Expr fir11 = f1exprs.elementAt(1);
		    if (fir10.getTag() == TagConstants.VARIABLEACCESS 
			&& fir11.getTag() == TagConstants.VARIABLEACCESS) {
			String name = ((VariableAccess)fir11).id.toString();
			if (name.startsWith("brokenObj") || name.equals("this")) 
			    set.add(((VariableAccess) fir10).id);
			else
			    return null;
		    }
		    else
			return null;
		}
		else
		    return null;
		Set res = possiblyInjectiveFields(second);
		Enumeration els = res.elements();
		while (els.hasMoreElements()) {
		    set.add(els.nextElement());
		}
		if (set.size() > 1)
		    return null;
		return set;
	    }
	    // neither special case applies, so do regular NaryExpr processing
	    else
		return checkNaryExpr((NaryExpr) e);
	}
	else if (e instanceof NaryExpr)	
	    return checkNaryExpr((NaryExpr) e);
	else if (e instanceof TypeExpr || e instanceof VariableAccess ||
		 e instanceof LiteralExpr)
	    return new Set();
	else if (e instanceof SubstExpr) {
	    // check that this is a substitution of brokenObj for this
	    SubstExpr s = (SubstExpr) e;
	    Assert.notFalse(s.var.id.toString().equals("this"));
	    Assert.notFalse(s.val instanceof VariableAccess);
	    Assert.notFalse(((VariableAccess) s.val).id.toString().startsWith("brokenObj"));
	    return possiblyInjectiveFields(s.target);
	}
	else {
	    Assert.fail("Unexpected Expr encountered");
	    return null;
	}
    }
    
    /* A helper function for possiblyInjectiveFields that checks for injective
       fields in NaryExprs. */
    private static Set checkNaryExpr(/*@ non_null */ NaryExpr e) {
	ExprVec exprs = e.exprs;
	int size = exprs.size();
	Set total = new Set();
	for (int i = 0; i < size; i++) {
	    Set cur = possiblyInjectiveFields(exprs.elementAt(i));
	    if (cur == null)
		return null;
	    Enumeration els = cur.elements();
	    while (els.hasMoreElements()) {
		total.add(els.nextElement());
	    }
	    if (total.size() > 1)
		return null;
	}
	return total;
    }
}
