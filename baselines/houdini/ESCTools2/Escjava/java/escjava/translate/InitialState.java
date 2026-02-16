/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;

import javafe.ast.*;
import javafe.tc.TypeSig;
import javafe.util.Location;

import escjava.ast.GhostDeclPragma;
import escjava.ast.TagConstants;
import escjava.ast.NaryExpr;

import escjava.backpred.FindContributors;


/**
 ** This class provides two methods used in the generation of a
 ** verification condition for a method or constructor (see section 8
 ** of ESCJ 16). <p>
 **/

public final class InitialState {
  private /*@ non_null @*/ Hashtable premap;
  private /*@ non_null @*/ Expr is;

  public InitialState(/*@ non_null @*/ FindContributors scope) {
    premap = new Hashtable();
    ExprVec conjuncts = ExprVec.make();

    // static fields and instance variables
    Enumeration fields = scope.fields();
    while (fields.hasMoreElements()) {
	  FieldDecl fd = (FieldDecl)fields.nextElement();
	  
	  VariableAccess va = TrAnExpr.makeVarAccess(fd, Location.NULL);
	  VariableAccess variant = addMapping(fd);

	  // g@pre == g    and    f@pre == f
	  conjuncts.addElement(GC.nary(TagConstants.ANYEQ, variant, va));
	  Expr typeCorrect;
	  if (Modifiers.isStatic(fd.modifiers)) {
	    // TypeCorrect[[ g ]]
	    typeCorrect = TrAnExpr.typeCorrect(fd);
	  } else {
	    // FieldTypeCorrect[[ f ]]
	    typeCorrect = TrAnExpr.fieldTypeCorrect(fd);
	  }
	  conjuncts.addElement(typeCorrect);
    }

	/* The preFields Set accumulates every location that occurs inside a \old()
	   construct.  These are the mappings needed to map variable uses that occur
	   in expressions back to a token representing the pre-state value.
	   This should be all that is needed (though it is overkill for any one method),
	   and we should not need the set added above (which are those locations that
	   appear in modifies clauses), but we keep those for good measure (to avoid
	   introducing bugs because some kind of access does not make it into a proper
	   representation in the loop below).
	   By using all the \old() locations we (a) make the verification of method bodies
	   robust against errors in the modifies clauses and (b) allow the implementation
	   of modifies \everything.
        */
    fields = scope.preFields.elements();
    while (fields.hasMoreElements()) {
	  FieldDecl fd = ((FieldAccess)fields.nextElement()).decl;
	  if (premap.get(fd) != null) continue;
	  
	  VariableAccess va = TrAnExpr.makeVarAccess(fd, Location.NULL);
	  VariableAccess variant = addMapping(fd);

	  // g@pre == g    and    f@pre == f
	  conjuncts.addElement(GC.nary(TagConstants.ANYEQ, variant, va));
	  Expr typeCorrect;
	  if (Modifiers.isStatic(fd.modifiers)) {
	    // TypeCorrect[[ g ]]
	    typeCorrect = TrAnExpr.typeCorrect(fd);
	  } else {
	    // FieldTypeCorrect[[ f ]], including non-null.
	    typeCorrect = TrAnExpr.fieldTypeCorrect(fd);
	  }
	  conjuncts.addElement(typeCorrect);
    }

    // elems@pre == elems
    conjuncts.addElement(GC.nary(TagConstants.ANYEQ,
				 addMapping(GC.elemsvar.decl), GC.elemsvar));
    // ElemsTypeCorrect[[ elems ]]
    conjuncts.addElement(TrAnExpr.elemsTypeCorrect(GC.elemsvar.decl));

    // LS == asLockSet(LS)
    conjuncts.addElement(GC.nary(TagConstants.ANYEQ, GC.LSvar,
				 GC.nary(TagConstants.ASLOCKSET, GC.LSvar)));

    // alloc@pre == alloc
    conjuncts.addElement(GC.nary(TagConstants.ANYEQ,
				 addMapping(GC.allocvar.decl), GC.allocvar));

    // state@pre == state
    conjuncts.addElement(GC.nary(TagConstants.ANYEQ,
				 addMapping(GC.statevar.decl), GC.statevar));

    // conjoin the conjuncts
    is = GC.and(conjuncts);
  }

  private /*@ non_null */ VariableAccess addMapping(/*@ non_null */ GenericVarDecl vd) {
    VariableAccess variant = GetSpec.createVarVariant(vd, "pre");
    premap.put(vd, variant);
    return variant;
  }
  
  public /*@ non_null @*/ Hashtable getPreMap() {
    return premap;
  }
  
  public /*@ non_null @*/ Expr getInitialState() {
    return is;
  }
}
