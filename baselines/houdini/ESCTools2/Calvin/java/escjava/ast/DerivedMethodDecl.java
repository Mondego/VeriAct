/* Copyright Hewlett-Packard, 2002 */

package escjava.ast;


import java.util.Hashtable;

import javafe.ast.*;
import javafe.util.Assert;
import javafe.util.Location;
import escjava.Main;
import escjava.translate.Substitute;


/** This class represents the intermediate method declaration/specification
 ** structure used in escjava.translate.GetSpec as described in
 ** section 7 of ESCJ 16.
 **/

public class DerivedMethodDecl {
    /*@ non_null */ public RoutineDecl original;

    public FormalParaDeclVec args;
    public Type returnType;
    public TypeNameVec throwsSet;
    public boolean usesFresh;
    public ExprModifierPragmaVec requires;
    public ExprModifierPragmaVec modifies;
    public ExprModifierPragmaVec ensures;
    public VarExprModifierPragmaVec exsures;
    public PerformsModifierPragma performs; // sanjit add

    public DerivedMethodDecl(/*@ non_null */ RoutineDecl rd) {
	original = rd;
    }

    //@ ensures RES == (original instanceof ConstructorDecl);
    public boolean isConstructor() {
	return original instanceof ConstructorDecl;
    }

    // The following method should be called only when generating a spec
    // for a body, in which case it is known that "body" is not null.
    //@ requires original.body != null;
    //@ ensures RES ==> (original instanceof ConstructorDecl);
    public boolean isConstructorThatCallsSibling() {
	if (! isConstructor()) {
	    return false;
	}
	if (original.body.getTag() == TagConstants.BLOCKSTMT) {
	    GenericBlockStmt body = (GenericBlockStmt)original.body;
	    if (1 <= body.stmts.size()) {
		Stmt s0 = body.stmts.elementAt(0);
		if (s0.getTag() == TagConstants.CONSTRUCTORINVOCATION) {
		    ConstructorInvocation ccall = (ConstructorInvocation)s0;
		    if (ccall.decl.parent == original.parent) {
			return true;
		    }
		}
	    }
	}
	return false;
    }

    public boolean isStaticMethod() {
	return original instanceof MethodDecl &&
	    Modifiers.isStatic(original.modifiers);
    }
  
    public boolean isInstanceMethod() {
	return original instanceof MethodDecl &&
	    !Modifiers.isStatic(original.modifiers);
    }

    public boolean isSynchronized() {
	return Modifiers.isSynchronized(original.modifiers);
    }
  
    public Identifier getId() {
	if (original instanceof MethodDecl)
	    return ((MethodDecl)original).id;
	else
	    return original.parent.id;
    }

    public TypeDecl getContainingClass() {
	return original.parent;
    }

    public int getRoutineDeclStartLoc() {
	return original.getStartLoc();
    }

    public int getRoutineDeclEndLoc() {
	return original.getEndLoc();
    }

    public void computeFreshUsage() {
	usesFresh = false;
	if (!Main.allocUseOpt) {
	    // continue as if "fresh" (and hence "alloc") were used
	    usesFresh = true;
	    return;
	}
	if (original instanceof ConstructorDecl) {
	    // constructors always need to be considered as mentioning "fresh"
	    usesFresh = true;
	    return;
	}
	int n = ensures.size();
	for (int i = 0; i < n; i++) {
	    ExprModifierPragma emp = ensures.elementAt(i);
	    if (Substitute.mentionsFresh(emp.expr)) {
		usesFresh = true;
		return;
	    }
	}
	n = exsures.size();
	for (int i = 0; i < n; i++) {
	    VarExprModifierPragma vemp = exsures.elementAt(i);
	    if (Substitute.mentionsFresh(vemp.expr)) {
		usesFresh = true;
		return;
	    }
	}

	if (performs != null) {
	    usesFresh = mentionsFresh(performs.stmt);
	}
    }    
    
    private boolean mentionsFresh(PerformsStmt amp) {
	switch (amp.getTag()) {
	  case TagConstants.ACTION: {
	      PerformsAction samp = (PerformsAction)amp;
	      return Substitute.mentionsFresh(samp.pred);
	  }
	  case TagConstants.SEMICOLON: {
	      PerformsSequence samp = (PerformsSequence)amp;
	      int i;
	      for (i = 0; i < samp.seq.size(); i++) {
		  if (mentionsFresh(samp.seq.elementAt(i))) {
		      return true;
		  }
	      }
	      return false;
	  }
	  case TagConstants.CHOICE: {
	      PerformsChoice samp = (PerformsChoice)amp;
	      return (mentionsFresh(samp.left) || mentionsFresh(samp.right));
	  }
	  default: {
	      Assert.notFalse(false);
	      return false;
	  }
	}
    }
}
