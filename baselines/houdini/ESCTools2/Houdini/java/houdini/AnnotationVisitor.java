/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 1999, Compaq Computer Corporation
// Authors:  Cormac Flanagan and Rustan Leino.
// Change history:
//      Sep 1999  rustan & flanagan  Created
//   17 Sep 2000  flanagan           Put helper annotations on private methods
//                                   called only from constructors
//   21 Sep 2000  flanagan           Put final modifiers on non-overridden methods

package houdini;

import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;

import escjava.tc.FlowInsensitiveChecks;


public class AnnotationVisitor extends DefaultVisitor {
    
    /*@ non_null */ AnnotationGuesser guesser;
    /*@ non_null */ Hashtable envBase;
    /*@ non_null */ Set methodsCalledFromMethods = new Set();
    /*@ non_null */ Set overriddenMethods = new Set();
    
    public AnnotationVisitor(/*@ non_null */ AnnotationGuesser guesser) {
	this.guesser = guesser;
    } //@ nowarn NonNullInit;

    public void visitTypeDeclVec(Vector tdv) {

	overriddenMethodVisitor nfmv = new overriddenMethodVisitor(overriddenMethods);
	for(Enumeration e=tdv.elements(); e.hasMoreElements(); ) {
	    nfmv.visitTypeDecl( (TypeDecl)e.nextElement() );
	}
	for(Enumeration e=tdv.elements(); e.hasMoreElements(); ) {
	    visitTypeDecl( (TypeDecl)e.nextElement() );
	}
    }

    public void visitTypeDecl(TypeDecl x) {
	envBase = new Hashtable();
	envBase.put("0", Types.intType);
	envBase.put("1", Types.intType);
	envBase.put("2", Types.intType);
	envBase.put("3", Types.intType);
	envBase.put("-1", Types.intType);

	ConstantDimensionVisitor cdv = new ConstantDimensionVisitor(envBase);
	x.accept(cdv);

	x.accept(new MethodCalledFromMethodVisitor(methodsCalledFromMethods));

	boolean hasConstructor = false;
	super.visitTypeDecl(x);
	for(int i=0; i<x.elems.size(); i++) {
	    TypeDeclElem e = x.elems.elementAt(i);
	    if( e instanceof ConstructorDecl ) {
		if( ((ConstructorDecl)e).getStartLoc() == x.locOpenBrace 
		    && !x.id.toString().equals("Object")) {
		    // Explicate default constructor, mark as helper
		    new Annotator(">", x.locOpenBrace, "{", "defaultconstructor", "")
			.put("*/public "+x.id.toString()+"(){}/* Explicating default constructor here");
							      
		}
	    }
	}

    }
  
    public void visitRoutineDecl(RoutineDecl rd) {

	guesser.guessThrows(rd);
	int kind;
	String ms;
	Identifier routineName;
	if (! (rd instanceof MethodDecl)) {
	    ms = "constructor";
	    kind = 0;
	    //@ assume rd.parent != null;
	    routineName = rd.parent.id;
	} else {
	    MethodDecl md = (MethodDecl)rd;
	    routineName = md.id;
	    if (Modifiers.isStatic(md.modifiers)) {
		ms = "static method";
		kind = 1;
		if (md.id.toString().equals("main") && md.args.size() == 1) {
		  GenericVarDecl decl = rd.args.elementAt(0);
		  if (AnnotationGuesser.isObjectArrayType(decl.type)) {
		    Annotator requiresAnnotator = 
		      new Annotator(rd, "parameter:" + ms, "requires ");
		    requiresAnnotator.put("\\nonnullelements(" + decl.id.toString() + ");");
		    return;
		  }
		}
	    } else {
	      int mstat = FlowInsensitiveChecks.getOverrideStatus(md);
	      if (mstat == FlowInsensitiveChecks.MSTATUS_NEW_ROUTINE) {
		// "md" overrides nothing
		if (md.parent instanceof ClassDecl) {
		  ms = "instance method";

		  if( Modifiers.isPrivate(md.modifiers) &&
		      !methodsCalledFromMethods.contains(md) &&
		      !Modifiers.isNative(md.modifiers) &&
		      guesser.guessHelper() ) {
		      // is a private method called only from constructors, 
		      // so make it a helper method
		      (new Annotator(rd, ms, "")).put("helper");
		      
		      return;
		  }
		} else {
		  ms = "interface method";
		}
		kind = 2;
	      } else if (mstat == FlowInsensitiveChecks.MSTATUS_CLASS_NEW_METHOD) {
		// "md" is "class-new", that is, it is declared in a class
		// and overrides methods only in interfaces
		Assert.notFalse(md.parent instanceof ClassDecl);
		ms = "class-new method override";
		kind = 3;
	      } else {
		Assert.notFalse(mstat == FlowInsensitiveChecks.MSTATUS_OVERRIDE);
		// "md" overrides a method, but does not fall into the previous
		// case
		if (md.parent instanceof ClassDecl) {
		  ms = "class method override";
		} else {
		  ms = "interface method override";
		}
		kind = 4;
	      }

	      // Check if never overridden and non-final and non-abstract, 
	      // and declare as final
	      if( !Modifiers.isFinal(md.modifiers) &&
		  !Modifiers.isAbstract(md.modifiers) &&
		  !overriddenMethods.contains( md ) &&
		  guesser.guessFinalMethods() ) {
		  // is never overridden
		  (new Annotator(md, ms, "")).put("*/final/*");
	      }
	    } 
	}
	
	boolean isStatic=Modifiers.isStatic(rd.modifiers) ||
	    rd instanceof ConstructorDecl;
	//@ assume rd.parent != null;
	Hashtable envReq = getEnv( rd.parent,
				   !isStatic,
				   accessLevel(rd.modifiers),
				   true,
				   null );
	Hashtable envMod = getEnv( rd.parent, 
				   !Modifiers.isStatic( rd.modifiers ),
				   0,
				   true,
				   null );
	Hashtable envEns = getEnv( rd.parent, 
				   !Modifiers.isStatic( rd.modifiers ),
				   (Modifiers.isPrivate(rd.modifiers) ||
				    Modifiers.isFinal(rd.modifiers) ?
				    0 : 1),
				   true,
				   null );
	
	if (kind < 3) {
	    Annotator requiresAnnotator = 
		new Annotator(rd, "parameter:" + ms, "requires ");
	    
	    for(int i = 0; i < rd.args.size(); i++) {
		GenericVarDecl decl = rd.args.elementAt(i);
		Annotator modifierAnnotator = 
		    new Annotator("<P", decl.locId, decl.id.toString(),
				  "parameter:" + ms, "");
		guesser.guessExpr(decl.id.toString(), decl.type,
				  requiresAnnotator, modifierAnnotator,
				  envReq, decl );
		// Add arg to env
		envReq.put( decl.id.toString(), decl.type );
		envMod.put( decl.id.toString(), decl.type );
		envEns.put( decl.id.toString(), decl.type );
	    }
	}
	
	// Annotate method with reference return type 
	if (kind != 0) {
	    MethodDecl md = (MethodDecl)rd;
	    if (! Types.isVoidType(md.returnType)) {
		String ens = kind < 3 ? "ensures " : "also ensures ";
		Annotator ensuresAnnotator = new Annotator(rd, ms, ens);
		guesser.guessExpr("\\result", md.returnType, ensuresAnnotator, null,
				  envEns, null );
	    }
	}
	
	if (kind != 0 || rd.locId != rd.parent.locOpenBrace) {
	    Annotator requiresAnnotator;
	    Annotator modifiesAnnotator;
	    Annotator ensuresAnnotator;
	    Annotator exsuresAnnotator;
	    
	    if (kind < 3) {
		requiresAnnotator = new Annotator(rd, ms, "requires ");
		modifiesAnnotator = new Annotator(rd, ms, "modifies ");
		ensuresAnnotator = new Annotator(rd, ms, "ensures ");
	    } else {
		requiresAnnotator = null;
		modifiesAnnotator = new Annotator(rd, ms, "also modifies ");
		ensuresAnnotator = new Annotator(rd, ms, "also ensures ");
	    }
	    if (kind < 3) {
		exsuresAnnotator = new Annotator(rd, ms, "exsures ");
	    } else {
		exsuresAnnotator = new Annotator(rd, ms, "also exsures ");
	    }

//              if (kind != 0) {
//                Set vars = modifiesVisitor.processMethod(rd);
//                Enumeration e = vars.elements();
//                while (e.hasMoreElements()) {
//                  modifiesAnnotator.put("this." + (String)e.nextElement());
//                }
//              }

	    guesser.guessRoutine(rd, requiresAnnotator, modifiesAnnotator,
				 ensuresAnnotator, exsuresAnnotator,
				 envReq, envMod, envEns );
	}
    }
    
    public void visitFieldDecl(FieldDecl fd) {

	// Check if this field is part of a multiple field declaration
	int declsWithSameTypeLoc = 0;
	TypeDeclElemVec elems = fd.parent.elems;
	for(int i=0; i<elems.size(); i++) {
	    TypeDeclElem e = elems.elementAt(i);
	    if( e instanceof FieldDecl &&
		((FieldDecl)e).type.getStartLoc() == fd.getStartLoc() ) {
		declsWithSameTypeLoc++;
	    }
	}
	Assert.notFalse( declsWithSameTypeLoc > 0 );

	String kind = Modifiers.isStatic(fd.modifiers) ? "static field" : "instance field";
	String name = fd.id.toString();
	Annotator invariantAnnotator = new Annotator(">>", fd.locId, name,
						     kind, "invariant ");
	Annotator modifierAnnotator = new Annotator("<", fd.locId, name, kind, "");

	if (guesser.shouldFieldsBeSpecPublic()) {
	  // spec_public
	  if (!Modifiers.isPublic(fd.modifiers)) {
	    new Annotator("<P", fd.locId, name, kind, "")
		.put("spec_public");
	  }
	}
        if (isConstantFinal(fd)) {
	    return;
        }

	if( declsWithSameTypeLoc > 1 ) {
	    // Do not put non_null annotations on this decl
	    // use an invariant instead
	    new Annotator("<P", fd.locId, name, kind, "")
		.put("*//* SEE INVARIANTS ");
	    modifierAnnotator = null;
	}
	    
	//@ assume fd.parent != null;
	Hashtable env = getEnv( fd.parent, 
				!Modifiers.isStatic( fd.modifiers),
				0,
				!isConstantFinal(fd),
				fd );
	guesser.guessFieldDecl(fd, invariantAnnotator, modifierAnnotator);
	guesser.guessExpr(name, fd.type,
			  invariantAnnotator, modifierAnnotator,
			  env, fd);
    }

    
    //@ ensures \result != null;
    //@ ensures \result.keyType == \type(String);
    //@ ensures \result.elementType == \type(Type);
    private Hashtable getEnv(/*@ non_null */ TypeDecl td,
			     boolean allowInstance,
			     int accessLevel,
			     boolean allowConstantFinals,
			     FieldDecl fdUpto) {
	Hashtable env = (Hashtable)envBase.clone();
	
	//@ assume env.keyType == \type(String);
	//@ assume env.elementType == \type(Type);
	
	TypeSig sig = TypeSig.getSig( td );
	FieldDeclVec fds = sig.getFields(true);
	
	for(int i=0; i<fds.size(); i++) {
	    FieldDecl fd = fds.elementAt(i);
	    //@ assume fd.parent != null;
	    
	    // Consider adding fd to env
	    
	    if( fd == fdUpto ) {
		return env;
	    }
	    
	    if( !allowInstance && !Modifiers.isStatic( fd.modifiers ))
		// Dont include instance field
		continue;
	    
	    if( !guesser.shouldFieldsBeSpecPublic() &&
		accessLevel( fd.modifiers ) < accessLevel)
		// Field is more private than the reference to it would be
		continue;
	    
	    TypeSig fdsig = TypeSig.getSig( fd.parent );
	    if( !TypeCheck.inst.canAccess( sig, fdsig, fd.modifiers, fd.pmodifiers ))
		// Cannot access the field
		continue;
	    
	    if (!allowConstantFinals && isConstantFinal(fd)) {
		continue;
	    }
	    
	    
	    // Ok, add field to env
	    env.put( fd.id.toString(), fd.type );
	    
	}
	
	return env;
    }
    
    private boolean isConstantFinal(FieldDecl fd) {
	return Modifiers.isFinal(fd.modifiers) &&
	    fd.init != null &&
	    fd.init instanceof Expr &&
	    ConstantExpr.eval( (Expr)fd.init ) != null;
    }
    
    
    /** Returns an integer corresponding to the access modifiers in
	"modifiers". */
    
    private int accessLevel(int modifiers) {
	// TBW:  also check for spec_public
	if( Modifiers.isPrivate(modifiers)) return 0;
	if( Modifiers.isProtected(modifiers)) return 2;
	if( Modifiers.isPublic(modifiers)) return 3;
	return 1;
    }
}

class ConstantDimensionVisitor extends DefaultVisitor {
  /*@ non_null */ Hashtable env;

  ConstantDimensionVisitor(/*@ non_null */ Hashtable env) {
    this.env = env;
  }

  public void visitNewArrayExpr(NewArrayExpr x) {
    super.visitNewArrayExpr(x);
    
    for(int i=0; i<x.dims.size(); i++) {
      Expr dim = x.dims.elementAt(i);
      Integer val = (Integer)ConstantExpr.eval(dim);
      if (val != null) {
	env.put(val.toString(), Types.intType);
      }
    }
  }

  public void visitArrayInit(ArrayInit x) {
    super.visitArrayInit(x);
    env.put(""+x.elems.size(), Types.intType);
  }
}

class MethodCalledFromMethodVisitor extends DefaultVisitor {
  /*@ non_null */ Set set;

  MethodCalledFromMethodVisitor(/*@ non_null */ Set set) {
    this.set = set;
  }

  public void visitConstructorDecl(ConstructorDecl x) {
      // nothing, do not visit children
  }

  public void visitMethodInvocation(MethodInvocation x) {
      super.visitMethodInvocation(x);
      // add to set of methods called from other methods
      set.add( x.decl );
  }

}

class overriddenMethodVisitor extends DefaultVisitor {
  /*@ non_null */ Set overriddenMethods;

  overriddenMethodVisitor(/*@ non_null */ Set overriddenMethods) {
    this.overriddenMethods = overriddenMethods;
  }

  public void visitMethodDecl(MethodDecl x) {
      super.visitMethodDecl(x);
      overriddenMethods.addEnumeration( TypeCheck.inst.getAllOverrides(x).elements() );
  }

}

