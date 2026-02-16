/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;


import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;
import java.io.ByteArrayOutputStream;

import javafe.ast.*;
import javafe.util.Location;
import javafe.util.StackVector;
import javafe.util.Assert;
import javafe.util.Set;
import javafe.util.ErrorSet;
import javafe.util.Info;
import javafe.tc.ConstantExpr;
import javafe.tc.TypeSig;

import escjava.Main;

import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.tc.FlowInsensitiveChecks;

import escjava.backpred.FindContributors;

import escjava.tc.Types;
import escjava.tc.TypeCheck;

/** This class contains methods to check the
 *  reflexivity and transitivity of OTIs (assumptions).
**/

public final class ChkReflTrans {


    public static GuardedCmd getOtiReflTransChecks(/*@ non_null */ FindContributors scope, int location) {

	GuardedCmdVec transAssumes = GuardedCmdVec.make();
	GuardedCmdVec typeAssumes = GuardedCmdVec.make();
	GuardedCmdVec transAsserts = GuardedCmdVec.make();
	GuardedCmdVec reflAsserts = GuardedCmdVec.make();

	Hashtable map1 = new Hashtable();
	Hashtable map2 = new Hashtable(); 
	Hashtable map3 = new Hashtable();    

	/* Prepare the fields in scope for havoc */
	Enumeration fields = scope.fields(); 
	while (fields.hasMoreElements()) {
	    FieldDecl fd = (FieldDecl) fields.nextElement();

	    FieldDecl fd1, fd2, fd3;
	    if (Main.havocFinal || !Modifiers.isFinal(fd.modifiers)) {	
		fd1 = FieldDecl.make(fd.modifiers, 
				     fd.pmodifiers, 
				     Identifier.intern(fd.id.toString() + "$oti1"),
				     fd.type, 
				     fd.locId, 
				     fd.init, 
				     fd.locAssignOp);
		fd2 = FieldDecl.make(fd.modifiers, 
				     fd.pmodifiers, 
				     Identifier.intern(fd.id.toString() + "$oti2"),
				     fd.type, 
				     fd.locId, 
				     fd.init, 
				     fd.locAssignOp);
		fd3 = FieldDecl.make(fd.modifiers, 
				     fd.pmodifiers, 
				     Identifier.intern(fd.id.toString() + "$oti3"),
				     fd.type, 
				     fd.locId, 
				     fd.init, 
				     fd.locAssignOp);
	    }
	    else {
		fd1 = FieldDecl.make(fd.modifiers, 
				     fd.pmodifiers, 
				     Identifier.intern(fd.id.toString() + "$oti0"),
				     fd.type, 
				     fd.locId, 
				     fd.init, 
				     fd.locAssignOp);
		
		fd2 = fd1;
		fd3 = fd1;
	    }

	    VariableAccess va1 = VariableAccess.make(fd1.id, fd1.locId, fd1);
	    VariableAccess va2 = VariableAccess.make(fd2.id, fd2.locId, fd2);
	    VariableAccess va3 = VariableAccess.make(fd3.id, fd3.locId, fd3);

	    map1.put(fd, va1);
	    map2.put(fd, va2);
	    map3.put(fd, va3);

	}

	for(int i = 0; i < 2; i++) {
	    LocalVarDecl vd = (LocalVarDecl) 
		((i == 0) ? GC.elemsvar.decl : GC.allocvar.decl);
	    
	    LocalVarDecl vd1 = LocalVarDecl.make(vd.modifiers, 
						 vd.pmodifiers, 
						 Identifier.intern(vd.id.toString() + "$oti1"), 
						 vd.type, 
						 vd.locId, 
						 vd.init, 
						 vd.locAssignOp);
	    LocalVarDecl vd2 = LocalVarDecl.make(vd.modifiers, 
						 vd.pmodifiers, 
						 Identifier.intern(vd.id.toString() + "$oti2"), 
						 vd.type, 
						 vd.locId, 
						 vd.init, 
						 vd.locAssignOp);
	    LocalVarDecl vd3 = LocalVarDecl.make(vd.modifiers, 
						 vd.pmodifiers, 
						 Identifier.intern(vd.id.toString() + "$oti3"), 
						 vd.type, 
						 vd.locId, 
						 vd.init, 
						 vd.locAssignOp);
	    
	    VariableAccess va1 = VariableAccess.make(vd1.id, vd1.locId, vd1);
	    VariableAccess va2 = VariableAccess.make(vd2.id, vd2.locId, vd2);
	    VariableAccess va3 = VariableAccess.make(vd3.id, vd3.locId, vd3);
	    map1.put(vd, va1);
	    map2.put(vd, va2);
	    map3.put(vd, va3);
	}


	// generate the type correctness conditions for all the havoced variables	
	for (Enumeration e = map1.elements(); 
	     e.hasMoreElements();) {	    
	    VariableAccess va = (VariableAccess) e.nextElement();
	    typeAssumes.addElement(GC.assume(TrAnExpr.targetTypeCorrect(va.decl, map1)));
	}
	for (Enumeration e = map2.elements(); 
	     e.hasMoreElements();) {	    
	    VariableAccess va = (VariableAccess) e.nextElement();
	    typeAssumes.addElement(GC.assume(TrAnExpr.targetTypeCorrect(va.decl, map2)));
	}
	for (Enumeration e = map3.elements(); 
	     e.hasMoreElements();) {	    
	    VariableAccess va = (VariableAccess) e.nextElement();
	    typeAssumes.addElement(GC.assume(TrAnExpr.targetTypeCorrect(va.decl, map3)));
	}


    
      for (InvariantInfo ii = GetSpec.collectOtis(scope); ii != null; ii = ii.next) {
	  ExprDeclPragma td = (ExprDeclPragma) ii.prag;
	  int loc = td.getStartLoc();

	  Expr assume1Expr = TrAnExpr.trSpecExpr(td.expr, TrAnExpr.union(ii.map, map2), map1);
	  Expr assume2Expr = TrAnExpr.trSpecExpr(td.expr, TrAnExpr.union(ii.map, map3), map2);
	  Expr assertExpr = TrAnExpr.trSpecExpr(td.expr, TrAnExpr.union(ii.map, map3), map1);
	  Expr reflAssertExpr = TrAnExpr.trSpecExpr(td.expr, TrAnExpr.union(ii.map, map1), map1);


	  if (ii.isStatic) {
	      Condition condn = GC.condition(TagConstants.CHKOTITRANS,
					     assertExpr,
					     loc);
	      Condition rCondn = GC.condition(TagConstants.CHKOTIREFL,
					      reflAssertExpr,
					      loc);

	      reflAsserts.addElement(GC.check(location, rCondn));
	      transAsserts.addElement(GC.check(location, condn));
	      transAssumes.addElement(GC.assume(assume1Expr));
	      transAssumes.addElement(GC.assume(assume2Expr));

	  } else {

	      ExprVec ev = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
								 null, true,
								 null, false);
	      ExprVec nopats = ev.copy();	      
	      Expr ante = GC.and(ev);
	      
	      Expr quantAssertExpr = GC.quantifiedExpr( loc, loc, 
							TagConstants.FORALL,
							ii.sdecl,
							GC.implies(ante, assertExpr),
							nopats );

	      Expr quantReflAssertExpr = GC.quantifiedExpr( loc, loc, 
							    TagConstants.FORALL,
							    ii.sdecl,
							    GC.implies(ante, reflAssertExpr),
							    nopats );

	      Expr quantAssume1Expr = GC.quantifiedExpr( loc, loc, 
							 TagConstants.FORALL,
							 ii.sdecl,
							 GC.implies(ante, assume1Expr),
							 nopats );

	      Expr quantAssume2Expr = GC.quantifiedExpr( loc, loc, 
							 TagConstants.FORALL,
							 ii.sdecl,
							 GC.implies(ante, assume2Expr),
							 nopats );


	      Condition condn = GC.condition(TagConstants.CHKOTITRANS,
					     quantAssertExpr,
					     loc);
	      Condition rCondn = GC.condition(TagConstants.CHKOTIREFL,
					      quantReflAssertExpr,
					      loc);

	      reflAsserts.addElement(GC.check(location, rCondn));
	      transAsserts.addElement(GC.check(location, condn));
	      transAssumes.addElement(GC.assume(quantAssume1Expr));
	      transAssumes.addElement(GC.assume(quantAssume2Expr));
	  }
      }


      /* Add the oti for the ghost holder field to the transAssumes */
      /* We don't add it to the transAsserts as we already know
	 that these OTIs are transitive. */
      TypeSig obj = Types.javaLangObject();
      FieldDecl holder = null; 
      try {
	  holder = Types.lookupField(obj, Identifier.intern("holder"), obj);
	  VariableAccess holderVA1 = (VariableAccess) map1.get(holder);
	  if (holderVA1 == null) {
	      //	      System.out.println("FindContributors did not find ghost field holder in scope ");
	  } else {
	      VariableAccess holderVA = TrAnExpr.makeVarAccess(holder, Location.NULL);
	      LocalVarDecl sdecl = UniqName.newBoundThis();
	      VariableAccess sdeclVA = TrAnExpr.makeVarAccess(sdecl, Location.NULL);
	      ExprVec ev = TrAnExpr.typeAndNonNullAllocCorrectAs(sdecl, obj,
								 null, true,
								 null, false);
	      ExprVec nopats = ev.copy();	      
	      Expr ante = GC.and(ev);
	      VariableAccess holderVA2 = (VariableAccess) map2.get(holder);
	      VariableAccess holderVA3 = (VariableAccess) map3.get(holder);
	      Expr expr1 = GC.nary(TagConstants.BOOLEQ, 
				   GC.nary(TagConstants.ANYEQ, 
					   GC.select(holderVA1, sdeclVA), 
					   GC.thisThread),
				   GC.nary(TagConstants.ANYEQ, 
					   GC.select(holderVA2, sdeclVA),
					   GC.thisThread));
	      Expr expr2 = GC.nary(TagConstants.BOOLEQ, 
				   GC.nary(TagConstants.ANYEQ, 
					   GC.select(holderVA2, sdeclVA), 
					   GC.thisThread),
				   GC.nary(TagConstants.ANYEQ, 
					   GC.select(holderVA3, sdeclVA),
					   GC.thisThread));
	      Expr expr3 = GC.nary(TagConstants.BOOLEQ, 
				   GC.nary(TagConstants.ANYEQ, 
					   GC.select(holderVA1, sdeclVA), 
					   GC.thisThread),
				   GC.nary(TagConstants.ANYEQ, 
					   GC.select(holderVA3, sdeclVA),
					   GC.thisThread));
	      
	      Expr qExpr1 = GC.quantifiedExpr( Location.NULL, Location.NULL,
					       TagConstants.FORALL,
					       sdecl,
					       GC.implies(ante, expr1),
					       nopats );
	      Expr qExpr2 = GC.quantifiedExpr( Location.NULL, Location.NULL,
					       TagConstants.FORALL,
					       sdecl,
					       GC.implies(ante, expr2),
					       nopats );
	      Expr qExpr3 = GC.quantifiedExpr( Location.NULL, Location.NULL,
					       TagConstants.FORALL,
					       sdecl,
					       GC.implies(ante, expr3),
					       nopats );
	      //	      Expr assumeExpr = GC.implies(GC.and(qExpr1, qExpr2), qExpr3);
	      Expr assumeExpr = GC.and(GC.and(qExpr1, qExpr2), qExpr3);
	      transAssumes.addElement(GC.assume(assumeExpr));
	  }
      }
      catch (javafe.tc.LookupException e) {
	  // if we couldn't find the holder ghost field, there's nothing to do
	  System.out.println("Could not find the ghost holder field");
      }


      return GC.seq(GC.seq(typeAssumes), GC.seq(reflAsserts), GC.seq(transAssumes), GC.seq(transAsserts));
    }


} // end of class



