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

/** This class deals with the generation and storage
 *  of the desugared code for a "yield".
**/
public final class TrYield {

    /** Scope of the method that this object corresponds to */
    private FindContributors scope;

    /** maps v to "v$old" */
    //private  Hashtable assertMap; 
    public  Hashtable assertMap; 

    /** to replace tid by notTid */
    //private  Hashtable tidMap;
    public  Hashtable tidMap;

    /** maps GenericVarDecl v to VariableAccess v$tmp */
    //private Hashtable assumeMap;
    public Hashtable assumeMap;

    /** maps VariableAccess v$tmp to GenericVarDecl v */
    private Hashtable reverseAssumeMap;

    /** maps v to v */
    private  Hashtable identityMap;


    private  ConditionVec assertOtiVec; 
    private  ConditionVec assertGiVec; 

    private  GuardedCmdVec oldAssigns;

    /** Vector storing the HavocAssume objects
	corresponding to equiv classes of OTIs
    */
    private HavocAssumeVec havocAssumeVec;

    /** Linked list storing the HavocAssume objects
	corresponding to each OTI
    */
    private HavocAssume havocAssumeList;
    
    /** Creates a new TrYield object containing the code that a
	"yield" guarded command is desugared into.  */
    TrYield(/*@ non_null */ FindContributors scope, 
	    /*@ non_null */ Spec spec) {
	this.scope = scope;

	assertOtiVec = ConditionVec.make(); 
	assertGiVec = ConditionVec.make(); 
	assertMap = new Hashtable(); 
	

	assumeMap = new Hashtable(); 
	reverseAssumeMap = new Hashtable();
	identityMap = new Hashtable(); 
	tidMap = new Hashtable();    

	havocAssumeList = null;
	havocAssumeVec = HavocAssumeVec.make();

	oldAssigns = GuardedCmdVec.make();

	tidMap.put(GC.thisThread.decl, GC.notThisThread);	  	 

	// these methods must be called in this order
	prepareSharedVariables();
	computeHavocAssumeCode();

	// create old version of witness variable
	LocalVarDecl vd, oldvd;
	vd = (LocalVarDecl) GC.witness.decl;
 	oldvd = LocalVarDecl.make(vd.modifiers, 
				  vd.pmodifiers, 
				  Identifier.intern(vd.id.toString() + "$old"), 
				  vd.type, 
				  vd.locId, 
				  vd.init, 
				  vd.locAssignOp);
	VariableAccess oldva = VariableAccess.make(oldvd.id, oldvd.locId, oldvd);
	assertMap.put(vd, oldva);
	oldAssigns.addElement(GC.gets(oldva, GC.witness));

	computePerformsCode(spec);
    }
    
    private void prepareSharedVariables() {
	/* Prepare the fields in scope for havoc */
	Enumeration fields = scope.fields(); 
	while (fields.hasMoreElements()) {
	    FieldDecl fd = (FieldDecl) fields.nextElement();
	    if ( !Translate.isThreadLocal(fd.parent) && 
		 (Main.havocFinal || !Modifiers.isFinal(fd.modifiers)) ) {		
		FieldDecl oldfd = FieldDecl.make(fd.modifiers, 
						 fd.pmodifiers, 
						 Identifier.intern(fd.id.toString() + "$old"),
						 // this might need to be revisited
						 fd.type, 
						 fd.locId, 
						 fd.init, 
						 fd.locAssignOp);
		processVarForHavoc(fd, oldfd);
	    }
	}

	LocalVarDecl vd, oldvd;
	/* Prepare elemsvar for havoc */
	vd = (LocalVarDecl) GC.elemsvar.decl;
 	oldvd = LocalVarDecl.make(vd.modifiers, 
				  vd.pmodifiers, 
				  Identifier.intern(vd.id.toString() + "$old"), 
				  vd.type, 
				  vd.locId, 
				  vd.init, 
				  vd.locAssignOp);
	processVarForHavoc(vd, oldvd);

	/* Pepare allocvar for havoc */
	vd = (LocalVarDecl) GC.allocvar.decl;
	oldvd = LocalVarDecl.make(vd.modifiers, 
				  vd.pmodifiers, 
				  Identifier.intern(vd.id.toString() + "$old"), 
				  vd.type, 
				  vd.locId, 
				  vd.init, 
				  vd.locAssignOp);
	processVarForHavoc(vd, oldvd);
    }

    /** For each field (instance and static) in scope prepare the data 
       structures for havocing them.
       1. Contruct a variable f$old to represent the old value of the 
       field at the end of the previous yield.  assertMap maps f to 
       f$old.  Add assignment to oldAssigns that assigns current value
       of f to f$old.  This will be put at the end of the yield.
       2. Havoc f by assigning fresh variable f$tmp to f, put the 
       variable in havocList and the assignment in havocAssign.  
    */	        
    private  void processVarForHavoc(GenericVarDecl fd, GenericVarDecl oldfd) {
	Assert.notFalse(fd instanceof FieldDecl || fd instanceof LocalVarDecl);	
	VariableAccess va = VariableAccess.make(fd.id, fd.locId, fd);
	VariableAccess oldva = VariableAccess.make(oldfd.id, oldfd.locId, oldfd);
	oldAssigns.addElement(GC.gets(oldva,va));
	assertMap.put(fd, oldva);

	LocalVarDecl tmpvd;
	if (fd instanceof FieldDecl) {
	    FieldDecl fd1 = (FieldDecl) fd;
	    tmpvd = LocalVarDecl.make(fd1.modifiers, 
				      fd1.pmodifiers, 
				      Identifier.intern(fd1.id.toString() + "$tmp"), 
				      fd1.type, 
				      fd1.locId, 
				      fd1.init,
				      fd1.locAssignOp);
	}
	else {
	    LocalVarDecl vd1 = (LocalVarDecl) fd;
	    tmpvd = LocalVarDecl.make(vd1.modifiers, 
				      vd1.pmodifiers, 
				      Identifier.intern(vd1.id.toString() + "$tmp"), 
				      vd1.type, 
				      vd1.locId, 
				      vd1.init, 
				      vd1.locAssignOp);
	}

	VariableAccess result1 = VariableAccess.make(tmpvd.id, tmpvd.locId, tmpvd);

	assumeMap.put(fd, result1);
	reverseAssumeMap.put(result1, fd);
	identityMap.put(fd, va);
    }

    private void mentions(Expr expr, Hashtable map) {
	if (expr instanceof VariableAccess) {
	    VariableAccess va = (VariableAccess) expr;
	    Object vd = reverseAssumeMap.get(va);
	    if (vd != null) {
		map.put(vd, va);
	    } else {
		Object va1 = assumeMap.get(va.decl);
		if (va1 != null) {
		    map.put(va.decl, va1);
		}
	    }
	} else {
	    //	    System.out.println(expr);
	    for (int i = 0; i < expr.childCount(); i++) {
		Object child = expr.childAt(i);
		if (child instanceof Expr) {
		    mentions((Expr) child, map);
		}
	    }
	}	
    }


    private boolean mentionsVar(Expr expr, GenericVarDecl vd) {
	if (expr instanceof VariableAccess) {
	    VariableAccess va = (VariableAccess) expr;
	    return (va.decl == vd);
	} else {
	    for (int i = 0; i < expr.childCount(); i++) {
		Object child = expr.childAt(i);
		if ((child instanceof Expr) &&
		    mentionsVar((Expr) child, vd))
		    return true;
	    }
	    return false;
	}	
    }


    private void computeHavocAssumeCode() {
	computeOtiCode();
	if (Main.otiInferOpt) {
	    inferOtiFromAnnotations();
	}
	inferOtiForGhostHolder();
	
	computeGiCode();

	computeEquivalenceClasses();

	/* Shaz: March 22, 2002
	   This code will not call ha.finish.  It only generates the remainder of 
	   havocAssumes for the variables that do not occur in any oti seen so far.
	   The havocAssumes are parameterized by the set of thread-local variables.
	*/
	Hashtable equivUnion = new Hashtable();
	for (int i = 0; i < havocAssumeVec.size(); i++) {
	    HavocAssume ha = havocAssumeVec.elementAt(i);
	    equivUnion.putAll(ha.getHavocMap());
	    //ha.finish();
	}      	

	Enumeration e = assumeMap.keys();
	while(e.hasMoreElements()) {
	    GenericVarDecl vd = (GenericVarDecl) e.nextElement();
	    if (!equivUnion.containsKey(vd)) {
		Hashtable tmp = new Hashtable();
		tmp.put(vd, assumeMap.get(vd));
		HavocAssume ha = new HavocAssume(GC.skip(),
						 tmp);
		//ha.finish();
		havocAssumeVec.addElement(ha);
	    }
	}
    }

    private void computeEquivalenceClasses() {
	while (havocAssumeList != null) {
	    HavocAssume head = havocAssumeList;
	    HavocAssume iter = havocAssumeList;
	    boolean unchanged = true;

	    while (iter.next != null) {
		if (head.merge(iter.next)) {
		    iter.next = iter.next.next;
		    unchanged = false;
		} else {
		    iter = iter.next;
		}
	    }

	    if (unchanged) {
		havocAssumeVec.addElement(head);
		havocAssumeList = head.next;
 	    }

	}
    }
    
    /** Computes the part of yield code that contains the
	following: 1) Asserts of guarantees, 2) Havoc-Assumes
	(of assumptions), 3) Assignments to save the current
	state of fields for future use in guarantees 
    */
    private void computeOtiCode() {      
      /* The following loop generates 
	 (1) the asserts, and 
	 (2) the assumes after havocs
      */
      for (InvariantInfo ii = GetSpec.collectOtis(scope); ii != null; ii = ii.next) {
	  ExprDeclPragma td = (ExprDeclPragma) ii.prag;
	  int loc = td.getStartLoc();	  
	  Expr notTidConstraint = 
	      GC.and(TrAnExpr.typeAndNonNullAllocCorrectAs(GC.notThisThread.decl,
							   Types.getJavaLang("Thread"),
							   null,
							   true,
							   null,
							   true));

	  notTidConstraint = GC.and(loc, loc, 
				    notTidConstraint, 
				    GC.nary(loc, loc, 
					    TagConstants.REFNE, GC.notThisThread, GC.thisThread));

	  Expr assertExpr = TrAnExpr.trSpecExpr(td.expr, TrAnExpr.union(tidMap, ii.map), assertMap);
	  assertExpr = GC.implies( notTidConstraint, assertExpr );

	  Expr assumeExpr = TrAnExpr.trSpecExpr(td.expr, TrAnExpr.union(ii.map, assumeMap), identityMap);

	  Hashtable map = new Hashtable();
	  mentions(assumeExpr, map);

	  if (ii.isStatic) {
	      Expr f = GC.quantifiedExpr( loc, loc,
					  TagConstants.FORALL,
					  GC.notThisThread.decl,
					  assertExpr,
					  null );

	      Condition condn = GC.condition(TagConstants.CHKOTI,
					     f,
					     loc);
	      assertOtiVec.addElement(condn);

	      // assumeBeforeHavoc.addElement(GC.assume(assumeExpr));
	      HavocAssume ha = new HavocAssume(GC.assume(assumeExpr), map);
	      ha.next = havocAssumeList;
	      havocAssumeList = ha;
	  } else {
	      ExprVec ev = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
								 null, true,
								 null, true /*false*/);
	      ExprVec nopats = ev.copy();	      
	      Expr ante = GC.and(ev);
	      ExprVec ev1 = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
								  null, true,
								  null, false);
	      ExprVec nopats1 = ev1.copy();	      
	      Expr ante1 = GC.and(ev1);
	      
	      GenericVarDeclVec vs = GenericVarDeclVec.make();
	      vs.addElement(GC.notThisThread.decl);
	      vs.addElement(ii.sdecl);
	      Expr quantifiedAssertExpr = GC.quantifiedExpr( loc, loc,
							     TagConstants.FORALL,
							     vs,
							     GC.implies(ante, assertExpr),
							     nopats );

	      Condition condn = GC.condition(TagConstants.CHKOTI,
					     quantifiedAssertExpr,
					     loc);
	      assertOtiVec.addElement(condn);
	      
	      Expr quantifiedAssumeExpr = GC.quantifiedExpr( loc, loc, 
							     TagConstants.FORALL,
							     ii.sdecl,
							     GC.implies(ante1, assumeExpr),
							     nopats1 );
	      // assumeBeforeHavoc.addElement(GC.assume(quantifiedAssumeExpr));
	      HavocAssume ha = new HavocAssume(GC.assume(quantifiedAssumeExpr), map);
	      ha.next = havocAssumeList;
	      havocAssumeList = ha;
	  }
      }
    }

    private void inferOtiForGhostHolder() {
      /* Add the oti for the ghost holder field to assumeBeforeHavoc */
      TypeSig obj = Types.javaLangObject();
      FieldDecl holder = null; // make the compiler happy
      try {
	  holder = Types.lookupField(obj, Identifier.intern("holder"), obj);
	  VariableAccess oldHolderVA = (VariableAccess) assumeMap.get(holder);
	  if (oldHolderVA == null) {
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
	      Expr assumeExpr = GC.nary(TagConstants.BOOLEQ, 
					GC.nary(TagConstants.ANYEQ, 
						GC.select(oldHolderVA, sdeclVA), 
						GC.thisThread),
					GC.nary(TagConstants.ANYEQ, 
						GC.select(holderVA, sdeclVA),
						GC.thisThread));
	      Expr quantifiedAssumeExpr = GC.quantifiedExpr( Location.NULL, Location.NULL,
							     TagConstants.FORALL,
							     sdecl,
							     GC.implies(ante, assumeExpr),
							     nopats );
	      // assumeBeforeHavoc.addElement(GC.assume(quantifiedAssumeExpr));
	      Hashtable map = new Hashtable();
	      map.put(holder, oldHolderVA);
	      HavocAssume ha = new HavocAssume(GC.assume(quantifiedAssumeExpr), map);
	      ha.next = havocAssumeList;
	      havocAssumeList = ha;
	  }
      }
      catch (javafe.tc.LookupException e) {
	  // if we couldn't find the holder ghost field, there's nothing to do
	  System.out.println("Could not find the ghost holder field");
      }
    }

    private void inferOtiFromAnnotations() {
	inferOtiForFields();
	inferOtiForElems();
    }

    private void inferOtiForFields() {
	Enumeration fields = scope.fields(); 
	while (fields.hasMoreElements()) {
	    FieldDecl fd = (FieldDecl) fields.nextElement();
	    ParamExprModifierPragma prag = Translate.FindUnwritableByEnvIfModifierPragma(fd);
	    if (prag != null) {
		createOtiFromGuardedBy(fd, prag);
	    }  
	}
    }

    /*
    private void inferOtiForElems() {
	LocalVarDecl array = UniqName.newBoundVariable("arrayObj");
	VariableAccess arrayVA = TrAnExpr.makeVarAccess(array, Location.NULL);

	ExprVec v = ExprVec.make();
	Enumeration e = scope.typeSigs();
	while (e.hasMoreElements()) {
	    TypeSig sig = (TypeSig) e.nextElement();
	    if (Translate.isThreadLocal(sig.getTypeDecl())) {
		ArrayType arrayType = ArrayType.make(sig, Location.NULL);
		Expr eq = GC.nary(TagConstants.TYPEEQ,
				  GC.nary(TagConstants.TYPEOF, arrayVA),
				  TypeExpr.make(Location.NULL, Location.NULL, arrayType));		
		v.addElement(eq);
	    }
	}

	VariableAccess tmpVA = (VariableAccess) assumeMap.get(GC.elemsvar.decl);
	Expr eq = GC.nary(TagConstants.ANYEQ, 
			  GC.select(GC.elemsvar, arrayVA),
			  GC.select(tmpVA, arrayVA));

	Expr forallExpr = GC.forall(array, GC.implies(GC.or(v), eq));

	Hashtable map = new Hashtable();
	mentions(forallExpr, map);
	HavocAssume ha = new HavocAssume(GC.assume(forallExpr), map);
	ha.next = havocAssumeList;
	havocAssumeList = ha;
    }
    */

    private void inferOtiForElems() {
	ExprVec v = ExprVec.make();
	Enumeration e = scope.typeSigs();
	VariableAccess tmpVA = (VariableAccess) assumeMap.get(GC.elemsvar.decl);

	while (e.hasMoreElements()) {
	    TypeSig sig = (TypeSig) e.nextElement();
	    if (Translate.isThreadLocal(sig.getTypeDecl())) {
		ArrayType arrayType = ArrayType.make(sig, Location.NULL);
		LocalVarDecl array = UniqName.newBoundVariable("arrayObj", arrayType);
		VariableAccess arrayVA = TrAnExpr.makeVarAccess(array, Location.NULL);

		Expr eq = GC.nary(TagConstants.ANYEQ, 
				  GC.select(GC.elemsvar, arrayVA),
				  GC.select(tmpVA, arrayVA));
		Expr all = GC.forall(array, eq);

		Hashtable map = new Hashtable();
		mentions(all, map);
		HavocAssume ha = new HavocAssume(GC.assume(all), map);
		ha.next = havocAssumeList;
		havocAssumeList = ha;
	    }
	}
    }

    /*
    private void createOtiFromGuardedBy(FieldDecl fd, ExprModifierPragma mp) {
	LocalVarDecl obj = UniqName.newBoundThis();
	VariableAccess objva = TrAnExpr.makeVarAccess(obj, Location.NULL);
	Hashtable thisMap = new Hashtable();
	thisMap.put(GC.thisvar.decl, objva);
	
	Expr disjunct; 
	if (Main.reduce) {
	    Expr e1 = TrAnExpr.trSpecExpr(mp.expr, TrAnExpr.union(thisMap, assumeMap), thisMap);
	    Expr e2 = TrAnExpr.trSpecExpr(mp.expr, thisMap, thisMap);
	    disjunct = GC.not(GC.or(e1, e2));
	} else {
	    disjunct = GC.not(TrAnExpr.trSpecExpr(mp.expr, thisMap, thisMap));
	}
	
	// Create the assume to be added to assumeBeforeHavoc
	VariableAccess va = VariableAccess.make(fd.id, fd.locId, fd);
	VariableAccess oldva = (VariableAccess) assumeMap.get(fd);
	if (Modifiers.isStatic(fd.modifiers)) {
	    // static field
	    Expr assumeExpr = GC.or(disjunct, GC.nary(TagConstants.ANYEQ, va, oldva));
	    
	    // assumeBeforeHavoc.addElement(GC.assume(assumeExpr));
	    Hashtable map = new Hashtable();
	    mentions(assumeExpr, map);
	    HavocAssume ha = new HavocAssume(GC.assume(assumeExpr), map);
	    ha.next = havocAssumeList;
	    havocAssumeList = ha;
	} else {
	    // instance field
	    TypeSig parentSig = TypeSig.getSig(fd.parent);
	    int loc = fd.getStartLoc();

	    ExprVec evec = TrAnExpr.typeAndNonNullAllocCorrectAs(obj, parentSig,
								 null, true,
								 null, false);
	    Expr e = GC.or(disjunct, GC.nary(TagConstants.ANYEQ, 
					     GC.select(va, objva), 
					     GC.select(oldva, objva)));
	    Expr forallExpr = GC.quantifiedExpr( loc, loc,
						 TagConstants.FORALL,
						 obj,
						 GC.implies(GC.and(evec), e),
						 evec.copy() );
	    // assumeBeforeHavoc.addElement(GC.assume(forallExpr));
	    Hashtable map = new Hashtable();
	    mentions(forallExpr, map);
	    HavocAssume ha = new HavocAssume(GC.assume(forallExpr), map);
	    ha.next = havocAssumeList;
	    havocAssumeList = ha;
	}	  
    }
    */

    private void createOtiFromElemsGuardedBy() {
	
    }

    private void createOtiFromGuardedBy(FieldDecl fd, ParamExprModifierPragma mp) {
	LocalVarDecl obj = UniqName.newBoundThis();
	VariableAccess objva = TrAnExpr.makeVarAccess(obj, Location.NULL);
	Hashtable thisMap = new Hashtable();
	thisMap.put(GC.thisvar.decl, objva);
	
	Expr disjunct; 
	if (Main.reduce) {
	    Expr e1 = TrAnExpr.trSpecExpr(mp.expr, TrAnExpr.union(thisMap, assumeMap), thisMap);
	    Expr e2 = TrAnExpr.trSpecExpr(mp.expr, thisMap, thisMap);
	    disjunct = GC.not(GC.or(e1, e2));
	} else {
	    /*
	      ev1.addElement(TrAnExpr.trSpecExpr(((ExprModifierPragma)mp).expr, 
	      TrAnExpr.union(thisMap, assumeMap), 
	      thisMap));
	    */
	    disjunct = GC.not(TrAnExpr.trSpecExpr(mp.expr, thisMap, thisMap));
	}
	
	// Create the assume to be added to assumeBeforeHavoc
	VariableAccess va = VariableAccess.make(fd.id, fd.locId, fd);
	VariableAccess oldva = (VariableAccess) assumeMap.get(fd);
	Expr expr = va;
	Expr oldExpr = oldva;
	GenericVarDeclVec vdVec = GenericVarDeclVec.make();
	ExprVec eVec = ExprVec.make();

	if (!Modifiers.isStatic(fd.modifiers)) {
	    TypeSig parentSig = TypeSig.getSig(fd.parent);
	    expr = GC.select(expr, objva);
	    oldExpr = GC.select(oldExpr, objva);
	    vdVec.addElement(obj);
	    eVec.append(TrAnExpr.typeAndNonNullAllocCorrectAs(obj, parentSig,
							      null, true,
							      null, false));
	}

	for (int i = 0; i < mp.vds.size(); i++) {
	    GenericVarDecl vd = mp.vds.elementAt(i);
	    VariableAccess temp = TrAnExpr.makeVarAccess(vd, Location.NULL);
	    expr = GC.select(expr, temp);
	    oldExpr = GC.select(oldExpr, temp);		
	    vdVec.addElement(vd);
	    eVec.append(TrAnExpr.typeAndNonNullAllocCorrectAs(vd, vd.type,
							      null, true,
							      null, false));
	}

	Expr e = GC.or(disjunct, GC.nary(TagConstants.ANYEQ, expr, oldExpr));
	int loc = fd.getStartLoc();
	Expr forallExpr = GC.quantifiedExpr( loc, loc,
					     TagConstants.FORALL,
					     vdVec,
					     GC.implies(GC.and(eVec), e),
					     eVec.copy() );
	Hashtable map = new Hashtable();
	mentions(forallExpr, map);
	HavocAssume ha = new HavocAssume(GC.assume(forallExpr), map);
	ha.next = havocAssumeList;
	havocAssumeList = ha;
    }

    private void computeGiCode() {
      for (InvariantInfo ii = GetSpec.collectGlobalInvariants(scope); ii != null; ii = ii.next) {	  
	  ExprDeclPragma td = (ExprDeclPragma) ii.prag;
	  int loc = td.getStartLoc();
	  /*
	    Expr assertAssumeExpr = TrAnExpr.trSpecExpr(td.expr, ii.map, null);
	  */

	  Hashtable map = new Hashtable();
	  mentions(ii.J, map);
	  HavocAssume ha;

	  if (ii.isStatic) {
	      Condition condn = GC.condition(TagConstants.CHKGLOBALINVARIANT,
					     ii.J,
					     td.getStartLoc());
	      assertGiVec.addElement(condn);
	      
	      // assumeBeforeHavoc.addElement(GC.assume(Substitute.doSubst(assumeMap, ii.J)));      
	      ha = new HavocAssume(GC.assume(Substitute.doSubst(assumeMap, ii.J)), map);

	      //	System.out.println("++++++++++ GI BEGINS +++++++++++++");
	      //System.out.println(PrettyPrint.inst.toString(f1));
	      //System.out.println("++++++++++ GI ENDS +++++++++++++");

	  } else {
	      ExprVec ev = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
								 null, true,
								 null, true /*false*/);
	      ExprVec nopats = ev.copy();	      
	      Expr ante = GC.and(ev);
	      ExprVec ev1 = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
								  null, true,
								  null, false);
	      ExprVec nopats1 = ev1.copy();	      
	      Expr ante1 = GC.and(ev1);
	      
	      Expr f = GC.quantifiedExpr( loc, loc,
					  TagConstants.FORALL,
					  ii.sdecl,
					  GC.implies(ante, ii.J),
					  nopats );
	      Expr f1 = GC.quantifiedExpr( loc, loc,
					  TagConstants.FORALL,
					   ii.sdecl,
					   GC.implies(ante1, ii.J),
					   nopats1 );
	      
	      Condition condn = GC.condition(TagConstants.CHKGLOBALINVARIANT,
					     f,
					     td.getStartLoc());
	      assertGiVec.addElement(condn);	      

	      // assumeBeforeHavoc.addElement(GC.assume(Substitute.doSubst(assumeMap, f)));
	      ha = new HavocAssume(GC.assume(Substitute.doSubst(assumeMap, f1)), map);

	      //	System.out.println("++++++++++ GI2 BEGINS +++++++++++++");
	      //	System.out.println(PrettyPrint.inst.toString(f1));
	      //	System.out.println("++++++++++ GI2 ENDS +++++++++++++");


	  }

	  ha.next = havocAssumeList;
	  havocAssumeList = ha;	      
      }
    }

    private PerformsModifierPragma performsPragma; 
    private Condition performsCondition ;
    private Condition endWitnessCondition;

    /** Computes the part of yield desugaring code that is
	used to check a performs specification.
    */
    private  void computePerformsCode(/*@ non_null */ Spec spec) {
	performsPragma = spec.dmd.performs;

	if (performsPragma == null) {
	    return;
	}

	VariableAccess witness = GC.witness;
	VariableAccess oldWitness = (VariableAccess) assertMap.get(witness.decl);
	ExprVec eVec = ExprVec.make();
	eVec.addElement(GC.nary(TagConstants.ANYEQ, witness, oldWitness));

	LabelRelation lr = performsPragma.labelRelation;

	Enumeration domain = lr.domain();
	while (domain.hasMoreElements()) {
	    String s = (String) domain.nextElement();
	    int sint = lr.stringToInt(s);
	    Enumeration range = lr.range(s);
	    while (range.hasMoreElements()) {
		String t = (String) range.nextElement(); 
		if (t.equals("end")) {
		    continue;
		}
		int tint = lr.stringToInt(t);
		PerformsAction action = lr.getAction(t);
		Expr pred = TrAnExpr.trSpecExpr(action.pred, null, assertMap);

		ExprVec andVec = ExprVec.make();
		andVec.addElement(GC.nary(TagConstants.ANYEQ, oldWitness, GC.intlit(sint)));
		andVec.addElement(GC.nary(TagConstants.ANYEQ, witness, GC.intlit(tint)));
		andVec.addElement(pred);		

		eVec.addElement(GC.and(andVec));
	    }
	}

	performsCondition = GC.condition(TagConstants.CHKPERFORMS, GC.or(eVec), performsPragma.loc);
	
	ExprVec orVec = ExprVec.make();
	domain = lr.domain();	
	while (domain.hasMoreElements()) {
	    String s = (String) domain.nextElement();
	    if (lr.inRelation(s, "end")) {
		orVec.addElement(GC.nary(TagConstants.ANYEQ, GC.witness, GC.intlit(lr.stringToInt(s))));
	    }
	}
	
	endWitnessCondition = GC.condition(TagConstants.CHKPERFORMSEND, 
					   GC.implies(GC.nary(TagConstants.ANYEQ, GC.ecvar, GC.ec_return), 
						      GC.or(orVec)),
					   performsPragma.loc);	    
    }


    /** Returns the code to save the current shared state
	for future use in guarantees
    */
    public  GuardedCmd getAssignCode() {
	return GC.seq(oldAssigns);
    }


    /** Returns the code that initializes the performs state machine
	in the BEGIN state. The output of this method is to be placed
	at the start of the method being analyzed */
    public  GuardedCmd getStateTrueAssignCode() {
	if (performsPragma != null) {
	    return GC.gets(GC.witness, GC.intlit(performsPragma.
						 labelRelation.
						 stringToInt("begin")));
	} else {
	    return GC.skip();
	}
    }

    /** Returns a vector of Conditions to check that the performs
	state machine finishes in the END state. The output of this
	method is to be checked at the end of the method being
	analyzed, along with the post conditions. */
    /*
    private  ConditionVec getStateFalseAssertConds() {
	return stateFalseAsserts;
	}*/

    /** Returns the code to check that the performs state machine
	finishes in the END state. The output of this method is to be
	placed at the end of the method being analyzed. */
    public  GuardedCmd getPerformsEndAssertCode(int loc) {
	if (performsPragma != null) {
	    return GC.check(loc, endWitnessCondition);
	} else {
	    return GC.skip();
	}

    }

    /** Generates and returns the code to check the "performs" clause
       inside a method. This method is called from getYieldCode */
    private  GuardedCmd getPerformsYieldCode(int loc) {
	if (performsPragma != null) {
	   return GC.check(loc, performsCondition);
	} else {	
	    return GC.skip();
	}
    }

    /** Returns the part of yield code that havocs
	the shared state under the assumptions(OTIs).
	The returned code contains havocs of all
	variables in scope.
     */
    public GuardedCmd getHavocAssumes(GenericVarDeclVec threadLocalVars) {
	GuardedCmdVec all = GuardedCmdVec.make();
	for(int i=0; i< havocAssumeVec.size(); i++) {
	    HavocAssume ha = havocAssumeVec.elementAt(i);
	    all.addElement(ha.getCmd(threadLocalVars));    
	}

	return GC.seq(all);
    }

    /** Returns the part of yield code that havocs
	the shared state under the assumptions(OTIs).
	The returned code contains havocs of only
	variables in the OTI-closure of <code>vd</code>.
     */
    public GuardedCmd getHavocAssumes(GenericVarDecl gvd, GenericVarDeclVec threadLocalVars) {
	for(int i=0; i< havocAssumeVec.size(); i++) {
	    HavocAssume ha = havocAssumeVec.elementAt(i);
	    // Only one ha should refers to gvd
	    if (ha.refersTo(gvd)) {
		return ha.getCmd(threadLocalVars);
	    }
	}

	return GC.skip();
    }


    /** Returns the part of yield code that havocs
	the shared state under the assumptions(OTIs).
	The returned code contains havocs of 
	variables in the OTI-closure of the decls in
	<code>vdVec</code>.
     */
    public GuardedCmd getHavocAssumes(GenericVarDeclVec vdVec, GenericVarDeclVec threadLocalVars) {
	GuardedCmdVec all = GuardedCmdVec.make();
	for(int i=0; i<vdVec.size(); i++) {
	    GenericVarDecl vd = vdVec.elementAt(i);
	    GuardedCmd gc = getHavocAssumes(vd, threadLocalVars);
	    if (!all.contains(gc)) {
		all.addElement(gc);
	    }
	}

	return GC.seq(all);
    }

    private GuardedCmd getAssertCode(int loc) {
	GuardedCmdVec assertCmdVec = GuardedCmdVec.make();

	for(int i=0; i< assertOtiVec.size(); i++) {
	  Condition cond = assertOtiVec.elementAt(i);
	  assertCmdVec.addElement(GC.check(loc, cond));
	}	
	for(int i=0; i< assertGiVec.size(); i++) {
	  Condition cond = assertGiVec.elementAt(i);
	  assertCmdVec.addElement(GC.check(loc, cond));
	}

	return GC.seq(assertCmdVec);
    }

    private GuardedCmd getAssertCode(int loc, GenericVarDecl vd) {
	GuardedCmdVec assertCmdVec = GuardedCmdVec.make();

	for(int i=0; i< assertOtiVec.size(); i++) {
	  Condition cond = assertOtiVec.elementAt(i);
	  if (mentionsVar(cond.pred, vd)) 
	      assertCmdVec.addElement(GC.check(loc, cond));
	}	
	for(int i=0; i< assertGiVec.size(); i++) {
	  Condition cond = assertGiVec.elementAt(i);
	  if (mentionsVar(cond.pred, vd)) 
	      assertCmdVec.addElement(GC.check(loc, cond));
	}

	return GC.seq(assertCmdVec);
    }

    public GuardedCmd getGiAssertCode(int loc) {
	GuardedCmdVec assertCmdVec = GuardedCmdVec.make();

	/*
	for(int i=0; i< assertOtiVec.size(); i++) {
	  Condition cond = assertOtiVec.elementAt(i);
	  if (mentionsVar(cond.pred, vd)) 
	      assertCmdVec.addElement(GC.check(loc, cond));
	}	
	*/
	for(int i=0; i< assertGiVec.size(); i++) {
	  Condition cond = assertGiVec.elementAt(i);
	  /*if (mentionsVar(cond.pred, vd)) */
	  assertCmdVec.addElement(GC.check(loc, cond));
	}

	return GC.seq(assertCmdVec);
    }


    private GuardedCmd getAssertCode(int loc, 
				     GenericVarDeclVec vdVec) {
	GuardedCmdVec assertCmdVec = GuardedCmdVec.make();

	ConditionVec otiAsserts = assertOtiVec.copy();
	ConditionVec giAsserts = assertGiVec.copy();
	
	for(int j=0; j < vdVec.size(); j++) {
	    GenericVarDecl vd = vdVec.elementAt(j);
	    ConditionVec tmpVec = ConditionVec.make();
	    for(int i=0; i< otiAsserts.size(); i++) {
		Condition cond = otiAsserts.elementAt(i);
		if (!tmpVec.contains(cond)) 
		    if (mentionsVar(cond.pred, vd)) { 
			assertCmdVec.addElement(GC.check(loc, cond));
			tmpVec.addElement(cond);
		    }
	    }
	    for(int i=0; i<tmpVec.size(); i++) {
		Condition cond = tmpVec.elementAt(i);
		otiAsserts.removeElement(cond);
	    }

	    tmpVec = ConditionVec.make();
	    for(int i=0; i< giAsserts.size(); i++) {
		Condition cond = giAsserts.elementAt(i);
		if (!tmpVec.contains(cond)) 
		    if (mentionsVar(cond.pred, vd)) {
			//			System.out.println("For variable " + vd.id.toString() + ", Asserting " + PrettyPrint.inst.toString(cond.pred)); 
			assertCmdVec.addElement(GC.check(loc, cond));
			tmpVec.addElement(cond);
		    }
	    }
	    for(int i=0; i<tmpVec.size(); i++) {
		Condition cond = tmpVec.elementAt(i);
		giAsserts.removeElement(cond);
	    }
	    tmpVec = null;
	}

	return GC.seq(assertCmdVec);
    }


    /** Assembles the desugaring of the yield to
	be placed at location <code> loc </code>.
	@param loc The location where the yield is placed
    */
    public GuardedCmd getYieldCode(int loc, 
				   GenericVarDecl vd, 
				   boolean isWrite, 
				   GenericVarDeclVec threadLocalVarVec) {
	GuardedCmd pyc = getPerformsYieldCode(loc);
	GuardedCmd assertCode = GC.skip();
	if (isWrite)
	    assertCode = getAssertCode(loc, vd);
	    
	return GC.seq(pyc, assertCode, 
		      getHavocAssumes(vd, threadLocalVarVec), 
		      GC.seq(oldAssigns));	  
    }

     /** Assembles the desugaring of the yield to
	be placed at location <code> loc </code>.
	@param loc The location where the yield is placed
    */
    public GuardedCmd getYieldCode(int loc, 
				   GenericVarDeclVec assertVars, 
				   GenericVarDeclVec assumeVars,
				   GenericVarDeclVec threadLocalVarVec) {
	GuardedCmd pyc = getPerformsYieldCode(loc);
	GuardedCmd assertCode = GC.skip();
	if (assertVars.size() > 0)
	    assertCode = getAssertCode(loc, assertVars);

	return GC.seq(pyc, assertCode, 
		      getHavocAssumes(assumeVars, threadLocalVarVec), 
		      GC.seq(oldAssigns));	  
    }

    public GuardedCmd getYieldCode(int loc,
				   GenericVarDeclVec threadLocalVarVec) {
	GuardedCmd pyc = getPerformsYieldCode(loc);
	GuardedCmd assertCode = getAssertCode(loc);
	return GC.seq(pyc, assertCode, 
		      getHavocAssumes(threadLocalVarVec),
		      GC.seq(oldAssigns));
    }

    public GuardedCmd getYieldCode(int loc, 
				   Set writeVars, 
				   Set readVars,
				   GenericVarDeclVec threadLocalVarVec) {
	GenericVarDeclVec assertVarVec = GenericVarDeclVec.make();
	GenericVarDeclVec assumeVarVec = GenericVarDeclVec.make();

	for (Enumeration e = readVars.elements(); e.hasMoreElements(); ) {
	    assumeVarVec.addElement((GenericVarDecl) e.nextElement());
	}

	for (Enumeration e = writeVars.elements(); e.hasMoreElements(); ) {
	    GenericVarDecl vd = (GenericVarDecl) e.nextElement();
	    if (!readVars.contains(vd)) {
		assumeVarVec.addElement(vd);
	    }
	    assertVarVec.addElement(vd);
	}

	return getYieldCode(loc, assertVarVec, assumeVarVec, threadLocalVarVec);
    }

    /** Checks if the invariant in <code>edp</code> occurs
	in <code>calleeII</code>
    */
    private  boolean isCalleeInv(InvariantInfo calleeII,
				 ExprDeclPragma edp) {
    /* There are unlikely to be very many INVs, so 
       this O(n) implementation seems OK */
	for(InvariantInfo ii = calleeII; ii != null; 
	    ii = ii.next) {
	    ExprDeclPragma cand = (ExprDeclPragma) ii.prag;
	    if (cand.expr == edp.expr) { 
		//		System.out.println("isCalleeInv: Found true " + PrettyPrint.inst.toString(cand.expr));
		return true;
	    }
	}
	return false;
    }


    private  ConditionVec callSiteAssertVec;
    /** Generates the subset of guarantees that must
	be asserted at the call site of a method.
	@param scope Caller scope
	@param calleeScope Callee scope
    */
    /* This method assumes that assertMap has already been filled, and
	it uses it. */
    //    private void genCallSiteAsserts(
    public void genCallSiteAsserts(/*@ non_null */ FindContributors calleeScope) {
	callSiteAssertVec = ConditionVec.make(); 

	/* The following loop generates the OTI asserts */
	InvariantInfo calleeOtis = GetSpec.collectOtis(calleeScope);
	for (InvariantInfo ii = GetSpec.collectOtis(scope); ii != null; ii = ii.next) {
	  ExprDeclPragma td = (ExprDeclPragma) ii.prag;
	  if (!isCalleeInv(calleeOtis, td)) {
	      int loc = td.getStartLoc();
	      Expr notTidConstraint = 
		  GC.and(TrAnExpr.typeAndNonNullAllocCorrectAs(GC.notThisThread.decl,
							       Types.getJavaLang("Thread"),
							       null,
							       true,
							       null,
							       true));
	  
	      notTidConstraint = GC.and(loc, loc, 
					notTidConstraint, 
					GC.nary(loc, loc, 
						TagConstants.REFNE, 
						GC.notThisThread, 
						GC.thisThread));
	      
	      Expr assertExpr = TrAnExpr.trSpecExpr(td.expr, TrAnExpr.union(tidMap, ii.map), assertMap);
	      assertExpr = GC.implies( notTidConstraint, assertExpr );

	      if (ii.isStatic) {
		  Expr f = GC.quantifiedExpr( loc, loc,
					      TagConstants.FORALL,
					      GC.notThisThread.decl,
					      assertExpr,
					      null );
		  Condition condn = GC.condition(TagConstants.CHKOTI,
						 f,
						 loc);
		  callSiteAssertVec.addElement(condn);

	      } else {
		  ExprVec ev = TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, ii.U,
								     null, true,
								     null, true /*false*/);
		  ExprVec nopats = ev.copy();	      
		  Expr ante = GC.and(ev);
	      
		  GenericVarDeclVec vs = GenericVarDeclVec.make();
		  vs.addElement(GC.notThisThread.decl);
		  vs.addElement(ii.sdecl);
		  Expr quantifiedAssertExpr = GC.quantifiedExpr( loc, loc,
								 TagConstants.FORALL,
								 vs,
								 GC.implies(ante, assertExpr),
							     nopats );
		  Condition condn = GC.condition(TagConstants.CHKOTI,
					     quantifiedAssertExpr,
					     loc);
		  callSiteAssertVec.addElement(condn);
	      
	      }
	  }
	}

	/* The following loop generates the GI asserts */
	InvariantInfo calleeGIs = GetSpec.collectGlobalInvariants(calleeScope);

      /* Generation of global invariants begins */
      for (InvariantInfo ii = GetSpec.collectGlobalInvariants(scope); 
	   ii != null; ii = ii.next) {
	  ExprDeclPragma td = (ExprDeclPragma) ii.prag;
	  if (!isCalleeInv(calleeGIs, td)) {
	      int loc = td.getStartLoc();

	      if (ii.isStatic) {
		  Condition condn = GC.condition(TagConstants.CHKGLOBALINVARIANT,
						 ii.J,
						 td.getStartLoc());
	      callSiteAssertVec.addElement(condn);
	      } else {
		  ExprVec ev = 
		      TrAnExpr.typeAndNonNullAllocCorrectAs(ii.sdecl, 
							    ii.U,
							    null, true,
							    null, true /*false*/);
		  ExprVec nopats = ev.copy();	      
		  Expr ante = GC.and(ev);
		  
		  Expr f = GC.quantifiedExpr( loc, loc,
					      TagConstants.FORALL,
					      ii.sdecl,
					      GC.implies(ante, ii.J),
					      nopats );
		  Condition condn = GC.condition(TagConstants.CHKGLOBALINVARIANT,
						 f,
						 td.getStartLoc());
		  callSiteAssertVec.addElement(condn);	      
	      }
	  }
      }
      return;
    }



    //    private boolean calledEarlier = false;
    /** Assembles the desugaring of the yield to be placed at location
<code> loc </code>, which is at the call site of a method with an
associated performs pragma
	@param loc The location where the yield is placed 
	@param calleeScope Scope of the method being called
    */
    /* Shaz: 22 Mar, 2002: not being used right now
    public GuardedCmd getCallSiteYieldCode(int loc,
					   FindContributors calleeScope) {
	genCallSiteAsserts(calleeScope);
	Assert.notFalse(callSiteAssertVec != null);

	GuardedCmdVec yieldCmdVec = GuardedCmdVec.make();

	yieldCmdVec.addElement(getPerformsYieldCode(loc));

	for(int i=0; i< callSiteAssertVec.size(); i++) {
	  Condition cond = callSiteAssertVec.elementAt(i);
	  yieldCmdVec.addElement(GC.check(loc, cond));
	}
	
	return GC.seq(GC.seq(yieldCmdVec), getHavocAssumes(), GC.seq(oldAssigns));	  
    }
    */
    public GuardedCmd desugarYields(GuardedCmd stmt, int loc) {	
	desugarYields(stmt, Reduction.ENDOFEXE, Reduction.ENDOFEXE);
	return stmt;
    }


    public int desugarYields(GuardedCmd stmt, int inN, int inE) {
	int tag = stmt.getTag();
	switch (tag) {
  	  case TagConstants.SKIPCMD:
	  case TagConstants.ASSERTCMD:
 	  case TagConstants.ASSUMECMD:    
          case TagConstants.GETSCMD: 
	  case TagConstants.SUBGETSCMD: 
 	  case TagConstants.SUBSUBGETSCMD:
	  case TagConstants.RESTOREFROMCMD: {
	      return inN;
	  }
    
	  case TagConstants.RAISECMD : {
	      return inE;
	  }

	  case TagConstants.YIELDCMD: {
	      YieldCmd yc = (YieldCmd)stmt;

	      if (inN == Reduction.ENDOFEXE ||
		  ((inN == Reduction.NONMOVER || inN == Reduction.RIGHTMOVER) &&
		   (yc.mark == Reduction.NONMOVER || yc.mark == Reduction.LEFTMOVER))) {
		  GuardedCmd gc = null;
		  GenericVarDeclVec threadLocalVars = genThreadLocalVariables(yc.localVarDecls);
		  if (Main.reduce) {
		      gc = getYieldCode(yc.loc, threadLocalVars);
		  } else {
		      gc = getYieldCode(yc.loc, 
					yc.writeVars, 
					yc.readVars, 
					threadLocalVars);
		  }
		  yc.desugared = GC.seq(yc.code, gc);
	      } else {
		  yc.desugared = yc.code;
	      }

	      yc.inN = inN;
	      yc.inE = inE;

	      return yc.mark;		  
	  }

	  case TagConstants.CALL : {
	      Call call = (Call)stmt;
	      return desugarYields(call.desugared, inN, inE);
	  }

	  case TagConstants.VARINCMD : {
	      VarInCmd vc = (VarInCmd)stmt;
	      return desugarYields(vc.g, inN, inE);	    
	  }

  	  case TagConstants.DYNINSTCMD : { 
	      DynInstCmd dc = (DynInstCmd)stmt;
	      return desugarYields(dc.g, inN, inE);
	  }

 	  case TagConstants.SEQCMD : {
	      SeqCmd sc = (SeqCmd)stmt;
	      int temp = inN;

	      for (int i = sc.cmds.size() - 1; i >= 0; i--) {
		  temp = desugarYields(sc.cmds.elementAt(i), temp, inE);		  
	      }

	      return temp;
	  }

	  case TagConstants.TRYCMD: {
	      CmdCmdCmd tc = (CmdCmdCmd)stmt;

	      return desugarYields(tc.g1, inN, desugarYields(tc.g2, inN, inE));
	  }

	  case TagConstants.CHOOSECMD: {
	      CmdCmdCmd cc = (CmdCmdCmd)stmt;

	      int temp1 = desugarYields(cc.g1, inN, inE);
	      int temp2 = desugarYields(cc.g2, inN, inE);
	      if (temp1 == Reduction.ENDOFEXE || temp2 == Reduction.ENDOFEXE) 
		  return Reduction.ENDOFEXE;
	      if (temp1 == Reduction.LEFTMOVER && temp2 == Reduction.LEFTMOVER)
		  return Reduction.LEFTMOVER;
	      if (temp1 == Reduction.RIGHTMOVER && temp2 == Reduction.RIGHTMOVER)
		  return Reduction.RIGHTMOVER;
	      return Reduction.NONMOVER;
	  }

	  case TagConstants.LOOPCMD : {
	      /* This currently works only for loop unrolling */
	      LoopCmd loop = (LoopCmd) stmt;
	      return desugarYields(loop.desugared, inN, inE);
	  }

	  default : {
	      Assert.fail("Fall through on " + stmt);
	      return -1;
	  }
	}
    }

    static GenericVarDeclVec genThreadLocalVariables(FormalParaDeclVec decls) {
	GenericVarDeclVec threadLocalVars = GenericVarDeclVec.make();
	for (int i = 0; i < decls.size(); i++) {
	    GenericVarDecl vd = (GenericVarDecl) decls.elementAt(i);
	    if (vd.pmodifiers != null) {
		for (int j = 0; j < vd.pmodifiers.size(); j++) {
		    ModifierPragma mp = vd.pmodifiers.elementAt(j);
		    if (mp.getTag() == TagConstants.THREAD_LOCAL) {
			threadLocalVars.addElement(vd);
			break;
		    }			  
		}
	    }
	}
	return threadLocalVars;
    }

    static GenericVarDeclVec genThreadLocalVariables(Vector decls) {
	GenericVarDeclVec threadLocalVars = GenericVarDeclVec.make();
	for (int i = 0; i < decls.size(); i++) {
	    GenericVarDecl vd = (GenericVarDecl) decls.elementAt(i);
	    if (vd.pmodifiers != null) {
		for (int j = 0; j < vd.pmodifiers.size(); j++) {
		    ModifierPragma mp = vd.pmodifiers.elementAt(j);
		    if (mp.getTag() == TagConstants.THREAD_LOCAL) {
			threadLocalVars.addElement(vd);
			break;
		    }			  
		}
	    }
	}	    
	return threadLocalVars;
    }     

    /*
    void desugarYields(GuardedCmd stmt, Set in, Set[] out) {

	   in[0] : incoming readVars
	   out[0] : normal outgoing readVars
	   out[1] : exceptional outgoing readVars


	int tag = stmt.getTag();
	switch (tag) {
  	  case TagConstants.SKIPCMD:
	  case TagConstants.ASSERTCMD:
 	  case TagConstants.ASSUMECMD:    
          case TagConstants.GETSCMD: 
	  case TagConstants.SUBGETSCMD: 
 	  case TagConstants.SUBSUBGETSCMD:
	  case TagConstants.RESTOREFROMCMD: {
	      out[0] = (Set) in.clone();
	      out[1] = new Set();
	      return;
	  }
    
	  case TagConstants.RAISECMD : {
	      out[0] = new Set();
	      out[1] = (Set) in.clone();
	      return;
	  }

	  case TagConstants.YIELDCMD: {
	      YieldCmd yc = (YieldCmd)stmt;
	      if (!yc.mark) {
		  GenericVarDeclVec assertVarVec = GenericVarDeclVec.make();
		  GenericVarDeclVec assumeVarVec = GenericVarDeclVec.make();
		  
		  for (Enumeration e = in.elements(); e.hasMoreElements(); ) {
		      assumeVarVec.addElement((GenericVarDecl) e.nextElement());
		  }
		  
		  for (Enumeration e = yc.writeVars.elements(); e.hasMoreElements(); ) {
		      GenericVarDecl vd = (GenericVarDecl) e.nextElement();
		      assertVarVec.addElement(vd);
		  }

		  GuardedCmdVec gcVec = GuardedCmdVec.make();
		  gcVec.addElement(getHavocAssumes(assumeVarVec));
		  gcVec.addElement(getAssignCode());
		  gcVec.addElement(yc.code);
		  gcVec.addElement(getAssertCode(yc.loc, assertVarVec));
		  gcVec.addElement(getPerformsYieldCode(yc.loc));

		  yc.desugared = GC.seq(gcVec);
		  out[0] = new Set();
	      } else {
		  yc.desugared = yc.code;
		  out[0] = (Set) in.clone();
	      }
	      
	      out[0].union(yc.readVars);
	      out[0].union(yc.writeVars);
	      out[1] = new Set();

	      return;		  
	  }

	  case TagConstants.CALL : {
	      Call call = (Call)stmt;
	      desugarYields(call.desugared, in, out);
	      return;
	  }

	  case TagConstants.VARINCMD : {
	      VarInCmd vc = (VarInCmd)stmt;
	      desugarYields(vc.g, in, out);	    
	      return;
	  }

  	  case TagConstants.DYNINSTCMD : { 
	      DynInstCmd dc = (DynInstCmd)stmt;
	      desugarYields(dc.g, in, out);
	      return;
	  }

 	  case TagConstants.SEQCMD : {
	      SeqCmd sc = (SeqCmd)stmt;
	      Set[] tout = new Set[2];

	      out[0] = in;
	      out[1] = new Set();

	      for (int i = 0; i < sc.cmds.size(); i++) {
		  desugarYields(sc.cmds.elementAt(i), out[0], tout);
		  out[0] = tout[0];
		  out[1].union(tout[1]);
	      }
	      return;
	  }

	  case TagConstants.TRYCMD: {
	      CmdCmdCmd tc = (CmdCmdCmd)stmt;
	      Set[] tout = new Set[2];

	      desugarYields(tc.g1, in, tout);
	      
	      out[0] = tout[0];
	      out[1] = tout[1];

	      desugarYields(tc.g2, out[1], tout);

	      out[0].union(tout[0]);
	      out[1] = tout[1];
	      return;
	  }

	  case TagConstants.CHOOSECMD: {
	      CmdCmdCmd cc = (CmdCmdCmd)stmt;

	      desugarYields(cc.g1, in, out);
	      
	      Set[] tout = new Set[2];
	      desugarYields(cc.g2, in, tout);

	      out[0].union(tout[0]);
	      out[1].union(tout[1]);
	      return;
	  }

	  case TagConstants.LOOPCMD : {
	      LoopCmd loop = (LoopCmd) stmt;
	      desugarYields(loop.desugared, in, out);
	      return;
	  }

	  default : {
	      Assert.fail("Fall through on " + stmt);
	  }
	}
    }
    */
} // end of class



	  /*  The code for creating another version of the inferred OTI, now obsolete. 
	    // Create the existentially quantified disjunct
	    ExprVec newEv = ExprVec.make();
	    for (int i = 0; i < ev.size(); i++) {
	    newEv.addElement(Substitute.doSubst(tidMap, ev.elementAt(i)));
	    }
	    
	    newEv.append(TrAnExpr.typeAndNonNullAllocCorrectAs(GC.notThisThread.decl,
	    Types.getJavaLang("Thread"),
	    null,
	    true,
	    null,
	    true));
	    
	    // newEv.addElement(GC.nary(TagConstants.REFNE, notTidva, GC.nulllit));
	    newEv.addElement(GC.nary(TagConstants.REFNE, GC.notThisThread, GC.thisThread));
	    disjunct = GC.quantifiedExpr( loc, loc,
	    TagConstants.EXISTS,
	    GC.notThisThread.decl,
	    GC.and(newEv),
	    null );	      
	  */

	/*
	Expr e;
	e = GC.nary(TagConstants.INTEGRALEQ, GC.zerolit, oldWitness);
	e = GC.and(e,
		   GC.or(GC.nary(TagConstants.INTEGRALEQ, GC.zerolit, witness),
			 GC.nary(TagConstants.INTEGRALEQ, GC.onelit, witness)));
	e = GC.or(e,
		  GC.and(GC.nary(TagConstants.INTEGRALEQ, GC.onelit, oldWitness),
			 GC.nary(TagConstants.INTEGRALEQ, GC.onelit, witness)));
	
	// need a pragma location for this condition. get the location of the first amp.
	legalWitnessCond = GC.condition(TagConstants.CHKPERFORMS, e, performsPragma.loc);
	
	oldWitnessChoiceVec = GuardedCmdVec.make();
	oldWitnessChoiceVec.addElement(GC.assume(GC.nary(TagConstants.ANYEQ, 
							 (VariableAccess) assertMap.get(GC.witness.decl), 
							 GC.zerolit)));
	oldWitnessChoiceVec.addElement(GC.assume(GC.nary(TagConstants.ANYEQ, 
							 (VariableAccess) assertMap.get(GC.witness.decl), 
							 GC.onelit)));
	
	witnessChoiceVec = GuardedCmdVec.make();
	witnessChoiceVec.addElement(GC.assume(GC.nary(TagConstants.ANYEQ, 
						      GC.witness, 
						      GC.zerolit)));
	witnessChoiceVec.addElement(GC.assume(GC.nary(TagConstants.ANYEQ, 
						      GC.witness, 
						      GC.onelit)));
	

	Expr stutter = GC.truelit;


	PerformsAction samp = (PerformsAction) performsPragma.stmt;
	// The predicate that needs to be checked 
	// substitute all the "\old"s out 
	Expr pred = TrAnExpr.trSpecExpr(samp.pred, null, assertMap);
	performsPredCond = GC.condition(TagConstants.CHKPERFORMS, pred, performsPragma.loc);
	
	// generate conditions of the form \old(v) = v 
	//   These will be used both in assumes and asserts.
	//   To do this we will need to collect all the 
	//   fields and variables in amp.exprs. These must
	//   occur already in the list generated by 
	//   scope.fields(), and so the old values already 
	//   generated by computeOtiGiCode will work.
	for(int j = 0; j< samp.exprs.size(); j++) {
	    e = samp.exprs.elementAt(j);
	    
	    Expr newE = TrAnExpr.trSpecExpr(e);
	    Expr oldE = TrAnExpr.trSpecExpr(e, assertMap, assertMap);
	    stutter = GC.and(stutter, GC.nary(TagConstants.ANYEQ, oldE, newE));
	} 

	stutterCond = GC.condition(TagConstants.CHKPERFORMS, stutter, performsPragma.loc);
	*/
    /*
    private Condition legalWitnessCond;
    private GuardedCmdVec oldWitnessChoiceVec;
    private GuardedCmdVec witnessChoiceVec;
    private Condition stutterCond;
    private Condition performsPredCond;
    private Condition endWitnessCond;
    */
	    /*
	    cmdVec.addElement(GC.check(loc, legalWitnessCond));
	    GuardedCmd c0 = 
		GC.seq(oldWitnessChoiceVec.elementAt(0), 
		       GC.choosecmd(GC.seq(witnessChoiceVec.elementAt(0), 
					   GC.check(loc, stutterCond)),
				    GC.seq(witnessChoiceVec.elementAt(1), 
					   GC.check(loc, performsPredCond))));
	    GuardedCmd c1 = GC.seq(oldWitnessChoiceVec.elementAt(1), 
				   GC.check(loc, stutterCond));
	    cmdVec.addElement(GC.choosecmd(c0, c1));
	    */
