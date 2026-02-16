/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

import java.util.Hashtable;
import java.util.Enumeration;
import java.io.ByteArrayOutputStream;

import javafe.util.Location;
import javafe.ast.*;
import javafe.tc.TypeSig;
import javafe.tc.Types;
import javafe.tc.TagConstants;
import javafe.util.Assert;
import escjava.ast.*;

final class HavocAssume {

    private Hashtable havocMap;
    private GuardedCmdVec assumeBeforeHavoc;

    //    private GuardedCmd havocAssumeCmd; 
    /* guarded command that does havocs and then assumes OTIs, 
       global_invariants, etc. */

    /** reference for linking HavocAssume objects in a linked list */
    HavocAssume next;

    HavocAssume(GuardedCmd cmd, Hashtable map) {
	assumeBeforeHavoc = GuardedCmdVec.make();
	assumeBeforeHavoc.addElement(cmd);
	havocMap = map;
	
	//	havocAssumeCmd = null;
	next = null;
    }

    boolean merge(HavocAssume ha) {
	boolean intersection = false;
	Enumeration decls = havocMap.keys();
	while (decls.hasMoreElements()) {
	    GenericVarDecl decl = (GenericVarDecl) decls.nextElement();
	    if (ha.havocMap.containsKey(decl)) {
		intersection = true;
		break;
	    }
	}
	
	if (intersection) {
	    havocMap.putAll(ha.havocMap);
	    assumeBeforeHavoc.append(ha.assumeBeforeHavoc);
	}

	return intersection;
    }
    
    private GuardedCmdVec protectThreadLocalArrays(GenericVarDeclVec threadLocalVars, 
						   VariableAccess va, 
						   VariableAccess oldva) {
	GuardedCmdVec resultVec = GuardedCmdVec.make();

	for (int i = 0; i < threadLocalVars.size(); i++) {
	    GenericVarDecl vd = (GenericVarDecl) threadLocalVars.elementAt(i);
	    Assert.notFalse(!(vd.type instanceof PrimitiveType || vd.type instanceof MapType));
	    Type type = vd.type;
	    int tag = type.getTag();

	    if (tag == javafe.ast.TagConstants.ARRAYTYPE) {
		VariableAccess obj = VariableAccess.make(vd.id, vd.locId, vd);
		Expr e = GC.select(va, obj);
		Expr olde = GC.select(oldva, obj);				
		Expr result = GC.nary(escjava.ast.TagConstants.ANYEQ, e, olde);
		type = ((ArrayType)type).elemType;		
		if (type.getTag() == javafe.ast.TagConstants.ARRAYTYPE) {
		    ExprVec exprVec = ExprVec.make();
		    ExprVec nopats = ExprVec.make();
		    exprVec.addElement(result);
		    GenericVarDeclVec vdVec = GenericVarDeclVec.make();
		    //		    int index = 1; 
		    // incremented each time, used for disambiguating various quantified variables generated
		    while (type.getTag() == javafe.ast.TagConstants.ARRAYTYPE) {
			LocalVarDecl sdecl = LocalVarDecl.make(Modifiers.NONE,  
							       null,            
							       Identifier.intern(vd.id.toString() + 
										 "$thread_local"),
							       Types.intType,
							       vd.locId,
							       null, 
							       Location.NULL); 
			vdVec.addElement(sdecl);
			VariableAccess sdeclva = TrAnExpr.makeVarAccess(sdecl, Location.NULL);
			nopats.addElement(GC.nary(escjava.ast.TagConstants.IS, 
						  sdeclva, 
						  GC.typeExpr(Types.intType)));
			e = GC.select(e, sdeclva);
			olde = GC.select(olde, sdeclva);
			exprVec.addElement(GC.nary(escjava.ast.TagConstants.ANYEQ, e, olde)); 
			type = ((ArrayType)type).elemType;		
			//			index++;
		    }	    
		    result = GC.quantifiedExpr( Location.NULL,
						Location.NULL,
						escjava.ast.TagConstants.FORALL,
						vdVec,
						GC.and(exprVec),						
						nopats );		    
		}
		resultVec.addElement(GC.assume(result));
	    }	    
	}

	return resultVec;
    }

    private GuardedCmdVec  protectThreadLocalObjects(TypeSig fieldParent, 
						     GenericVarDeclVec threadLocalVars,
						     VariableAccess va,
						     VariableAccess oldva) {
	GuardedCmdVec resultVec = GuardedCmdVec.make();

	for (int i = 0; i < threadLocalVars.size(); i++) {
	    GenericVarDecl vd = (GenericVarDecl) threadLocalVars.elementAt(i);
	    TypeSig sig;
	    Assert.notFalse(!(vd.type instanceof PrimitiveType || vd.type instanceof MapType));
	    int tag = vd.type.getTag();
	    if (tag == javafe.ast.TagConstants.TYPENAME) {
		sig = TypeSig.getSig((TypeName) vd.type);
	    } else if (tag == javafe.tc.TagConstants.TYPESIG) {
		sig = (TypeSig) vd.type;
	    } else {
		continue;
	    }
	    if (sig.isSubtypeOf(fieldParent)) {
		VariableAccess obj = VariableAccess.make(vd.id, vd.locId, vd);		
		resultVec.addElement(GC.assume(GC.nary(escjava.ast.TagConstants.ANYEQ, 
						       GC.select(va, obj), 
						       GC.select(oldva, obj))));
	    }
	}
	
	return resultVec;
    }


    private GuardedCmd finish(GenericVarDeclVec threadLocalVars) {
	GenericVarDeclVec havocVec = GenericVarDeclVec.make();
	GuardedCmdVec havocAssign = GuardedCmdVec.make();
	GuardedCmdVec assumeAfterHavoc = GuardedCmdVec.make();
	GuardedCmdVec assumeMoreBeforeHavoc = GuardedCmdVec.make();

	/* create a map to do the right thing for 
	   targetTypeCorrect (note that allocvar is
	   the only thing for which the map passed to
	   targetTypeCorrect makes any difference
	*/
	Hashtable allocMap = new Hashtable();
	VariableAccess allocTmpVA = (VariableAccess) havocMap.get(GC.allocvar.decl);
	if (allocTmpVA != null) {
	    allocMap.put(GC.allocvar.decl, allocTmpVA);
	    allocMap.put(allocTmpVA.decl, GC.allocvar);
	}

	Enumeration decls = havocMap.keys();
	while (decls.hasMoreElements()) {
	    GenericVarDecl decl = (GenericVarDecl) decls.nextElement();
	    VariableAccess va = VariableAccess.make(decl.id, decl.locId, decl);
	    VariableAccess oldva = (VariableAccess) havocMap.get(decl);
	    havocAssign.addElement(GC.gets(va, oldva));
	    havocVec.addElement(oldva.decl);	    
	    
	    // generate the assume for thread-local variables
	    if (decl == GC.elemsvar.decl) {
		assumeMoreBeforeHavoc.append(protectThreadLocalArrays(threadLocalVars, 
								      va, 
								      oldva));
	    } else if (decl instanceof FieldDecl) {
		Assert.notFalse(decl != GC.allocvar.decl);
		FieldDecl fd = (FieldDecl) decl;
		assumeMoreBeforeHavoc.append(protectThreadLocalObjects(TypeSig.getSig(fd.getParent()),
								       threadLocalVars, 
								       va, 
								       oldva));
	    }

	    Expr expr = TrAnExpr.targetTypeCorrect(decl, havocMap);
	    if (GC.allocvar.decl == decl) {
		Assert.notFalse(allocTmpVA != null);
		expr = Substitute.doSubst(allocMap, expr);
		assumeMoreBeforeHavoc.addElement(GC.assume(expr));
	    }
	    else {
		assumeAfterHavoc.addElement(GC.assume(expr));
	    }
	}

	GuardedCmdVec all = GuardedCmdVec.make();
	all.append(assumeBeforeHavoc);
	all.append(assumeMoreBeforeHavoc);
	all.append(havocAssign);
	all.append(assumeAfterHavoc);
	return GC.block(havocVec, GC.seq(all));
    }

    GuardedCmd getCmd(GenericVarDeclVec threadLocalVars) {
	return finish(threadLocalVars);
    }

    Hashtable getHavocMap() {
	return havocMap;
    }

    /** Returns true if this HavocAssume refers to vd,
	false o.w. */
    boolean refersTo(GenericVarDecl vd) {
	return ((havocMap.get(vd)) != null);
    }
    

}

