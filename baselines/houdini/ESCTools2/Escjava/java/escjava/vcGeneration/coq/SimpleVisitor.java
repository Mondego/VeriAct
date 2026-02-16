package escjava.vcGeneration.coq;

import java.io.Writer;
import java.util.Iterator;

import escjava.vcGeneration.TAllocLE;
import escjava.vcGeneration.TAllocLT;
import escjava.vcGeneration.TAnyEQ;
import escjava.vcGeneration.TAnyNE;
import escjava.vcGeneration.TArrayFresh;
import escjava.vcGeneration.TArrayLength;
import escjava.vcGeneration.TArrayShapeMore;
import escjava.vcGeneration.TArrayShapeOne;
import escjava.vcGeneration.TAsElems;
import escjava.vcGeneration.TAsField;
import escjava.vcGeneration.TAsLockSet;
import escjava.vcGeneration.TBoolAnd;
import escjava.vcGeneration.TBoolEQ;
import escjava.vcGeneration.TBoolImplies;
import escjava.vcGeneration.TBoolNE;
import escjava.vcGeneration.TBoolNot;
import escjava.vcGeneration.TBoolOr;
import escjava.vcGeneration.TBoolean;
import escjava.vcGeneration.TCast;
import escjava.vcGeneration.TChar;
import escjava.vcGeneration.TDouble;
import escjava.vcGeneration.TEClosedTime;
import escjava.vcGeneration.TExist;
import escjava.vcGeneration.TFClosedTime;
import escjava.vcGeneration.TFloat;
import escjava.vcGeneration.TFloatAdd;
import escjava.vcGeneration.TFloatDiv;
import escjava.vcGeneration.TFloatEQ;
import escjava.vcGeneration.TFloatGE;
import escjava.vcGeneration.TFloatGT;
import escjava.vcGeneration.TFloatLE;
import escjava.vcGeneration.TFloatLT;
import escjava.vcGeneration.TFloatMod;
import escjava.vcGeneration.TFloatMul;
import escjava.vcGeneration.TFloatNE;
import escjava.vcGeneration.TForAll;
import escjava.vcGeneration.TFunction;
import escjava.vcGeneration.TInt;
import escjava.vcGeneration.TIntegralAdd;
import escjava.vcGeneration.TIntegralDiv;
import escjava.vcGeneration.TIntegralEQ;
import escjava.vcGeneration.TIntegralGE;
import escjava.vcGeneration.TIntegralGT;
import escjava.vcGeneration.TIntegralLE;
import escjava.vcGeneration.TIntegralLT;
import escjava.vcGeneration.TIntegralMod;
import escjava.vcGeneration.TIntegralMul;
import escjava.vcGeneration.TIntegralNE;
import escjava.vcGeneration.TIntegralSub;
import escjava.vcGeneration.TIs;
import escjava.vcGeneration.TIsAllocated;
import escjava.vcGeneration.TIsNewArray;
import escjava.vcGeneration.TLiteral;
import escjava.vcGeneration.TLockLE;
import escjava.vcGeneration.TLockLT;
import escjava.vcGeneration.TMethodCall;
import escjava.vcGeneration.TName;
import escjava.vcGeneration.TNode;
import escjava.vcGeneration.TNull;
import escjava.vcGeneration.TRefEQ;
import escjava.vcGeneration.TRefNE;
import escjava.vcGeneration.TRoot;
import escjava.vcGeneration.TSelect;
import escjava.vcGeneration.TStore;
import escjava.vcGeneration.TString;
import escjava.vcGeneration.TSum;
import escjava.vcGeneration.TTypeEQ;
import escjava.vcGeneration.TTypeLE;
import escjava.vcGeneration.TTypeNE;
import escjava.vcGeneration.TTypeOf;
import escjava.vcGeneration.TUnset;
import escjava.vcGeneration.coq.visitor.ABasicCoqVisitor;

public class SimpleVisitor extends ABasicCoqVisitor {

	protected SimpleVisitor(Writer out, CoqProver prover) {
		super(out, prover, null);
		this.tcbv = this;
		this.tcv = this;
	}
	
	// FIXME @ ensures vc == vc ++ [[s]] ++ 
    public void genericFun(/*@ non_null @*/ String s, TFunction n) {
    	out.appendI(s+" ");
    	// FIXME @ list args <- sons
    	int i =0;
    	
    	/* FIXME @ loop_invariant 
    	  @		\forall int i i in n.sons
    	  @ 			vc ++
    	  @*/
    	for(; i < n.sons.size(); i++) {
    		n.getChildAt(i).accept(tcv);
    		if(i != n.sons.size() - 1)
    			out.appendN(" ");
    	}


    	if((n.getChildAt(--i)) instanceof TName || (n.getChildAt(--i)) instanceof TLiteral)
    		out.reduceIwNl();
    	else
    		out.reduceI();    
    }
	// FIXME @ ensures vc == vc ++ (allocLE interp(arg1), interp(arg2))
	public void visitTAllocLE(/*@non_null*/ TAllocLE n) {
		genericFun("allocLE", n);
	}

	public void visitTAllocLT(/*@non_null*/ TAllocLT n) {
		genericFun("allocLT", n);		
	}

	public void visitTAnyEQ(/*@non_null*/ TAnyEQ n) {
		genericFun("anyEQ", n);	
	}

	public void visitTAnyNE(/*@non_null*/ TAnyNE n) {
		genericFun("anyNE", n);	
		
	}

	public void visitTArrayFresh(/*@non_null*/ TArrayFresh n) {
		genericFun("arrayFresh", n);
		
	}

	public void visitTArrayLength(/*@non_null*/ TArrayLength n) {
		genericFun("arrayLength", n);
	}

	public void visitTArrayShapeMore(/*@non_null*/ TArrayShapeMore n) {
		genericFun("arrayShapeMore", n);
	}

	public void visitTArrayShapeOne(/*@non_null*/ TArrayShapeOne n) {
		genericFun("arrayShapeOne", n);
	}

	public void visitTAsElems(/*@non_null*/ TAsElems n) {
		genericFun("asElems", n);
		
	}

	public void visitTAsField(/*@non_null*/ TAsField n)  {
		genericFun("asField", n);
	}

	public void visitTAsLockSet(/*@non_null*/ TAsLockSet n) {
		genericFun("asLockSet", n);
	}

	public void visitTBoolAnd(/*@non_null*/ TBoolAnd n) {
		genericFun("band", n);
	}

	public void visitTBoolEQ(/*@non_null*/ TBoolEQ n) {
		genericFun("beq", n);
	}

	public void visitTBoolImplies(/*@non_null*/ TBoolImplies n) {
		genericFun("bimplies", n);
	}

	public void visitTBoolNE(/*@non_null*/ TBoolNE n) {
		genericFun("bne", n);
	}

	public void visitTBoolNot(/*@non_null*/ TBoolNot n) {
		genericFun("bnot", n);
		
	}

	public void visitTBoolOr(/*@non_null*/ TBoolOr n) {
		genericFun("bor", n);
	}

	public void visitTBoolean(/*@non_null*/ TBoolean n) {
		out.appendN(""+ n.value);
	}

	public void visitTCast(/*@non_null*/ TCast n) {
		genericFun("cast", n);
		
	}

	public void visitTChar(/*@non_null*/ TChar n) {
		out.appendN(""+ n.value);
	}

	public void visitTDouble(/*@non_null*/ TDouble n) {
		out.appendN(""+ n.value);
	}

	public void visitTEClosedTime(/*@non_null*/ TEClosedTime n)  {
		genericFun("eClosedTime", n);
	}

	public void visitTExist(/*@non_null*/ TExist n)  {
		genericFun("exist ", n);
	}

	public void visitTFClosedTime(/*@non_null*/ TFClosedTime n)  {
		genericFun("fClosedTime", n);
	}

	public void visitTFloat(/*@non_null*/ TFloat n)  {
		out.appendN(""+ n.value);		
	}

	public void visitTFloatAdd(/*@non_null*/ TFloatAdd n)  {
		genericFun("fAdd", n);
		
	}

	public void visitTFloatDiv(/*@non_null*/ TFloatDiv n) {
		genericFun("fDiv", n);
		
	}

	public void visitTFloatEQ(/*@non_null*/ TFloatEQ n) {
		genericFun("fEQ", n);	
	}

	public void visitTFloatGE(/*@non_null*/ TFloatGE n) {
		genericFun("fGE", n);
	}

	public void visitTFloatGT(/*@non_null*/ TFloatGT n) {
		genericFun("fGT", n);
	}

	public void visitTFloatLE(/*@non_null*/ TFloatLE n) {
		genericFun("fLE", n);
	}

	public void visitTFloatLT(/*@non_null*/ TFloatLT n)  {
		genericFun("fLT", n);
	}

	public void visitTFloatMod(/*@non_null*/ TFloatMod n)  {
		genericFun("fMod", n);
	}

	public void visitTFloatMul(/*@non_null*/ TFloatMul n)  {
		genericFun("fMul", n);
	}

	public void visitTFloatNE(/*@non_null*/ TFloatNE n)  {
		genericFun("fNE", n);
	}

	public void visitTForAll(/*@non_null*/ TForAll n)  {
		genericFun("bforall", n);
	}

	public void visitTInt(/*@non_null*/ TInt n)  {
		out.appendN(""+ n.value);
	}

	public void visitTIntegralAdd(/*@non_null*/ TIntegralAdd n)  {
		genericFun("iAdd", n);
		
	}

	public void visitTIntegralDiv(/*@non_null*/ TIntegralDiv n)  {
		genericFun("iDiv", n);
	}

	public void visitTIntegralEQ(/*@non_null*/ TIntegralEQ n)  {
		genericFun("iEQ", n);
	}

	public void visitTIntegralGE(/*@non_null*/ TIntegralGE n)  {
		genericFun("iGE", n);
	}

	public void visitTIntegralGT(/*@non_null*/ TIntegralGT n)  {
		genericFun("iGT", n);
	}

	public void visitTIntegralLE(/*@non_null*/ TIntegralLE n)  {
		genericFun("iLE", n);
	}

	public void visitTIntegralLT(/*@non_null*/ TIntegralLT n)  {
		genericFun("iLT", n);
	}

	public void visitTIntegralMod(/*@non_null*/ TIntegralMod n)  {
		genericFun("iMod", n);
	}

	public void visitTIntegralMul(/*@non_null*/ TIntegralMul n)  {
		genericFun("iMul", n);
		
	}

	public void visitTIntegralNE(/*@non_null*/ TIntegralNE n)  {
		genericFun("iNE", n);
		
	}

	public void visitTIntegralSub(/*@non_null*/ TIntegralSub n)  {
		genericFun("iSub", n);
	}

	public void visitTIs(/*@non_null*/ TIs n)  {
		genericFun("is", n);
	}

	public void visitTIsAllocated(/*@non_null*/ TIsAllocated n)  {
		genericFun("bIsAllocated", n);
	}

	public void visitTIsNewArray(/*@non_null*/ TIsNewArray n)  {
		genericFun("isNewArray", n);		
	}

	public void visitTLockLE(/*@non_null*/ TLockLE n)  {
		genericFun("lockLE", n);
	}

	public void visitTLockLT(/*@non_null*/ TLockLT n)  {
		genericFun("lockLT", n);
	}

	public void visitTMethodCall(/*@non_null*/ TMethodCall call)  {
		genericFun(p.getVariableInfo(call.getName()), call);
	}

	public void visitTName(/*@non_null*/ TName n)  {
		if(TNode._boolean.equals(n.type)){
			out.appendN("(bvar \"" + n.name + "\")");
		}
		else if(TNode._Path.equals(n.type)) {
			out.appendN("(pvar \"" + n.name + "\")");
		}
		else if(TNode._Reference.equals(n.type)) {
			out.appendN("(rvar \"" + n.name + "\")");
		}
		else if(TNode._Time.equals(n.type)) {
			out.appendN("(tmvar \"" + n.name + "\")");
		}
		else if(TNode._Type.equals(n.type)) {
			out.appendN("(tvar \"" + n.name + "\")");
		}
		else {
			out.appendN("(var \"" + n.name + "\")");
		}
	}

	public void visitTNull(/*@non_null*/ TNull n)  {
		out.appendN("null");
	}

	public void visitTRefEQ(/*@non_null*/ TRefEQ n)  {
		genericFun("refEQ", n);
		
	}

	public void visitTRefNE(/*@non_null*/ TRefNE n)  {
		genericFun("refNE", n);
		
	}

	public void visitTRoot(/*@non_null*/ TRoot n)  {
		Iterator iter = n.sons.iterator();
    	while(iter.hasNext()) {
    		((TNode)iter.next()).accept(tcv);
    	}
	}

	public void visitTTypeEQ(/*@non_null*/ TTypeEQ n)  {
		genericFun("typeEQ", n);
		
	}

	public void visitTTypeLE(/*@non_null*/ TTypeLE n)  {
		genericFun("typeLE", n);
		
	}

	public void visitTTypeNE(/*@non_null*/ TTypeNE n)  {
		genericFun("typeNE", n);
		
	}

	public void visitTTypeOf(/*@non_null*/ TTypeOf n)  {
		genericFun("typeof", n);	
	}
	
	public void visitTSum(/*@non_null*/ TSum n) {
		genericFun("sum", n);
	}
	
	public void visitTUnset(/*@non_null*/ TUnset n)  {
		genericFun("unset", n);	
	}

	public void visitTString(/*@non_null*/ TString n)  {
		out.appendN("\""+ n.value +"\"");
	}
	

    public void visitTSelect(/*@non_null*/ TSelect n) {
    	String pre = "";
    	if(TNode._integer.equals(((TNode)n.sons.get(1)).type))
    		pre = "arr";
    	genericFun("Heap." +pre +"select ", n);
    }
    
    
    public void visitTStore(/*@non_null*/ TStore n) {
    	String pre = "";
    	TNode index =(TNode)n.sons.get(1);
    	if(TNode._integer.equals(index.type))
    		pre = "arr";
    	genericFun("Heap." + pre + "store ", n);
    }

}
