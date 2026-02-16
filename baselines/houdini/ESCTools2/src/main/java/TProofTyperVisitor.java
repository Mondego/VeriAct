package escjava.vcGeneration.coq.visitor.simplifiers;

import java.io.IOException;

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
import escjava.vcGeneration.TDisplay;
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
import escjava.vcGeneration.TVisitor;
import escjava.vcGeneration.TypeInfo;
import escjava.vcGeneration.VariableInfo;

/**
 * Coq extension needs 'clear' typing.
 * This retyping is done using the known function signatures.
 * This is done without any subtleties.
 * @author J. Charles
 * @version last update: 15/11/2005
 */
public class TProofTyperVisitor extends TVisitor {
	
    public TProofTyperVisitor() {
        // Since this visitor modifies the original tree, it has no need to output to a stream
        super(null);
    }
    
	public void setAllTypesTo(TFunction f, TypeInfo ti) {
		for(int i = 0; i < f.sons.size(); i++) {
			TNode t = f.getChildAt(i);
			t.type = ti;
			t.accept(this);
		}
	}
	public void visitSons(TFunction f) {
		for(int i = 0; i < f.sons.size(); i++) {
			TNode t = f.getChildAt(i);
			t.accept(this);
		}
	}
	public boolean sonsAreBool(TFunction f) {
		for(int i = 0; i < f.sons.size(); i++) {
			TNode t = f.getChildAt(i);
			if(TNode._boolean.equals(t.type))
				return true;
		}
		return false;
	}
	

	public void visitTName(/*@non_null*/TName n) {
		if (n.type != null) {		
			VariableInfo vi = TNode.getVariableInfo(n.name);
			if(vi.type == null) {
//				TDisplay.info("Typing the variable " + n.name);
				vi.type = n.type;
			}
		}
		else {
			VariableInfo vi = TNode.getVariableInfo(n.name);
			if(vi != null) {
				TDisplay.warn("I had no informations to type " + n.name + 
						" and it is not in the list of known variables!");
			}
		}
	}

	public void visitTRoot(/*@non_null*/TRoot n) {
		visitSons(n);
	}

	public void visitTBoolImplies(/*@non_null*/TBoolImplies n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._boolean);
	}

	public void visitTBoolAnd(/*@non_null*/TBoolAnd n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._boolean);
	}

	public void visitTBoolOr(/*@non_null*/TBoolOr n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._boolean);
	}

	public void visitTBoolNot(/*@non_null*/TBoolNot n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._boolean);
	}

	public void visitTBoolEQ(/*@non_null*/TBoolEQ n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._boolean);
	}

	public void visitTBoolNE(/*@non_null*/TBoolNE n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._boolean);
	}

	public void visitTAllocLT(/*@non_null*/TAllocLT n) {
		visitSons(n);
	}

	public void visitTAllocLE(/*@non_null*/TAllocLE n) {
		visitSons(n);
	}

	public void visitTAnyEQ(/*@non_null*/TAnyEQ n) {
		visitSons(n);
		if(sonsAreBool(n)) {
			// in fact the node was wrong.
			TFunction parent = n.parent;
			int ind = parent.sons.indexOf(n);
			TBoolEQ beq = new TBoolEQ();
			for(int i = 0; i < n.sons.size(); i++) {
				TNode t = n.getChildAt(i);
				beq.sons.add(t);
			}
			parent.sons.setElementAt(beq, ind);
			beq.accept(this);
		}
		else {
			setAllTypesTo(n, TNode._Reference);
		}
		n.type = TNode._boolean;

		
	}

	public void visitTAnyNE(/*@non_null*/TAnyNE n) {
		visitSons(n);
		n.type = TNode._boolean;
	}

	public void visitTIntegralEQ(/*@non_null*/TIntegralEQ n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._integer);
	}

	public void visitTIntegralGE(/*@non_null*/TIntegralGE n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._integer);
	}

	public void visitTIntegralGT(/*@non_null*/TIntegralGT n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._integer);
	}

	public void visitTIntegralLE(/*@non_null*/TIntegralLE n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._integer);
		
	}

	public void visitTIntegralLT(/*@non_null*/TIntegralLT n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._integer);
		
	}

	public void visitTIntegralNE(/*@non_null*/TIntegralNE n) {
		n.type = TNode._boolean;
		setAllTypesTo(n, TNode._integer);
	}

	public void visitTIntegralAdd(/*@non_null*/TIntegralAdd n) {
		n.type = TNode._integer;
		setAllTypesTo(n, TNode._integer);
	}

	public void visitTIntegralDiv(/*@non_null*/TIntegralDiv n) {
		n.type = TNode._integer;
		setAllTypesTo(n, TNode._integer);
		
	}

	public void visitTIntegralMod(/*@non_null*/TIntegralMod n) {
		n.type = TNode._integer;
		setAllTypesTo(n, TNode._integer);
	}

	public void visitTIntegralMul(/*@non_null*/TIntegralMul n) {
		n.type = TNode._integer;
		setAllTypesTo(n, TNode._integer);	
	}
	
	public void visitTIntegralSub(/*@non_null*/TIntegralSub n) {
		n.type = TNode._integer;
		setAllTypesTo(n, TNode._integer);
	}

	public void visitTFloatEQ(/*@non_null*/TFloatEQ n) {
		n.type = TNode._boolean;
		visitSons(n);
	}

	public void visitTFloatGE(/*@non_null*/TFloatGE n) {
		n.type = TNode._boolean;
		visitSons(n);
	}

	public void visitTFloatGT(/*@non_null*/TFloatGT n) {
		n.type = TNode._boolean;
		visitSons(n);
	}

	public void visitTFloatLE(/*@non_null*/TFloatLE n) {
		n.type = TNode._boolean;
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTFloatLT(/*@non_null*/TFloatLT n) {
		n.type = TNode._boolean;
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTFloatNE(/*@non_null*/TFloatNE n) {
		n.type = TNode._boolean;
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTFloatAdd(/*@non_null*/TFloatAdd n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTFloatDiv(/*@non_null*/TFloatDiv n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTFloatMod(/*@non_null*/TFloatMod n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTFloatMul(/*@non_null*/TFloatMul n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTLockLE(/*@non_null*/TLockLE n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTLockLT(/*@non_null*/TLockLT n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTRefEQ(/*@non_null*/TRefEQ n) {
		visitSons(n);
		if(sonsAreBool(n)) {
			// in fact the node was wrong.
			TFunction parent = n.parent;
			int ind = parent.sons.indexOf(n);
			TBoolEQ beq = new TBoolEQ();
			for(int i = 0; i < n.sons.size(); i++) {
				TNode t = n.getChildAt(i);
				beq.sons.add(t);
			}
			parent.sons.setElementAt(beq, ind);
			beq.accept(this);
		}
		else {
			n.type = TNode._boolean;
		}
	}

	public void visitTRefNE(/*@non_null*/TRefNE n) {
		n.type = TNode._boolean;
		visitSons(n);
	}

	public void visitTTypeEQ(/*@non_null*/TTypeEQ n) {
		visitSons(n);
		if(sonsAreBool(n)) {
			// in fact the node was wrong.
			TFunction parent = n.parent;
			int ind = parent.sons.indexOf(n);
			TBoolEQ beq = new TBoolEQ();
			for(int i = 0; i < n.sons.size(); i++) {
				TNode t = n.getChildAt(i);
				beq.sons.add(t);
			}
			parent.sons.setElementAt(beq, ind);
			beq.accept(this);
		}
		else {
			n.type = TNode._boolean;
		}
	}

	public void visitTTypeNE(/*@non_null*/TTypeNE n) {
		n.type = TNode._boolean;
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTTypeLE(/*@non_null*/TTypeLE n) {
		n.type = TNode._boolean;
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTCast(/*@non_null*/TCast n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTIs(/*@non_null*/TIs n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTSelect(/*@non_null*/TSelect n) {
		((TNode)n.sons.get(0)).type = TNode._Reference;
		visitSons(n);
		
	}

	public void visitTStore(/*@non_null*/TStore n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTTypeOf(/*@non_null*/TTypeOf n) {
		setAllTypesTo(n, TNode._Reference);
		
	}

	public void visitTForAll(/*@non_null*/TForAll n) {
		n.type = TNode._boolean;
		visitSons(n);	
	}

	public void visitTExist(/*@non_null*/TExist n) {
		n.type = TNode._boolean;
		visitSons(n);
		
	}

	public void visitTIsAllocated(/*@non_null*/TIsAllocated n) {
		visitSons(n);
	}

	public void visitTEClosedTime(/*@non_null*/TEClosedTime n) {
		setAllTypesTo(n, TNode._Reference); // type the sons with the elem type
	}

	public void visitTFClosedTime(/*@non_null*/TFClosedTime n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTAsElems(/*@non_null*/TAsElems n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTAsField(/*@non_null*/TAsField n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTAsLockSet(/*@non_null*/TAsLockSet n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTArrayLength(/*@non_null*/TArrayLength n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTArrayFresh(/*@non_null*/TArrayFresh n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTArrayShapeOne(/*@non_null*/TArrayShapeOne n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTArrayShapeMore(/*@non_null*/TArrayShapeMore n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTIsNewArray(/*@non_null*/TIsNewArray n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTUnset(/*@non_null*/TUnset n) {
		visitSons(n);
		// TODO Auto-generated method stub
		
	}

	public void visitTMethodCall(/*@non_null*/TMethodCall n) {
		visitSons(n);
	}
	
	public void visitTString(/*@non_null*/TString n) {
		n.type = TNode._String;
	}

	public void visitTBoolean(/*@non_null*/TBoolean n) {
		n.type = TNode._boolean;
		
	}

	public void visitTChar(/*@non_null*/TChar n) {
		n.type = TNode._char;
	}

	public void visitTInt(/*@non_null*/TInt n) {
		n.type = TNode._integer;
	}

	public void visitTFloat(/*@non_null*/TFloat n) {
		n.type = TNode._float;
		
	}

	public void visitTDouble(/*@non_null*/TDouble n) {
		n.type = TNode._double;
		
	}

	public void visitTNull(/*@non_null*/TNull n) {
		n.type = TNode._Reference;
		
	}

	public void visitTSum(/*@non_null@*/ TSum s) {
	
	}


	

}
