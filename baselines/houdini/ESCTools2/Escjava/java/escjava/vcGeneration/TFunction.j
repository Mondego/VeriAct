package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TBoolImplies extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolImplies(this);
    }
    
}

public class TBoolAnd extends TBoolOp {
    
    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolAnd(this);
    }

}

public class TBoolOr extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolOr(this);
    }

}

public class TBoolNot extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolNot(this);
    }

}

public class TBoolEQ extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolEQ(this);
    }

}

public class TBoolNE extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolNE(this);
    }

}

// allocation comparisons
// _Time * _Time -> boolean
public class TAllocLT extends TBoolRes {

    public void typeTree(){
	
	/*
	 * Semantic control
	 */

	if(sons.size() < 2)
	    TDisplay.err("Node has a different number of sons = "+sons.size()+" which is < from 2, bizarre...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);

	    // we are sure about the type of the sons
	    n1.setType(_Time,true);
	    n2.setType(_Time,true);

	    n1.typeTree();
	    n2.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAllocLT(this);
    }

}

// _Time * _Time -> boolean
public class TAllocLE extends TBoolRes {

    public void typeTree(){
	
	/*
	 * Semantic control
	 */

	if(sons.size() < 2)
	    TDisplay.err("Node has a different number of sons = "+sons.size()+" which is < from 2, bizarre...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);

	    // we are sure about the type of the sons
	    n1.setType(_Time,true);
	    n2.setType(_Time,true);

	    n1.typeTree();
	    n2.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAllocLE(this);
    }

}



// fixme
public class TAnyEQ extends TFunction {

    public void typeTree(){
	
	/*
	 * Semantic control
	 */

	if(sons.size() < 2)
	    TDisplay.err("Node has a different number of sons = "+sons.size()+" which is < from 2, bizarre...");
	else
	    // retrieve the son and compare their type
	    {
		TNode n1 = getChildAt(0);
		TNode n2 = getChildAt(1);
		TypeInfo vi1 = n1.getTypeInfo();
		TypeInfo vi2 = n2.getTypeInfo();
		

		if(vi1 == null & vi2 == null)
		    TDisplay.warn("Not able to infer type in an AnyEQ node");
		else {
		    if(vi1 == null & vi2 != null) {
			TDisplay.info("Inferring that node "+n1.toString()+ " has type "+vi2.old+" because it's a son of an AnyEQ node which other son has type "+vi2.old);
			
			n1.setType(vi2, true);
		    }
		    else if(vi1 != null & vi2 == null) {
			TDisplay.info("Inferring that node "+n2.toString()+ " has type "+vi1.old+" because it's a son of an AnyEQ node which other son has type "+vi1.old);
			n2.setType(vi1, true);
		    }
		}

		n1.typeTree();
		n2.typeTree();
	    }

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAnyEQ(this);
    }

}

public class TAnyNE extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAnyNE(this);
    }

}

// integral comparisons, TIntOp : int * int -> boolean
public class TIntegralEQ extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralEQ(this);
    }

}

public class TIntegralGE extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralGE(this);
    }

}

public class TIntegralGT extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralGT(this);
    }

}

public class TIntegralLE extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralLE(this);
    }

}

public class TIntegralLT extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralLT(this);
    }

}

public class TIntegralNE extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralNE(this);
    }

}

// integral operation : TIntFun : int * int -> int
public class TIntegralAdd extends TIntFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralAdd(this);
    }

}

public class TIntegralDiv extends TIntFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralDiv(this);
    }

}

public class TIntegralMod extends TIntFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralMod(this);
    }

}

public class TIntegralMul extends TIntFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralMul(this);
    }

}
public class TIntegralSub extends TIntFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralSub(this);
    }

}
//real comparisons : TFloatOp : float * float -> boolean
public class TFloatEQ extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatEQ(this);
    }

}

public class TFloatGE extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatGE(this);
    }

}

public class TFloatGT extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatGT(this);
    }

}

public class TFloatLE extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatLE(this);
    }

}

public class TFloatLT extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatLT(this);
    }

}

public class TFloatNE extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatNE(this);
    }

}

// float operation : TFloatFun : float * float -> float
public class TFloatAdd extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatAdd(this);
    }

}

public class TFloatDiv extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatDiv(this);
    }

}

public class TFloatMod extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatMod(this);
    }

}

public class TFloatMul extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatMul(this);
    }

}

// lock comparisons
public class TLockLE extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTLockLE(this);
    }

}

public class TLockLT extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTLockLT(this);
    }

}

// reference comparisons : %Reference * %Reference -> boolean
public class TRefEQ extends TRefOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTRefEQ(this);
    }

}

public class TRefNE extends TRefOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTRefNE(this);
    }

}

// type comparisons : %Type * %Type -> boolean
public class TTypeEQ extends TTypeOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeEQ(this);
    }

}

public class TTypeNE extends TTypeOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeNE(this);
    }

}

public class TTypeLE extends TTypeOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeLE(this);
    }

}

// usual functions, cast is select store typeof 

public class TCast extends TBoolRes{

    public void typeTree(){}
    
    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTCast(this);
    }

}

public class TIs extends TBoolRes { // %Reference | double | char etc ..., type -> boolean

    public void typeTree(){
	
	if(sons.size()!=2)
	    TDisplay.err("Node with "+sons.size()+" instead of 2, that's strange...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);

	    /*
	     * As the son #1 can be a reference of have a final type,
	     * we can't guess it here. We just know that the second son should
	     * be a type.
	     */
	    n2.setType(_Type,true);

	    n1.typeTree();
	    n2.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIs(this);
    }

} 

// %Field * %Reference -> %Reference | double | char etc... (final types)
public class TSelect extends TFunction {

    public void typeTree(){
	
	if(sons.size()!=2)
	    TDisplay.err("Node with "+sons.size()+" instead of 2, that's strange...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);

	    /*
	     * As the son #1 can be a reference of have a final type,
	     * we can't guess it here. We just know that the second son should
	     * be a %Reference.
	     */
	    n2.setType(_Reference,true);

	    n1.typeTree();
	    n2.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTSelect(this);
    }

}

// fixme
// %Field * %Reference * ? (value, %Reference?) -> memory
public class TStore extends TFunction {

    public void typeTree(){
	
	if(sons.size()!=3)
	    TDisplay.err("TStore node with "+sons.size()+" instead of 3, that's strange...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);
	    TNode n3 = getChildAt(2);

	    n1.setType(_Field, true);
	    n2.setType(_Reference,true);

	    n1.typeTree();
	    n2.typeTree();
	    n3.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTStore(this);
    }

} 

// %Reference -> %Type
public class TTypeOf extends TFunction {
    
    public TTypeOf(){
	type = _Type;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeOf(this);
    }

}

//quantifier
// bool -> bool // fixme
public class TForAll extends TBoolRes {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTForAll(this);
    }

} 

// bool -> bool // fixme
public class TExist extends TBoolRes {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTExist(this);
    }

} 

// allocation
public class TIsAllocated extends TBoolOp {

    public void typeTree(){
	
	if(sons.size()!=2)
	    TDisplay.err("Node with "+sons.size()+" instead of 2, that's strange...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);

	    // we are sure about the type of the sons
	    n1.setType(_Reference,true);
	    n2.setType(_Time,true);

	    n1.typeTree();
	    n2.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIsAllocated(this);
    }

} // %Reference -> bool

// %Reference -> %Time
public class TEClosedTime extends TFunction {

    protected TEClosedTime(){
	type = _Time;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTEClosedTime(this);
    }

} 

// %ReferenceField -> %Time
public class TFClosedTime extends TFunction {

    protected TFClosedTime(){
	type = _Time;
    }

    public void typeTree(){
	
	if(sons.size()!=1)
	    TDisplay.err("Node with "+sons.size()+" instead of 1, that's strange...");
	else {
	    TNode n1 = getChildAt(0);

	    // we are sure about the type of the sons
	    n1.setType(_Field,true);

	    n1.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFClosedTime(this);
    }

} 

// as trick : asElems asField asLockset
public class TAsElems extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAsElems(this);
    }

}

public class TAsField extends TFunction {

    public void typeTree(){
	
	if(sons.size()!=2)
	    TDisplay.err("Node with "+sons.size()+" instead of 2, that's strange...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);

	    /* we are sure about the type of the sons
	     * The types of the second son is set first, thus
	     * we can use it for the first one.
	     */
	    n2.setType(_Type,true);
	    n2.typeTree();
	    
	    // we say this is a field
	    n1.setType(_Field,true);
	    // we add his own type too
	    VariableInfo vi = n1.getVariableInfo();

	    vi.setSecondType(n2.getTypeInfo());

 	    n1.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAsField(this);
    }

}

public class TAsLockSet extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAsLockSet(this);
    }

}

//array 
// %ArrayReference -> integer
public class TArrayLength extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTArrayLength(this);
    }

}

//
public class TArrayFresh extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTArrayFresh(this);
    }

} 

public class TArrayShapeOne extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTArrayShapeOne(this);
    }

}

public class TArrayShapeMore extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTArrayShapeMore(this);
    }

}

public class TIsNewArray extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIsNewArray(this);
    }

}

public class TSum extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTSum(this);
    }

}
public class TUnset extends TFunction {

    public void typeTree(){
	
	if(sons.size()!=3)
	    TDisplay.err("TUnset node with "+sons.size()+" instead of 3, that's strange...");
	else {
	    TNode n1 = getChildAt(0);
	    TNode n2 = getChildAt(1);
	    TNode n3 = getChildAt(2);

	    n1.setType(_Field, true);
	    n2.setType(_Reference,true);

	    n1.typeTree();
	    n2.typeTree();
	    n3.typeTree();
	}

    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTUnset(this);
    }

} 
