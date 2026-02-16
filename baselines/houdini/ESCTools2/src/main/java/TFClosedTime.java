package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
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
