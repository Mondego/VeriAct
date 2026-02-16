package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
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
