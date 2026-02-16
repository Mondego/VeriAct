package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
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
