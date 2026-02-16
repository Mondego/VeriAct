package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
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
