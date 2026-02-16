package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
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

