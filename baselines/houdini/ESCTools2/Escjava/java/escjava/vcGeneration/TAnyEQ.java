package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
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

