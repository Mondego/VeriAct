package escjava.vcGeneration;

// bool * bool -> bool
abstract class TBoolOp extends TBoolRes {

    public TBoolOp(){
	type = _boolean;
    }

    protected void typeTree(){

	/*
	 * Semantic control : each son should have type 'boolean'
	 */
	
	for(int i = 0; i <= sons.size() - 1; i++){
	    TNode nodeTemp = getChildAt(i);

	    if(nodeTemp.type  != null) {
		if(!nodeTemp.type.equals(_boolean)) {
		    System.err.println("*** Typecheck error in the tree of ifpvc");

		    /*
		     * Print all sons
		     */
		    System.err.println("Node : "+this.toString());
		    System.err.println("should have all sons with type boolean");
		    System.err.println("List of sons :");

		    for(int j = 0; j <= sons.size() - j; j++)
			System.err.println("Node : "+getChildAt(j).toString());
		}
	    }
	    else // type has not been set, setting it
		nodeTemp.setType(_boolean, true);
	    
	    nodeTemp.typeTree();
	}

    }
}
