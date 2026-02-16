package escjava.vcGeneration;

// float * float -> float
abstract class TFloatFun extends TFunction {

    public TFloatFun(){
	type = _float;
    }

    protected void typeTree(){

	/*
	 * Semantic control : each son should have type 'float'
	 */
	
	for(int i = 0; i <= sons.size() - 1; i++){
	    TNode nodeTemp = getChildAt(i);

	    if(nodeTemp.type  != null) {
		if(!nodeTemp.type.equals(_float)) {
		    System.err.println("*** Typecheck error in the tree of ifpvc");

		    /*
		     * Print all sons
		     */
		    System.err.println("Node : "+this.toString());
		    System.err.println("should have all sons with type float");
		    System.err.println("List of sons :");

		    for(int j = 0; j <= sons.size() - j; i++)
			System.err.println("Node : "+getChildAt(j).toString());
		}
	    }
	    else // type has not been set, setting it
		nodeTemp.setType(_float, true);
	    
	    nodeTemp.typeTree();
	}

    }
}

