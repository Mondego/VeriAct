package escjava.vcGeneration;

// int * int -> int
abstract class TIntFun extends TFunction {

    public TIntFun(){
	type = _integer;
    }

    protected void typeTree(){

	/*
	 * Semantic control : each son should have type 'integer'
	 */
	
	for(int i = 0; i <= sons.size() - 1; i++){
	    TNode nodeTemp = getChildAt(i);

	    if(nodeTemp.type  != null){
		if(!nodeTemp.type.equals(_integer)) {
		    System.err.println("*** Typecheck error in the tree of ifpvc");

		    /*
		     * Print all sons
		     */
		    System.err.println("Node : "+this.toString());
		    System.err.println("should have all sons with type integer");
		    System.err.println("List of sons :");

		    for(int j = 0; j <= sons.size() - j; i++)
			System.err.println("Node : "+getChildAt(j).toString());
		}
	    }
	    else // type has not been set, setting it
		nodeTemp.setType(_integer, true);

	    nodeTemp.typeTree();
	}

    }
}

