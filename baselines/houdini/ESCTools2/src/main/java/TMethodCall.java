package escjava.vcGeneration;

import java.util.Vector;


/**
 * The class used to represent method calls for the new Vc gen tree.
 * @version 14/11/2005
 */
public class TMethodCall extends TFunction {
	private VariableInfo name;
	private Vector argTypes = new Vector();
	
	
	/**
	 * The vector containing all the method names (in forms of String).
	 */
	public static Vector methNames = new Vector();
	/**
	 * The vector containing the definition of the methods. The
	 * definition is fetched following this pattern, for a method
	 * called strMethName (a String).
	 * <code>
	 * methDefs.get({@link Vector#indexOf(java.lang.Object) methName.indexOf(strMethName)})
	 * </code>
	 */
	public static Vector methDefs = new Vector();
	
	
	/**
	 * @see TNode#accept(TVisitor)
	 * @param args
	 */ 
	public void accept(/*@ non_null @*/ TVisitor v){
			v.visitTMethodCall(this);
	}
	
	
	/**
	 * Construct a method call.
	 * @param name The name of the method called.
	 */
	public TMethodCall(/*@ non_null @*/ String name) {
		this.name = new VariableInfo(name, new TypeInfo("pure method"));
		if(!methNames.contains(this.name)) {
			methNames.add(this.name);
			methDefs.add(this);
		}
	}
	
	
	/**
	 * return the VariableInfo associated with the method name,
	 * of interest for the return type of the method.
	 * @return The variable info object associated with the method.
	 */
	public VariableInfo getName() {
		return name;
	}
	
	/**
	 * @see TFunction#typeTree()
	 */
    protected void typeTree(){
    	// in case of double typing...
    	if(argTypes.size() > 0)
    		argTypes.clear();
    	/*
    	 * Semantic control : for now we determine the type of the pure method
    	 */
    	for(int i = 0; i < sons.size(); i++){
    	    TNode nodeTemp = getChildAt(i);
    	    nodeTemp.typeTree();
    	    // does not work at all: we only get the same empty type -- julien
    	    argTypes.add(nodeTemp.type);
    	}

    }
    /**
     * Returns a vector containing the types from the different arguments of
     * the MethodCall. The pure method has the following signature: <code>elem(1) -> elem(1) -> ... -> {@link TNode#type this.type}</code>
     * @return A vector containing elements from the TypeInfo class.
     */
    public Vector getArgType() {
    	return argTypes;
    }
    protected void setType(TypeInfo type, boolean sure){
        	this.type = type;
    }
}
