package escjava.vcGeneration;

/*
 * This class is for the unique root of the tree.
 */
public class TRoot extends TFunction {

    public TRoot(){
	super();
	isroot = true;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTRoot(this);
    }
    
    
    protected void typeTree() {
    	/* FIXME: The typing is done twice because the first time
    	 * some types can't be determined. A better handling 
    	 * should be found // julien
    	 */
    	super.typeTree();
    	super.typeTree();

    }
}
