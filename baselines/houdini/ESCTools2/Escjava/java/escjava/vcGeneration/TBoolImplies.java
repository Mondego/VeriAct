package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TBoolImplies extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolImplies(this);
    }
    
}

