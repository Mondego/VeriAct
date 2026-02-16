package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TBoolAnd extends TBoolOp {
    
    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolAnd(this);
    }

}

