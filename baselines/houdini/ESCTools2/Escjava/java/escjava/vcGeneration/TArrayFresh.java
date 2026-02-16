package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TArrayFresh extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTArrayFresh(this);
    }

} 

