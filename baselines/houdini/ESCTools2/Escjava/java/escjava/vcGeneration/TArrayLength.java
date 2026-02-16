package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TArrayLength extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTArrayLength(this);
    }

}

//
