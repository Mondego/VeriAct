package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TFloatAdd extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatAdd(this);
    }

}

