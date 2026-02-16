package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TFloatDiv extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatDiv(this);
    }

}

