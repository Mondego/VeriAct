package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TFloatEQ extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatEQ(this);
    }

}

