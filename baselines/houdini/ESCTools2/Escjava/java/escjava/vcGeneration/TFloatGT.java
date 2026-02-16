package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TFloatGT extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatGT(this);
    }

}

