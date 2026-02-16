package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TFloatNE extends TFloatOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatNE(this);
    }

}

// float operation : TFloatFun : float * float -> float
