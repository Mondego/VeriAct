package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TFloatMul extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatMul(this);
    }

}

// lock comparisons
