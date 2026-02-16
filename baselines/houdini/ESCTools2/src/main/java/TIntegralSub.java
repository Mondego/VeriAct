package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIntegralSub extends TIntFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralSub(this);
    }

}
//real comparisons : TFloatOp : float * float -> boolean
