package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIntegralGT extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralGT(this);
    }

}

