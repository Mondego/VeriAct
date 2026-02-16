package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIntegralLT extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralLT(this);
    }

}

