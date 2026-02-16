package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIntegralGE extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralGE(this);
    }

}

