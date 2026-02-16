package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIntegralEQ extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralEQ(this);
    }

}

