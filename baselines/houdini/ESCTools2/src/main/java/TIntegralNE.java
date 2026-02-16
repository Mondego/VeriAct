package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIntegralNE extends TIntOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralNE(this);
    }

}

// integral operation : TIntFun : int * int -> int
