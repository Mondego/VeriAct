package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIntegralAdd extends TIntFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIntegralAdd(this);
    }

}

