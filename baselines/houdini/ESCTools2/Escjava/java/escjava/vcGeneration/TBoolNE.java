package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TBoolNE extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolNE(this);
    }

}

// allocation comparisons
// _Time * _Time -> boolean
