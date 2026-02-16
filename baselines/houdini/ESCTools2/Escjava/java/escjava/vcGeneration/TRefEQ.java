package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TRefEQ extends TRefOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTRefEQ(this);
    }

}

