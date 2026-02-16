package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TTypeEQ extends TTypeOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeEQ(this);
    }

}

