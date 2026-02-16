package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TTypeNE extends TTypeOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeNE(this);
    }

}

