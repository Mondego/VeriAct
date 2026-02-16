package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TBoolEQ extends TBoolOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolEQ(this);
    }

}

