package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TLockLT extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTLockLT(this);
    }

}

// reference comparisons : %Reference * %Reference -> boolean
