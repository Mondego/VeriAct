package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TRefNE extends TRefOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTRefNE(this);
    }

}

// type comparisons : %Type * %Type -> boolean
