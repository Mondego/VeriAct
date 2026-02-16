package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TAnyNE extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAnyNE(this);
    }

}

// integral comparisons, TIntOp : int * int -> boolean
