package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TTypeLE extends TTypeOp {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeLE(this);
    }

}

// usual functions, cast is select store typeof 

