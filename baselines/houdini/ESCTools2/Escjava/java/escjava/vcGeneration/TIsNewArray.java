package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TIsNewArray extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTIsNewArray(this);
    }

}

