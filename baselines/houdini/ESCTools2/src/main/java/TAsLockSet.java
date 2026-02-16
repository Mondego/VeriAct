package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TAsLockSet extends TFunction {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTAsLockSet(this);
    }

}

//array 
// %ArrayReference -> integer
