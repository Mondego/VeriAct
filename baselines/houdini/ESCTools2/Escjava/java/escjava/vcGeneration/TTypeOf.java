package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TTypeOf extends TFunction {
    
    public TTypeOf(){
	type = _Type;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTTypeOf(this);
    }

}

//quantifier
// bool -> bool // fixme
