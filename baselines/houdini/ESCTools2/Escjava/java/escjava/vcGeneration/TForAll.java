package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TForAll extends TBoolRes {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTForAll(this);
    }

} 

// bool -> bool // fixme
