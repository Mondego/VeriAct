package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TExist extends TBoolRes {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTExist(this);
    }

} 

// allocation
