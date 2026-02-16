package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TCast extends TBoolRes{

    public void typeTree(){}
    
    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTCast(this);
    }

}

