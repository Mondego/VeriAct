package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TFloatMod extends TFloatFun {

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloatMod(this);
    }

}

