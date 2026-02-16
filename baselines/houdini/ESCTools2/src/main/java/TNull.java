package escjava.vcGeneration;

public class TNull extends TLiteral {
    
    protected Object value = null;
    
    protected TNull(){
	type = _Reference;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTNull(this);
    }

}

