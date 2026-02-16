package escjava.vcGeneration;

public class TInt extends TLiteral {
    
    public int value;
    
    protected TInt(int value){
	this.value = value;
	type = _integer;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTInt(this);
    }

}

