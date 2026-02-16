package escjava.vcGeneration;

public class TFloat extends TLiteral {
    
    public  float value;
    
    protected TFloat(float value){
	this.value = value;
	type = _float;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTFloat(this);
    }

}

