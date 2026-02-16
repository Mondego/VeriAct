package escjava.vcGeneration;

public class TDouble extends TLiteral {
    
    public  double value;
    
    protected TDouble(double value){
	this.value = value;
	type = _float;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTDouble(this);
    }

}

