package escjava.vcGeneration;

public class TBoolean extends TLiteral{

    public boolean value;

    public TBoolean (boolean value){
	this.value = value;
	type = _boolean;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTBoolean(this);
    }

}

