package escjava.vcGeneration;

public class TChar extends TLiteral{

    public char value;

    protected TChar (int value){
	this.value = (char)value;
	type = _char;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTChar(this);
    }

}

