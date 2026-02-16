package escjava.vcGeneration;

public class TString extends TLiteral{

    public String value;

    protected TString (String value){
	this.value = value;
	type = _String;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTString(this);
    }

}

