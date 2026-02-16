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

public class TNull extends TLiteral {
    
    protected Object value = null;
    
    protected TNull(){
	type = _Reference;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTNull(this);
    }

}

