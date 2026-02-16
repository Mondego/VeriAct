package escjava.vcGeneration;

// TBoolOp = return a boolean and sons are boolean : list(boolean) -> boolean
public class TEClosedTime extends TFunction {

    protected TEClosedTime(){
	type = _Time;
    }

    public void accept(/*@ non_null @*/ TVisitor v){
	v.visitTEClosedTime(this);
    }

} 

// %ReferenceField -> %Time
