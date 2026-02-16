// Tests parsing errors in \typeof and \type

public class TypeErrors {

	public Object o;
	public int i;
	//@ ensures \typeof(o) <: \type(Object);
	//@ ensures \typeof(i) == \type(int);
	//@ ensures \typeof(i) <: \type(Object);
	//-@ ensures \typeof(i,i) <: \type(i);
	public void m() {}
}
