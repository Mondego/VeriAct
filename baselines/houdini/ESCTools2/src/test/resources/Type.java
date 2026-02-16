// Tests static checking with \typeof and \type

public class Type {
	Type();

	public Object o;
	public int i;
	public float f;
	public double d;
	public short s;
	public char c;
	public boolean b;
	public byte y;
	public long l;
	//@ ensures \typeof(null) <: \type(Object); // FAILS
	public void m1() {}

	//@ ensures \typeof(null) <: \typeof(null); 
	public void m2() {}

	//@ requires o != null;
	//@ ensures \typeof(o) <: \type(Object);
	//@ ensures \typeof(i) == \type(int);
	//@ ensures \typeof(c) == \type(char);
	//@ ensures \typeof(y) == \type(byte);
	//@ ensures \typeof(s) == \type(short);
	//@ ensures \typeof(l) == \type(long);
	//@ ensures \typeof(f) == \type(float);
	//@ ensures \typeof(d) == \type(double);
	//@ ensures \typeof(b) == \type(boolean);
	public void m() {}

	//@ ensures \typeof(b) == \type(int);  // FAILS
	public void m3() {}
	//@ ensures \typeof(i) <: \type(Object); // FAILS
	public void m4() {}
	//@ ensures \typeof(i) == \typeof(null);  // FAILS
	public void m5() {}
}
