// Tests the \elemtype construct

public class ElemType {

	public void m() {

		Object[] oa = new String[10];
		//@ assert \elemtype(\typeof(oa)) == \type(String);
		//@ assert \elemtype(\typeof(oa)) == \type(Object); // FAILS
	}

	public int i,j;
	public float f;
	public long l;
	//@ ensures \elemtype(\typeof(new int[1])) == \type(int);
	//@ ensures \elemtype(\typeof(l)) == \elemtype(\typeof(2+2)); // FAILS
	//@ ensures \elemtype(\typeof(i)) == \elemtype(\typeof(2+2));
	//@ ensures \elemtype(\typeof(i)) == \elemtype(\typeof(j));
	//@ ensures \elemtype(\typeof(i)) == \elemtype(\typeof(f)); // FAILS
	public void mm() {}
}
