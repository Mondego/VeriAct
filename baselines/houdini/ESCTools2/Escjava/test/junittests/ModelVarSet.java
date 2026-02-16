//#FLAGS: -classpath . -quiet

public class ModelVarSet {
	public ModelVarSet();

	static private int slen;
	//@ public static model int ssize;
	//@ static represents ssize = slen + 53;

	private int len;
	//@ public model int size;
	//@ represents size = len + 33;

	//@ ghost int i;
	//@ ghost static int si;

	//@ requires len == 1;
	//@ modifies len, size, i;
	public void m() {
		//@ set i = size;
		len = 2;
		//@ assert i == 34;
		//@ ghost int j = size;
		//@ assert j == i + 1;
	}

	//@ requires slen == 1;
	//@ modifies slen, ssize, si;
	static public void sm() {
		//@ set si = ssize;
		slen = 2;
		//@ assert si == 54;
		//@ ghost int j = ssize;
		//@ assert j == si + 1;
	}
}
