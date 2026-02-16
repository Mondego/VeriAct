


public class Readable {
	boolean b,bb;
	int i;
	int j;

	//@ readable i if b;
	//@ writable j if bb;

	public void m() {
	    b = true;
	    int k = i;
	}

	public void m2() {
	    b = false;
	    int k = i;
	}

	public void m3() {
	    bb = true;
	    j = 0;
	}

	public void m4() {
	    bb = false;
	    j = 0;
	}
}
