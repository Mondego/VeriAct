//#FLAGS: -classpath ..
package junittests;
//@ refine "RefineA.java-refined";

public class RefineA {

	public int j;

	public int m(int i) {
		return i;
	}

	//@ ensures b;
	public int n() {
		return 0;
	}

}
