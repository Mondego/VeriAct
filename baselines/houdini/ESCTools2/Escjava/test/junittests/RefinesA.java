//#FLAGS: -classpath ..
package junittests;
//@ refines "RefinesA.java-refined";

public class RefinesA {

	public int j;

	public int m(int i) {
		return i;
	}

	//@ ensures b;
	public int n() {
		return 0;
	}

}
