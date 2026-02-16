//@ refine "RefineB.java-refined";

public class RefineB {

	public int j;

	public int m(int i) {
		return i;
	}

	//@ ensures b;
	public int n() {
	}

}
