// Quick test of the assume annotation

public class Assume {

	public int m(int i);

	public void mm() {
		int i = 0;
		int j = m(0);
		//@ assume j == 3;
		i = j + j;
		//@ assert i == 6;
	}
}
