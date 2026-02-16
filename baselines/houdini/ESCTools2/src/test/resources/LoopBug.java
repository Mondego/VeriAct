// Atest that encountered a bug that caused a stack overflow.

public class LoopBug {

	//@ model public boolean initialized;

	//@ requires initialized;
	public void m() {
		int i = 1;
	}
}
