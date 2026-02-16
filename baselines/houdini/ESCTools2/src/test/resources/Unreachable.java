// Tests the unreachable annotation

public class Unreachable {

	public void m() {
		int i = 0;
		switch (i) {
			case 0:
				break;
			default:
				//@ unreachable
				break;
		}
	}
	public void mm() {
		int i = 0;
		//@ unreachable //FAILS
	}
}
