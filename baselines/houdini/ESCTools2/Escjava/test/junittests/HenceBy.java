// Tests the hence_by statement

public class HenceBy {

	public void m() {
		int i = 0;
		int j = i + 1;
		//@ hence_by j > 0;
	}
}
