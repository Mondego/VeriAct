// Tests the use of the uninitialized modifier
public class Uninit {

	public void m() {
		//@uninitialized
		int i = 5; 
		int j = i;
	}
}
