public class GhostLocalError {

	//@ ghost public int i;


	public void mm() {
		//@ model int aa; // ERROR
	}
	public void mmm() {
		//@ final model int aaa; // ERROR 
	}
	public void mmmm() {
		//@ final non_null model int aaaa; // ERROR 
	}
	public void m() {

		int a = 0;
		//@ unreachable;
		//@ set i = 0;
		float k = 0;
		a++;
		//@ ghost 		// ERROR
		//@ ghost int		// ERROR
		//@ ghost X		// ERROR
		//@ ghost int z =;	// ERROR
		a++;			// FIXME - does not restart well
		//@ set i = 9;
	}
}
