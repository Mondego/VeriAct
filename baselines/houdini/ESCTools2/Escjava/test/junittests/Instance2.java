// Tests static default for ghost and model fields

public class Instance2 implements INSTI {

	public int in;
	//@ modifies g;
	static void m() {
		int i = N;
		// N = 7;
		//@ ghost int ii;
/*@
		set ii = g;
		set g = 9; // is ghost final ?
		set ii = mmm;
*/
	}
	//@ modifies gi;
	void mm() {

		//@ ghost int ii;
		//@ set ii = gi;
		//@ set gi = 10; // is ghost final?
		//@ set ii = g; 
		//@ set ii = mmm;
		//@ set ii = mmmi;
	}

}

interface INSTI {

	int N = 0;

	//@ ghost int g = 0;
	//@ instance ghost int gi = 0;

	//@ model int mmm;
	//@ model instance int mmmi;
}
