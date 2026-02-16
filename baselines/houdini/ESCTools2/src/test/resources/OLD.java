// Tests use of \old \fresh in routine body.
//#FLAGS: -quiet
public class OLD {

	int k = 0;
	//@ ghost int kk = 0;

	//@ requires k == 0 && kk == 0;
	//@ modifies k,kk;
	void m() {
		k = 1;
		//@ set kk = \old(k);
		//@ assert kk == 0;
		//@ ghost int mm = \old(k);
		//@ ghost int mmm = k;
		//@ assert mm == mmm; // ERROR
		//@ assert \old(k) == 0;
		//@ assert k == 1;
	}

	void mm() {
		Object p = new Object();
		//@ assert \fresh(p);
	}
	void mmm() {
		Object p = new Object();
		//@ assert !\fresh(p); // ERROR
	}
}
