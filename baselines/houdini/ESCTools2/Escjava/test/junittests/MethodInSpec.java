// FIXME - Simplify spins forever on the recursive definition of fact
public class MethodInSpec {

	Object a,b,c;

	//@ ensures \result == k+1;
	/*@ pure */ int meth(int k);

	//@ ensures \result == meth(k+1) +1;
	//@ pure
	int meth2(int k);

	/*@
		ensures \result == (n==0? 1: n*fact(n-1));
		model public int fact(int n);
	*/

	//@ ensures fact(3) == 6;
	void mf() {} // OK

	//@ ensures fact(3) == 5;
	void mff() {} // FAILS

	//@ ensures meth2(0) == 3;
	void mr() {} // OK

	//@ ensures meth(1) == 0;
	void m() {} // FAILS

	//@ ensures meth(0) == 1;
	void mm() {} // OK

	//@ ensures meth(meth(1)) == 3;
	void mmm() {} // OK

	//@ ensures meth(k) == \result;
	int m4(int k) { return k+1; } // OK


	//@ ensures (\result) == mms.mms() + MMS.mmss() - MMS.k - MMS.k;
	int m5() {
		return MMS.k + mms.i;
	} // OK

	//@ non_null
	MMS mms = new MMS();

	MethodInSpec();
}

class MMS {

	static public int k;
	public int i;

	//@ ensures \result == k+i;
	//@ pure
	int mms();

	//@ ensures \result == k+k;
	//@ pure
	static int mmss();
}
