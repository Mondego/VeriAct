// Tests that things not changed by a called method are known to be
// unchanged.  Particularly other parts of an array and fields with 
// the same name but of other objects.

public class Unchanged extends Sup {

	public int f;
	static public int s;

	//@ non_null
	public Unchanged a = new Unchanged();
	//@ non_null
	public Unchanged aa = new Unchanged();

	//@ modifies f,aa.f;
	public void m();

	//@ requires a != this && aa != a;
	public void mm() {
		m();
		//@ assert \not_modified(a.f);
	}

	//@ requires a != this && aa != a;
	public void mm1() {
		m();
		//@ assert \not_modified(this.f); // FAILS
	}

	//@ requires a != this && aa != a;
	public void mm2() {
		m();
		//@ assert \not_modified(aa.f); // FAILS
	}

	//@ modifies super.sp;
	public void msup();

	//@ requires a != this && aa != a;
	public void mmsup() {
		msup();
		//@ assert \not_modified(a.sp); 
	}

	//@ requires a != this && aa != a;
	public void mmsup1() {
		msup();
		//@ assert \not_modified(super.sp); // FAILS
	}

	//@ modifies this.s, aa.s;
	public void ms();

	//@ requires a != this && aa != a;
	public void mms1() {
		ms();
		//@ assert \not_modified(a.s); // FAILS
	}

	//@ modifies Unchanged.s;
	public void mss();

	//@ requires a != this && aa != a;
	public void mmss1() {
		mss();
		//@ assert \not_modified(a.s); // FAILS
	}

        //@ public model JMLDataGroup contents;

	int[] ar = new int[20];
		//@ maps ar[1] \into contents;
		//@ maps ar[2] \into contents;

	//@ requires ar != null && ar.length >= 10;
	//@ modifies ar[1], ar[2];
	public void ma();


	//@ requires ar != null && ar.length >= 10;
	public void mma() {
		ma();
		//@ assert \not_modified(ar[0]);
	}

	//@ requires ar != null && ar.length >= 10;
	public void mma1() {
		ma();
		//@ assert \not_modified(ar[1]); // FAILS
	}

	//@ requires ar != null && ar.length >= 10;
	//@ modifies ar[1..2], ar[4..ar.length-1];
	public void mra();

	//@ requires ar != null && ar.length >= 10;
	//@ modifies ar[*];
	public void mrall();


	//@ requires ar != null && ar.length >= 10;
	public void mmra() {
		mra();
		//@ assert \not_modified(ar[0]);
	}

	//@ requires ar != null && ar.length >= 10;
	public void mmra1() {
		mra();
		//@ assert \not_modified(ar[1]); // FAILS
	}

	//@ requires ar != null && ar.length >= 10;
	public void mmrall1() {
		mrall();
		//@ assert \not_modified(ar[1]); // FAILS
	}

	//@ requires ar != null && ar.length >= 10;
	public void mmrall2() {
		mrall();
		//@ assert ar[1] == ar[2]; // FAILS
	}

	//@ requires ar != null && ar.length >= 10;
	//@ requires aa.ar != null && aa.ar.length >= 10;
	//@ modifies contents, aa.contents;
	public void mga();


	//@ requires ar != null && ar.length >= 10;
	//@ requires aa.ar != null && aa.ar.length >= 10;
	public void mmga() {
		mga();
		//@ assert \not_modified(ar[0]);
	}

	//@ requires ar != null && ar.length >= 10;
	//@ requires aa.ar != null && aa.ar.length >= 10;
	public void mmga1() {
		mga();
		//@ assert \not_modified(ar[1]); // FAILS
	}


	//@ modifies this.*;
	public void mw();

	public void mmw1() {
		mw();
		//@ assert \not_modified(f); // FAILS
	}

	//@ requires a != this;
	public void mmw() {
		mw();
		//@ assert \not_modified(\old(a).f); // OK
	}


	//@ modifies Unchanged.*;
	public void msw();

	public void mmsw1() {
		msw();
		//@ assert \not_modified(s); // FAILS
	}

	//@ requires a != this;
	public void mmsw2() {
		msw();
		//@ assert \not_modified(\old(a).s); // FAILS
	}

	//@ requires a != this;
	public void mmsw() {
		msw();
		//@ assert \not_modified(f); // OK
	}

	public int i;

	//@ requires ar != null && ar.length >= 10;
	//@ modifies i,ar[i];
	public void zp();

	//@ requires ar != null && ar.length >= 10;
	public void zmp() {
		i=1;
		ar[1] = 1;
		ar[2] = 2;
		zp();
		i = 2;
		//@ assert ar[2] == 2;
	}

	//@ requires ar != null && ar.length >= 10;
	public void zmp1() {
		i=1;
		ar[1] = 1;
		ar[2] = 2;
		zp();
		i = 2;
		//@ assert ar[1] == 1; // FAILS
	}

	//@ requires ar != null && ar.length >= 10;
	//@ modifies i,ar[i..i+1];
	public void zr();

	//@ requires ar != null && ar.length >= 10;
	public void zrp() {
		i=1;
		ar[0] = 0;
		ar[1] = 1;
		ar[2] = 2;
		zr();
		i = 2;
		//@ assert ar[0] == 0;
	}

	//@ requires ar != null && ar.length >= 10;
	public void zrp1() {
		i=1;
		ar[0] = 0;
		ar[1] = 1;
		ar[2] = 2;
		zr();
		i = 2;
		//@ assert ar[1] == 1; // FAILS
	}

        
	//@ modifies a,a.f;
	public void zf();

	//@ requires a != aa;
	public void zfp() {
		zf();
		a = aa;
		//@ assert a.f == \old(aa.f); // OK current a == aa, \old(a) is modified
	}
	//@ requires a != aa;
	public void zfpp() {
		zf();
		a = aa;
		//@ assert \not_modified(aa.f); // OK
	}

	//@ requires a != aa;
	public void zfp1() {
		zf();
		a = aa;
		//@ assert \not_modified(\old(a).f); // FAILS
	}

	//@ modifies ar, ar[*];
	public void zra();

	public void zrap1() {
		zra();
		//@ assert \not_modified(\old(ar)[0]); // FAILS
	}

	public void zrap() {
		int[] arr = new int[20];
		zra();
		ar = arr;
		//@ assert ar[0] == \old(arr[0]); // OK
	}


	//@ modifies a,a.*;
	public void zw();

	//@ requires a != aa && a != this;
	public void zwp() {
		zw();
		a = aa;
		//@ assert a.f == \old(aa.f); // OK
	}
	//@ requires a != aa && a != this;
	public void zwp1() {
		zw();
		a = aa;
		//@ assert \not_modified(\old(a).f); // FAILS
	}


	private Unchanged();
}

class Sup {

    public int sp;
}
