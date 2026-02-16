// Test various configurations of constructors, to be sure the Modifies
// assertion is correctly translated and placed.
//#FLAGS: -classpath . -quiet

public class CONX {

	static int i,ii;
	int j;
	Object o;

	//@ modifies \nothing;
	void m() {
		Object o = new CONX();
	}

	//@ modifies \nothing;
	static void mm() {
		Object o = new CONX(0); // WARNING
	}

	//@ modifies \nothing;
	void mmm() {
		Object o = new CONX(0.0);
	}

	//@ modifies \nothing;
	void m4() {
		Object oo = new CONX(o); // WARNING
	}

	//@ modifies CONX.*;
	void m5() {
		Object oo = new CONX(o); 
	}


	//@ modifies this.*;
	public CONX();

	//@ modifies i;
	//@ ensures i == \old(i) + 1;
	public CONX(int k);

	//@ modifies j;
	//@ ensures j == 0;
	public CONX(double d);

	//@ modifies CONX.*;
	public CONX(Object o);



	//@ requires i >= 0;
	//@ modifies i;
	public void m10() {
		i = 2;
		new CONX(0,0); // OK
	}

	//@ requires i >= 0;
	//@ modifies i;
	public void m11() {
		i = 3;
		new CONX(0,0);  // OK
	}

	//@ requires i >= 0;
	//@ modifies i;
	//@ ensures i == 4;
	//@ also
	//@ requires i >= 3;
	//@ modifies j ;
	//@ modifies i;
	//@ ensures i == 4;
	public CONX(int kk, int kkk);


	//@ requires i >= 0;
	//@ modifies i;
	public void m20() {
		i = 2;
		new CONX(0,0.0); // OK
	}

	//@ requires i >= 0;
	//@ modifies i;
	public void m21() {
		i = 3;
		new CONX(0,0.0);  // WARNING
	}

	//@ requires i >= 0;
	//@ modifies i;
	//@ ensures i == 4;
	//@ also
	//@ requires i >= 3;
	//@ modifies ii ;
	//@ modifies i;
	//@ ensures i == 4;
	public CONX(int kk, double kkk);


	//@ requires i == 2;
	//@ modifies i;
	public CONX(int kk, int kkk, int kkkk) {
	    this(0,0); // OK
        }

	//@ requires i == 3;
	//@ modifies i;
	public CONX(int kk, int kkk, double kkkk) {
	    this(0,0); // WARNING
        }
}

