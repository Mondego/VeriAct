// Tests parsing various monitored modifiers

public class Monitored {

	//@ spec_public non_null
	Object o = new Object();

	//@ non_null
	static Object os = new Object();

	//@ monitored
	public int i; // Needs this, o - via monitored, monitors_for

	//@ monitored
	//@ monitored_by o;
	public int ii; // Needs this, o - via monitored, monitored_by

	//-@ monitored
	static public int j; //Needs class - via monitored

	//-@ monitored_by Monitored.class;
	static public int jj; //Needs class - via monitored_by

	static public int jjj; //Needs class - via monitors_for

	//-@ monitored_by o;
	public int k;  // Needs o - via monitored_by, monitors_for

	public int z;  // Needs o - via monitors_for

	//@ monitors_for k = o;
	//@ monitors_for i = this,o;
	//@ monitors_for jjj = Monitored.class;
	//@ monitors_for z <- o;
	//@ modifies i;
	void m() {
		i = 0;  // ERROR - needs this, o
	}
	//@ modifies i;
	synchronized void mm() {
		i=0;    // ERROR - needs o
	}
	//@ modifies i;
	void m4() {
		synchronized(o) { i=0; }  // ERROR - needs this
	}
	//@ modifies i;
	synchronized void mmm() {
		synchronized(o) { i=0; } // OK
	}
	//@ modifies ii;
	synchronized void mmii() {
		ii=0; // ERROR - needs o
	}
	//@ modifies ii;
	synchronized void mmiiok() {
		synchronized(o) { ii=0; }  // OK
	}

	//@ modifies k;
	void m5() {
		k = 0; // ERROR -- needs o
	}
	//@ modifies z;
	void mz() {
		z = 0; // ERROR -- needs o
	}

	static void mj() { j = 0; } //@ nowarn Modifies;  // ERROR - needs class
	static void mjj() { jj = 0; } //@ nowarn Modifies;  // ERROR - needs class
	static void mjjj() { jjj = 0; } //@ nowarn Modifies;  // ERROR - needs class
	static synchronized void smj() { j = 0; } //@ nowarn Modifies; // OK
	static synchronized void smjj() { jj = 0; } //@ nowarn Modifies; // OK
	static synchronized void smjjj() { jjj = 0; } //@ nowarn Modifies; // OK

	// read accesses

	void r() {
		int p = i; // ERROR
	}

	synchronized void r2() { int p = i; }

	void r3() {
		synchronized(o) { int p = i; }
	}

	synchronized void r4() {
		synchronized(o) { int p = i; }
	}

	static void rj() { int p = j; } // ERROR
	static void rjj() { int p = jj; } // ERROR
	static void rjjj() { int p = jjj; } // ERROR
	static synchronized void srj() { int p = j; }
	static synchronized void srjj() { int p = jj; }
	static synchronized void srjjj() { int p = jjj; }



}
