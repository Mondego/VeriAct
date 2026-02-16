public class ModChecksMethod extends ModChecksMethodZ {

	//@ non_null
	static public ModChecksMethod o = new ModChecksMethod();

	public int i;
	public static int si;

	//@ non_null
	public int[] a = new int[10];

	//@ non_null
	public int[] b = new int[10];

	//@ invariant a.length >= 10;
	//@ invariant b.length >= 10 && a != b;

	//@ modifies \nothing;
	public void mnothing() {
	    pe(); // WARNING
	}
	//@ modifies \nothing;
	public void mnothingA() {
	    pi(); // WARNING
	}
	//@ modifies \nothing;
	public void mnothingB() {
	    pn();
	}
	//@ modifies \nothing;
	public void mnothingC() {
	    paelem(); // WARNING
	}
	//@ modifies \nothing;
	public void mnothingD() {
	    pastar(); // WARNING
	}
	//@ modifies \nothing;
	public void mnothingE() {
	    parange(); // WARNING
	}
	//@ modifies \nothing;
	public void mnothingF() {
	    ptstar(); // WARNING
	}
	//@ modifies \nothing;
	public void mnothingG() {
	    pTstar(); // WARNING
	}
	//@ modifies \nothing;
	public void mnothingH() {
	    pATstar(); // WARNING
	}


	//@ modifies \everything;
	public void meverything() {
	    pe();
	    pi();
	    pn();
	    paelem(); 
	    pastar();
	    parange();
	    ptstar();
	    pTstar();
	    pATstar(); 
	}

	//@ modifies i;
	public void mi() {
	    pn();
	    pi();
	}
	//@ modifies i;
	public void mi1() {
	    o.pi(); // WARNING
	}
	//@ modifies i;
	public void mi2() {
	    pe(); // WARNING
	}
	//@ modifies i;
	public void mi3() {
	    o.pe(); // WARNING
	}
	//@ modifies i;
	public void mi3A() {
	    paelem(); // WARNING
	}
	//@ modifies i;
	public void mi3B() {
	    pastar(); // WARNING
	}
	//@ modifies i;
	public void mi3C() {
	    parange(); // WARNING
	}
	//@ modifies i;
	public void mi3D() {
	    ptstar(); // WARNING
	}
	//@ modifies i;
	public void mi3E() {
	    pTstar(); // WARNING
	}
	//@ modifies i;
	public void mi3F() {
	    pATstar(); // WARNING
	}

	//@ modifies i,o.i;
	public void mii() {
	    pi();
	    o.pi();
	}

	//@ modifies this.*;
	public void mtstar() {
	    ptstar();
	    o.ptstar(); // WARNING
	}
	//@ modifies this.*;
	public void mtstarA() {
	    pTstar(); // WARNING
	}
	//@ modifies this.*;
	public void mtstarB() {
	    o.pTstar(); // WARNING
	}
	//@ modifies this.*;
	public void mtstarC() {
	    pATstar(); // WARNING
        }

	//@ modifies ModChecksMethod.*;
	public void mTstar() {
	    ptstar(); // WARNING
        }

	//@ modifies ModChecksMethod.*;
	public void mTstarA() {
	    o.ptstar(); // WARNING
        }

	//@ modifies ModChecksMethod.*;
	public void mTstarB() {
	    pTstar(); 
	    o.pTstar(); 
        }

	//@ modifies ModChecksMethod.*;
	public void mTstarC() {
	    pATstar(); // WARNING
        }

	//@ modifies ModChecksMethodA.*;
	public void mATstar() {
	    ptstar(); // WARNING
        }

	//@ modifies ModChecksMethodA.*;
	public void mATstarA() {
	    o.ptstar(); // WARNING
        }

	//@ modifies ModChecksMethodA.*;
	public void mATstarB() {
	    pTstar(); // WARNING
        }

	//@ modifies ModChecksMethodA.*;
	public void mATstarC() {
	    o.pTstar(); // WARNING
        }

	//@ modifies ModChecksMethodA.*;
	public void mATstarD() {
	    pATstar(); 
        }

	//@ modifies a[*];
	public void mastar() {
	    pastar();
	    parange();
	    paelem();
        }

	//@ modifies a[*];
	public void mastarA() {
	    pbstar(); // WARNING
	}
	//@ modifies a[*];
	public void mastarC() {
	    pbrange(); // WARNING
	}
	//@ modifies a[*];
	public void mastarB() {
	    pbelem(); // WARNING
        }

	//@ modifies a[3..5];
	public void marange() {
	    pastar(); // WARNING
        }

	//@ modifies a[3..5];
	public void marangeZ() {
	    paelem(); // WARNING
	}
	
	//@ modifies a[3..5];
	public void marangeA() {
	    parange(); // WARNING
	}

	//@ modifies a[2..4];
	public void marange1() {
	    pastar(); // WARNING
	}

	//@ modifies a[2..4];
	public void marange1A() {
	    paelem(); 
	}

	//@ modifies a[2..4];
	public void marange1B() {
	    parange(); // WARNING
	}

	//@ modifies a[1..6];
	public void marange2() {
	    pastar(); // WARNING
	}

	//@ modifies a[1..6];
	public void marange2A() {
	    paelem(); 
	    parange(); 
	}

	//@ modifies i;
	public void pi();

	//@ modifies \everything;
	public void pe();

	//@ modifies \nothing;
	public void pn();

	//@ modifies si;
	public void psi();

	//@ modifies a[*];
	public void pastar();

	//@ modifies a[2];
	public void paelem();

	//@ modifies a[2..5];
	public void parange();

	//@ modifies b[*];
	public void pbstar();

	//@ modifies b[2];
	public void pbelem();

	//@ modifies b[2..5];
	public void pbrange();

	//@ modifies this.*;
	public void ptstar();

	//@ modifies ModChecksMethodA.*;
	public void pATstar();

	//@ modifies ModChecksMethod.*;
	public void pTstar();


	//@ modifies \nothing;
	public ModChecksMethod() {
	    this(0); // WARNING
	}

	//@ modifies \nothing;
	public void mcnothing() {
	    new ModChecksMethod(0); // WARNING 
	}

	//@ modifies \nothing;
	public void mcnothingA() {
	    new ModChecksMethod(0.0); //OK
	}

	//@ modifies \everything;
	public ModChecksMethod(int iii);

	//@ modifies i;
	public ModChecksMethod(double fff);

	//@ modifies a[2];
	public void marangeB() {
	    parangeB();
	}

	//@ modifies a[2..2];
	public void parangeB();

	//@ requires a.length <= 10;
	//@ modifies a[0..10];
	public void marangeC() {
	    parangeC(); // OK - needs FIXME
	}

	//@ requires a != null;
	//@ requires a.length == 1;
	//@ modifies a[0];
	public void marangeCC() {
	    float[] f = new float[10];
	    parangeC(); // OK -- needs FIXME - is OK for a buggy reason
	}

	//@ requires a.length == 2;
	//@ modifies a[0];
	public void marangeCCC() {
	    parangeC(); // WARNING needs FIXME
	}

	//@ modifies a[*];
	public void parangeC();

	//@ modifies \nothing;
	public void mstatic() {
		new ModChecksMethod(null); // WARNING
	}

	//@ modifies ModChecksMethod.si;
	public ModChecksMethod(Object o);
	    

	//@ modifies \nothing;
	public ModChecksMethod(Object o, int i) {
		super();
	}

	//@ modifies \nothing;
	public ModChecksMethod(Object o, double i) {
		super(0); // WARNING
	}

	//@ modifies this.*;
	public ModChecksMethod(Object o, double i, int k) {
		super(0); 
	}

	//@ modifies \nothing;
	public ModChecksMethod(Object o, Object i) {
		super(0.0); // WARNING
	}

	//@ modifies ModChecksMethod.*;
	public ModChecksMethod(Object o, Object i, int k) {
		super(0.0); 
	}

	//@ modifies ModChecksMethodZ.*;
	public void pstaticA();

	//@ modifies ModChecksMethod.*;
	public void pstaticB();

	//@ modifies ModChecksMethod.*;
	public void mstaticA() {
		pstaticA();  // OK
	}

	//@ modifies ModChecksMethodZ.*;
	public void mstaticB() {
		pstaticB();  // WARNING
	}
}

class ModChecksMethodA {

	static public int msi;
	public int mi;
}

class ModChecksMethodZ {

	public int zi;
	public static int szi;

	//@ modifies \nothing;
	ModChecksMethodZ();

	//@ modifies zi;
	ModChecksMethodZ(int i);

	//@ modifies szi;
	ModChecksMethodZ(double d);
}
