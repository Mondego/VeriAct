// Tests parsing of instance

class InstanceSuperC {

	//@ instance ghost public int ff; // OK
	//@ static instance ghost public int fff; // ERROR
	//@ static ghost instance public int ffff; // ERROR
	//@ ghost static instance public int fffff; // ERROR
	//@ instance model public int mm; // OK
	//@ static instance model public int mmm; // ERROR

	//@ ghost int cg;
	//@ ghost instance int cgi;
	//@ ghost static int cgs;
	//@ model int cgm;
	//@ model instance int cgmi;
	//@ model static int cgms;
}

interface InstanceSuper {

	//@ instance ghost public int ff; // OK
	//@ static instance ghost public int fff; // ERROR
	//@ static ghost instance public int ffff; // ERROR
	//@ ghost static instance public int fffff; // ERROR
	//@ instance model public int mm; // OK
	//@ static instance model public int mmm; // ERROR

	//@ ghost int g;
	//@ ghost instance int gi;
	//@ ghost static int gs;
	//@ model int gm;
	//@ model instance int gmi;
	//@ model static int gms;
}

public class InstanceError extends InstanceSuperC implements InstanceSuper {
	//@ public instance static ghost boolean b1; // ERROR
	//@ public static instance ghost boolean b2; // ERROR
	//@ public instance ghost static boolean b3; // ERROR
	//@ public static ghost instance boolean b4; // ERROR
	//@ public ghost instance static boolean b5; // ERROR
	//@ public ghost static instance boolean b6; // ERROR
	//@ instance public ghost float f;	// OK
	//@ instance public model double d;	// OK

	void m() {
/*@
		ghost int ii;
		set ii = g;
		set ii = gi;
		set ii = gs;
		set ii = gm;
		set ii = gmi;
		set ii = gms;
		set ii = cg;
		set ii = cgi;
		set ii = cgs;
		set ii = cgm;
		set ii = cgmi;
		set ii = cgms;
*/
	}
	static void sm() {
/*@
		ghost int ii;
		set ii = g;
		set ii = gi; // ERROR
		set ii = gs;
		set ii = gm;
		set ii = gmi; // ERROR
		set ii = gms;
		set ii = cg; // ERROR
		set ii = cgi; // ERROR
		set ii = cgs;
		set ii = cgm; // ERROR
		set ii = cgmi; // ERROR
		set ii = cgms;
*/
	}

	//@ instance model interface I {}   // ERROR
	//@ instance model class C {}   // ERROR
	//@ instance model void mz() {}   // ERROR
	//@ instance model InstanceError(double d) {}   // ERROR
	//@ instance   // ERROR
	int ifield;

}

