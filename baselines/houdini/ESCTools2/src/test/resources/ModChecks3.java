// This checks some unsupported uses of wildcards and ranges.

public class ModChecks3{

        //@ modifies aa[*][*];   // ERROR
        //@ modifies aa[*][0];   // ERROR
        //@ modifies aa[2..4][0];   // ERROR
        //@ modifies this.*.*;   // ERROR
	void zz() {}
}

class ModChecks3A {

	static public int i;
	public int j; //@ in i; // ERROR

	ModChecks3A o; //@ maps o.j \into i; // ERROR

	static ModChecks3A oo; //@ maps oo.j \into i; // OK
}

