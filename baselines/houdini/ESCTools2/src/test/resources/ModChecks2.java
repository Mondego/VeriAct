// Tests that assignment statements are checked against modifies clauses.
// This checks items that do not need the static checker
//#FLAGS: -quiet -classpath .

public class ModChecks2 extends S{
	ModChecks2();

	int i,k;
	static int j;
	/*@ non_null */ int a[]; //@ invariant a.length > 10;

	//@ non_null
	static ModChecks2 o = new ModChecks2();

	//@ ghost int gi;

	//@ modifies \nothing;
	void meA() {
		i = 0;			// ERROR
	}
	//@ modifies \nothing;
	void meB() {
		j++;			// ERROR
	}
	//@ modifies \nothing;
	void meC() {
		a[0] = 7;		// ERROR
	}
	//@ modifies \nothing;
	void meD() {
		a = new int[15];	// ERROR
	}
	//@ modifies \nothing;
	void meE() {
		//@ set gi = 0;		// ERROR
	}
	//@ modifies \nothing;
	void meF() {
		o.j += 3;		// ERROR
	}

        //@ modifies ModChecks2.*;
        void mstatic3() {
                i = 0; // ERROR
        }


        //@ modifies this.*;
	void mstar2() {
                o.j = 0; // ERROR
        }

        //@ modifies this.*;
        void mstar3() {
                j = 0;   // ERROR
        }

        //@ modifies super.i;
        void msuper2() {
                i = 0; // ERROR
        }

        //@ modifies i;
        void msuper3() {
                super.i = 0; // ERROR
        }

}

class S {
	public int i;
}
