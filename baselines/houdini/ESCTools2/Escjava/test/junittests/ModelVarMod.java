// This test tests various interactions of model variables and modifies
// clauses.

public class ModelVarMod extends ModelVarModS { 

	//@ public model int m;
	//@ represents ms = j+2;
	public int j; //@ in ms;
	public int kkk;

	//@ non_null
	ModelVarMod a = new ModelVarMod();

// m is a model var for which there is no visible representation
// ms is a model var for which there is a representation, though not in the
//    same class as ms is declared

// In this case nothing is modified, so Esc/Java should realize that neither
// model var will change, despite not knowing the representation of m.
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires a.m == 0;
	//@ requires a.ms == 0;
	//@ modifies \nothing;
	//@ ensures m == 6;
	//@ ensures ms == 4;
	//@ ensures a.m == 0;
	//@ ensures a.ms == 0;
	public void p() {
		int ii = 0;
	    //@ assert m == 6;
	    //@ assert ms == 4;
	}

// In this case something is modified, but Esc/Java can still determine that
// ms has the specified value, because its representation is known.  However,
// m has an unknown representation, so we must rely on the fact that it is
// not listed as modified in order to know it does not change.
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires a.m == 0;
	//@ requires a.ms == 0;
	//@ modifies kkk;
	//@ ensures m == 6;
	//@ ensures ms == 4;
	//@ ensures a.m == 0;
	//@ ensures a.ms == 0;
	public void p2() {
	    kkk = 0;
	    //@ assert ms == 4;
	    //@ assert m == 6;
	}

// Here m and ms are modifiable.  ms can be calculated from its representation,
// so we can determine its value in the assert and ensures; m, however, has
// no representation, so we do not know its new values.
	//@ requires j == 1;
	//@ requires m == 6;
	//@ requires a.m == 0;
	//@ requires a.ms == 0;
	//@ modifies kkk,m,ms;
	//@ ensures ms == 4;
	//@ ensures a.ms == 0;
	public void p3() {
	    kkk = 0;
	    j = 2;
	    //@ assert ms == 4;
	    //@ assert m == 6; // ERROR - OK for now - presume m does not depend on variables
	}

// Same as above, except the modifies clause is a little different
// Here m and ms are modifiable.  ms can be calculated from its representation,
// so we can determine its value in the assert and ensures; m, however, has
// no representation, so we do not know its new values.
	//@ requires j == 1;
	//@ requires m == 6;
	//@ requires a.m == 0;
	//@ requires a.ms == 0;
	//@ modifies kkk,m,ms;
	//@ ensures ms == 4;
	//@ ensures a.m == 0; // ERROR - how do we know a != this - OK for now
	//@ ensures a.ms == 0;
	public void p3a() {
	    kkk = 0;
	    j = 2;
	    //@ assert ms == 4;
	}

// Same as above, except the modifies clause is a little different
	//@ requires j == 2;
	//@ requires m == 6;
	//@ modifies this.*;
	public void pp() {
	    int ii = 0;
	    //@ assert m == 6;
	    j = 10;
	    //@ assert m == 6;  // ERROR - OK for now
	}

// Same as p3, except the modifies clause is a little different
	//@ requires j == 2;
	//@ requires m == 6;
	//@ modifies this.*;
	//@ ensures ms == 4;  
	public void pp2() {
	    int ii = 0;
	    //@ assert ms == 4;
	    j = 10;
	    //@ assert ms == 4;  // ERROR
	}

	
}

class ModelVarModPS {

	//@ static public model int m;
	//@ static public model int ms;
	//@ static represents ms = jj+1;
	static public int jj;
}

class ModelVarModS {

	//@ public model int ms;

}

class ModelVarModD extends ModelVarMod {

	//@ represents m = i+1;
	public int i;

}

class ModelVarModStatic {

	//@ static public model int m;
	//@ static public model int ms;
	//@ static represents ms = j+2;
	static public int j; //@ in ms;
	static public int kkk;

// m is a model var for which there is no visible representation
// ms is a model var for which there is a representation

// In this case nothing is modified, so Esc/Java should realize that neither
// model var will change, despite not knowing the representation of m.
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies \nothing;
	//@ ensures m == 6;
	//@ ensures ms == 4;
	//@ ensures ModelVarModPS.m == 0;
	//@ ensures ModelVarModPS.ms == 0;
	public void q() {
		int ii = 0;
	    //@ assert m == 6;
	    //@ assert ms == 4;
	}

// In this case something is modified, but Esc/Java can still determine that
// ms has the specified value, because its representation is known.  However,
// m has an unknown representation, so we must rely on the fact that it is
// not listed as modified in order to know it does not change.
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies kkk;
	//@ ensures m == 6;
	//@ ensures ms == 4;
	//@ ensures ModelVarModPS.m == 0;
	//@ ensures ModelVarModPS.ms == 0;
	public void q2() {
	    kkk = 0;
	    //@ assert ms == 4;
	    //@ assert m == 6;
	}

// Here m and ms are modifiable.  ms can be calculated from its representation,
// so we can determine its value in the assert and ensures; m, however, has
// no representation, so we do not know its new values.
	//@ requires j == 1;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies kkk,m,ms;
	//@ ensures ms == 4;
	//@ ensures ModelVarModPS.m == 0; 
	//@ ensures ModelVarModPS.ms == 0;
	public void q3() {
	    kkk = 0;
	    j = 2;
	    //@ assert ms == 4;
	    //@ assert m == 6; // ERROR - OK for now
	}

// Same as above, except the modifies clause is a little different
	//@ requires j == 1;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies kkk,m,ms;
	//@ ensures ms == 4;
	//@ ensures ModelVarModPS.m == 0; 
	//@ ensures ModelVarModPS.ms == 0;
	public void q3a() {
	    kkk = 0;
	    j = 2;
	    //@ assert ms == 4;
	}


// Same as above, except the modifies clause is a little different
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies ModelVarModStatic.*;
	//@ ensures ModelVarModPS.m == 0; 
	//@ ensures ModelVarModPS.ms == 0;
	public void qq() {
	    int ii = 0;
	    //@ assert m == 6;
	    j = 10;
	    //@ assert m == 6;  // ERROR - OK for now
	}

// Same as p3, except the modifies clause is a little different
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies ModelVarModStatic.*;
	//@ ensures ms == 4;  
	//@ ensures ModelVarModPS.m == 0; 
	//@ ensures ModelVarModPS.ms == 0;
	public void qq2() {
	    int ii = 0;
	    //@ assert ms == 4;
	    j = 10;
	    //@ assert ms == 4;  // ERROR
	}

// Same as q3, except the modifies clause is a little different
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies ModelVarModStatic.*;
	//@ ensures ModelVarModPS.m == 0;
	//@ ensures ModelVarModPS.ms == 0;
	public void qq2a() {
	    int ii = 0;
	    //@ assert ms == 4;
	    j = 10;
	}

// Same as q3, except the modifies clause is a little different
	//@ requires j == 2;
	//@ requires m == 6;
	//@ requires ModelVarModPS.m == 0;
	//@ requires ModelVarModPS.ms == 0;
	//@ modifies ModelVarModStatic.*;
	//@ ensures ModelVarModPS.ms == 0;
	public void qq2b() {
	    int ii = 0;
	    //@ assert ms == 4;
	    j = 10;
	}
	
}
