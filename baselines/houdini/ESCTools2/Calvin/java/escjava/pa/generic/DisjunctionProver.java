/* Copyright Hewlett-Packard, 2002 */

package escjava.pa.generic;

import java.util.*;
import javafe.util.*;
import mocha.wrappers.jbdd.*;

/* The prover interface works on bdds. This is a wrapper that is
   optimized for Disjunctions, and avoids converting to bdds where
   possible.
*/

public class DisjunctionProver {

    public static final int VALID = 0;
    public static final int INVALID = 1;
    public static final int UNKNOWN = 2;
    
    private ArrayList valid = new ArrayList();
    private ArrayList invalid = new ArrayList();

    private Prover prover;
    private jbddManager bddManager;


    public DisjunctionProver(Prover prover, jbddManager bddManager) {
	this.prover = prover;
	this.bddManager = bddManager;
    }

    public int quickCheck(Disjunction d) {
	for(Iterator it = valid.iterator(); it.hasNext(); ) {
	    // see if valid[i] => query
	    if( implies((Disjunction)it.next(), d) ) {
		return VALID;
	    }
	}

	for(Iterator it = invalid.iterator(); it.hasNext(); ) {
	    // see if query => invalid[i]
	    if( implies(d, (Disjunction)it.next()) ) {
		return INVALID;
	    }
	}

	jbdd bdd = disjToBdd( d );
	int r = prover.quickCheck( bdd );
	bdd.jbdd_free();
	
	switch( r ) {
	  case prover.VALID:
	    valid.add( new Disjunction(d) );
	    return VALID;
	    
	  case prover.INVALID:
	    invalid.add( new Disjunction(d) );
	    return INVALID;
	}
	
	return UNKNOWN;
    }



    public boolean check(Disjunction d) {
	int r = quickCheck( d );
	switch( r ) {
	  case VALID: 
	    return true;
	  case INVALID: 
	    return false;
	  case UNKNOWN: 
	      {
		  jbdd bdd = disjToBdd( d );
		  // Assert.notFalse( prover.quickCheck(bdd) == prover.UNKNOWN );
		  
		  boolean b = prover.check( bdd );
		  if( b ) {
		      valid.add( new Disjunction(d) );
		  } else {
		      invalid.add( new Disjunction(d) );
		  }
		  return b;
	      }
	  default:
	    Assert.fail("");
	    return true;
	}
    }

    public String printClause(Disjunction d) {
	return prover.printClause(disjToBdd(d));
    }

    public String report() {
	return prover.report();
    }

    public boolean implies(Disjunction d1, Disjunction d2) {
	Assert.notFalse( (d1.stars & d1.bits) == 0 );
	Assert.notFalse( (d2.stars & d2.bits) == 0 );
	Assert.notFalse( (~d2.stars & d2.bits) == d2.bits );
	boolean r = ((d2.stars & ~d1.stars) == 0) 
	           && ((d2.bits & ~d1.stars) == d1.bits);
	return r;
    }

    jbdd disjToBdd(Disjunction d) {
	// Put choice into a bdd, 
	jbdd t = bddManager.jbdd_zero();
	for( int i=bddManager.jbdd_num_vars()-1; i>=0; i-- ) {
	    long b = 1<<i;
	    if( (d.stars & b) == 0 ) {
		// no star
		jbdd varBdd = bddManager.jbdd_get_variable( i );
		jbdd t2 = jbdd.jbdd_or( t, varBdd, true, (d.bits & b) != 0 );
		// t.jbdd_free();
		t = t2;
	    }
	}
	return t;
    }


}
			   
