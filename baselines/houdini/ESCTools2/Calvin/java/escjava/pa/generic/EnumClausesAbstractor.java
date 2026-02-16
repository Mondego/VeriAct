/* Copyright Hewlett-Packard, 2002 */

package escjava.pa.generic;

import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;

import javafe.util.Set;
import javafe.util.Location;
import javafe.util.Assert;
import javafe.util.StackVector;

import mocha.wrappers.jbdd.*;

/* Predicate abstraction implementation that works by enumerating
   clauses in order of increasing length [Saidi-Shankar'99].
*/

public class EnumClausesAbstractor implements Abstractor {

    private boolean noisy = Boolean.getBoolean("PANOISY");

    private jbddManager bddManager;

    private jbdd R;
    private Vector clauses = new Vector();
    private Vector enumSizes;
    private int size;

    public EnumClausesAbstractor(jbddManager bddManager) {
	say("creating");
	this.bddManager = bddManager;
	R = bddManager.jbdd_zero();
	this.enumSizes = new Vector();

	String s = System.getProperty("PA3n");

	int i;
	while( (i=s.indexOf('.')) != -1 ) {
	    String t = s.substring(0, i);
	    say(t);
	    int ti = Integer.parseInt(t);
	    ti = Math.min( ti, bddManager.jbdd_num_vars() );
	    Integer I = new Integer( ti );
	    if( !enumSizes.contains(I) ) {
		enumSizes.addElement( I );
		size = Math.max( ti, size );
	    }
	    s = s.substring(i+1);
	}
	if( s.length() != 0 ) {
	    Assert.fail("Last character in PA3n should be a dot; instead found \""+s+"\"");
	}
	say("Enum clauses of length: "+enumSizes);
    }
	
    public jbdd get() {
	return R;
    }

    public Vector getClauses() {
	return clauses;
    }

    public boolean union(Prover prover) {

	jbdd valid = bddManager.jbdd_one();
	Vector validClauses = new Vector();

	Vector allInvalidClauses = new Vector();
	Vector invalidClausesPre = new Vector();
	invalidClausesPre.add( bddManager.jbdd_zero() );

	boolean[] notAvailVar = new boolean[bddManager.jbdd_num_vars()];

	boolean bools[] = { false, true };

	int queriesConsideredTotal=0, queriesTriedTotal=0, queriesValidTotal=0;

	for(int i=1; i <= size; i++) {
	    boolean inV = enumSizes.contains( new Integer(i) );

	    int queriesConsidered=0, queriesTried=0, queriesValid=0;
	    say("Considering invariants of length "+i);

	    Vector invalidClauses = new Vector();

	    for(Enumeration e = invalidClausesPre.elements(); e.hasMoreElements(); ) {
		jbdd d = (jbdd)e.nextElement();

		int topVar = d.jbdd_is_tautology(false) ? bddManager.jbdd_num_vars() : d.jbdd_top_var_id();
		if( topVar < notAvailVar.length && notAvailVar[topVar]) {
		    // mentions a var known to be always true/always false
		    continue;
		}
		int nextVar = topVar - 1;

		for(int varNdx=nextVar; varNdx >=0; varNdx--) {
		    
		    if( notAvailVar[varNdx] ) continue;

		    jbdd varBdd = bddManager.jbdd_get_variable( varNdx );

		    for(int sign=0; sign < 2; sign++) {

			jbdd newD = jbdd.jbdd_or(d, varBdd, true, bools[sign]);

			if( noisy ) say("newD = "+prover.printClause( newD ));

			queriesConsidered++;

			// check if var already included with same sign
			if( newD.jbdd_equal( d ) ) {
			    say("ERROR: included with same sign");
			    continue;
			}

			// check if var already included with different sign
			if( newD.jbdd_is_tautology(true) ) {
			    say("ERROR: included with different sign");
			    continue;
			}

			// check if newD redundant
			if( valid.jbdd_leq( newD, true, true ) ) {
			    say("redundant");
			    continue;
			}

			// check if newD contradictory
			if( valid.jbdd_leq( newD, true, false ) ) {
			    say("contradictory");
			    continue;
			};

			// Chk if newD an extension of something in validClauses

			boolean newDMaybeValid = true;

			if( !R.jbdd_leq( newD, true, true ) ) {
			    newDMaybeValid = false;
			    say("not implied by R");
			}

			if( !inV ) {
			    invalidClauses.add( newD );
			    say("Not in v");
			    continue;
			}

			if( newDMaybeValid ) {

			    // newD is possible invariant, 
			    // see if there exists d in allInvalidClauses such that
			    // valid => (newD => d)
			    // if so, newD is not valid
			    // Optimized check is (valid /\ newD) => d

			    jbdd validAndNewD = jbdd.jbdd_and(valid, newD, true, true);

			    for(Enumeration e2 = allInvalidClauses.elements(); 
				e2.hasMoreElements() && newDMaybeValid; ) {
				jbdd d2 = (jbdd)e2.nextElement();
				if( validAndNewD.jbdd_leq( d2, true, true ) ) {
				    // newD not a tautology
				    newDMaybeValid = false;
				    say("invalid by earlier test");

				}
			    }

			    validAndNewD.jbdd_free();
			}

			if( newDMaybeValid ) {

			    // newD is possible invariant, call simplify
			    boolean newDvalid = prover. check(newD);
			    if( noisy ) say( "SIMPLIFY: "+(newDvalid ? "valid" : "invalid"));
			    queriesTried++;

			    if( newDvalid ) {
				queriesValid++;

				// say("Invariant: ");
				validClauses.add( newD );
				jbdd t = jbdd.jbdd_and( valid, newD, true, true );
				valid.jbdd_free();
				valid = t;

				if( i == 1 ) {
				    // have asserted a literal
				    // remove that variable from consideration in other
				    // disjunctions
				    notAvailVar[varNdx] = true;
				}

				continue;
			    } 
			}

			// newD not tautology or contradictory
			// maybe some extension is an invariant
			invalidClauses.add( newD );
			allInvalidClauses.add( newD );
		    } // signs
		} // var
	    } // invalidClausesPre
	    invalidClausesPre = invalidClauses;
	    System.out.println("Queries of size "+i
			       +": Considered "+ queriesConsidered
			       +" tried "+queriesTried
			       +" valid "+queriesValid);

	    queriesConsideredTotal += queriesConsidered;
	    queriesTriedTotal += queriesTried;
	    queriesValidTotal += queriesValid;
	} // i
	    System.out.println("Queries of all sizes"
			       +": Considered "+ queriesConsideredTotal
			       +" tried "+queriesTriedTotal
			       +" valid "+queriesValidTotal);
	    System.out.println("Prover: "+prover.report());
	    
	if( R.jbdd_equal( valid ) ) {
	    say("fixpt");
	    return true;
	} else {
	    R = valid;
	    clauses = validClauses;
	    
	    return false;
	}
    }
    private void say(String s) {
	if( noisy ) {
	    System.out.println(s);
	}
    }

}
