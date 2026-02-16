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

/* Generates maximal clauses, when valid tries to "shrink" to a
 minimal valid subclause.  */

public class EnumMaxClausesFindMinAbstractor implements Abstractor {

    private jbddManager bddManager;

    private jbdd R;
    private Vector clauses = new Vector();
    // invariant: R = conjunction of clauses

    private boolean noisy = Boolean.getBoolean("PANOISY");

    private static boolean doRestrict = !Boolean.getBoolean("NORESTRICT");

    public EnumMaxClausesFindMinAbstractor(jbddManager bddManager) {
	// bddManager.jbdd_num_vars
	this.bddManager = bddManager;
	R = bddManager.jbdd_zero();
    }
	
    public jbdd get() {
	return R;
    }

    public Vector getClauses() {
	return clauses;
    }

    public boolean union(Prover prover) {

	int notImpliedOldR = 0, impliedR = 0, ndisj=0;

	jbdd oldR = R;
	Vector oldClauses = clauses;

	R = bddManager.jbdd_one();
	clauses = new Vector();

	GenerateMaxClauses gen = new GenerateMaxClauses( bddManager );

	gen.restrict( oldR.jbdd_not() );

	for(int i=0; i<oldClauses.size(); i++) {
	    jbdd d = (jbdd)oldClauses.elementAt(i);
	    if( prover.check(d) ) {
		if( noisy ) say("Invariant still valid: "+prover.printClause(d));
		R = jbdd.jbdd_and( R, d, true, true );
		clauses.addElement(d);
		if(doRestrict) gen.restrict( d );
	    }
	}

	jbdd bdd;
	while( (bdd = gen.next()) != null ) {

	    ndisj++;
	    if( noisy ) say("bdd = "+prover.printClause( bdd ));

	    if( !oldR.jbdd_leq( bdd, true, true ) ) {
		notImpliedOldR++;
		say("not implied by oldR");
		continue;
	    }

	    if( R.jbdd_leq( bdd, true, true ) ) {
		impliedR++;
		say("implied by curInv");
		continue;
	    }

	    if( prover.check(bdd)) {
		// bdd valid, find subset that is valid
		jbdd minClause = findMinClauseValid( oldR, prover, bddManager.jbdd_zero(), bdd);
		if( noisy ) say("Invariant: "+prover.printClause(minClause));
		R = jbdd.jbdd_and( R, minClause, true, true );
		clauses.addElement(minClause);
		if(doRestrict) gen.restrict( minClause );
	    } else {
		//bdd.jbdd_free();
	    }
	}
	System.out.println("Clauses: "+ndisj
			   +" notImpliedOldR="+notImpliedOldR
			   +" impliedR="+impliedR);
	System.out.println("Prover: "+prover.report());

	return oldR.jbdd_equal( R );
    }

    private jbdd findMinClauseValid( jbdd oldR, Prover prover, jbdd a, jbdd b) {
	if( noisy ) 
	    say( "findMinClauseValid("+prover.printClause(a)+", "+prover.printClause(b)+")");

	if( !b.jbdd_is_tautology(false)) {
	    jbdd t = bddManager.jbdd_get_variable( b.jbdd_top_var_id() );
	    jbdd thn = b.jbdd_then();
	    jbdd b1, b2;
	    if( thn.jbdd_is_tautology(true)) {
		b1 = t;
		b2 = b.jbdd_else();
	    } else {
		b1 = b.jbdd_ite( t, bddManager.jbdd_zero(), bddManager.jbdd_one(), 
				 true, true, true );
		b2 = thn;
	    }
	    jbdd aorb2 = jbdd.jbdd_or( a, b2, true, true );
	    if( oldR.jbdd_leq( aorb2, true, true ) &&
		prover.check(aorb2) ) {
		return findMinClauseValid( oldR, prover, a, b2 );
	    } else {
		return findMinClauseValid( oldR, prover, jbdd.jbdd_or( a, b1, true, true), b2 );
	    }
	} else {
	    return a;
	}
    }	

    private jbdd invertLiterals(jbdd t) {
	if( t.jbdd_is_tautology( true ) || t.jbdd_is_tautology( false ) ) {
	    return t;
	} else {
	    return jbdd.jbdd_ite( t.jbdd_top_var(),
				  t.jbdd_else(),
				  t.jbdd_then(),
				  true, true, true );
	}
    }

    private void say(String s) {
	if( noisy ) {
	    System.out.println(s);
	}
    }

}
