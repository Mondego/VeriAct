/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.pa.generic;

import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;
import java.io.*;

import mocha.wrappers.jbdd.*;

/* Enumerates maximal clauses.
 */

public class GenerateMaxClauses {

    private jbddManager bddmanager;
    jbdd bdd;
    boolean maxClause[];

    public GenerateMaxClauses(jbddManager bddmanager) {
	this.bddmanager = bddmanager;
	bdd = bddmanager.jbdd_one();
	maxClause = new boolean[ bddmanager.jbdd_num_vars() ];

	for( int i=0; i<bddmanager.jbdd_num_vars(); i++ ) {
	    maxClause[i] = true;
	}

    }

    public void restrict(jbdd to) {
	bdd = jbdd.jbdd_and( bdd, to, true, true);
    }

    /** Put next max clause compatible with b into maxClause[i..n-1]
     * and return true (if such max clause exists),
     * or put true  into maxClause[i..n-1]
     * and return false.
     */	

    public boolean findValidMaxClause(int i, jbdd b) {

	if( b.jbdd_is_tautology(true) ) return true;

	if( i == bddmanager.jbdd_num_vars() ) return false;

	for(;;) {
	    if( findValidMaxClause(i+1,
		     (i == b.jbdd_top_var_id())
		     ? ( maxClause[i] ? b.jbdd_then() : b.jbdd_else())
		     : b )) {
		return true;
	    } else if( maxClause[i] == false) {
		maxClause[i] = true;
		return false;
	    } else {
		maxClause[i] = false;
		// go around loop
	    }
	}
    }
	
    public jbdd next() {
	
	if( maxClause == null ) return null;

	if( findValidMaxClause( 0, bdd ) == false ) {
	    maxClause = null;
	    return null;
	}

	// Put maxClause into a bdd, 
	jbdd t = bddmanager.jbdd_zero();
	for( int i=bddmanager.jbdd_num_vars()-1; i>=0; i-- ) {
	    jbdd varBdd = bddmanager.jbdd_get_variable( i );
	    t = jbdd.jbdd_or( t, varBdd, true, !maxClause[i] );
	    // System.out.println("maxClause["+i+"]="+maxClause[i] );
	}

	// decrement maxClause
	int i;
	for( i=bddmanager.jbdd_num_vars()-1; i>=0 && maxClause[i] == false; i-- ) {
	    maxClause[i] = true;
	}

	if( i >= 0 ) {
	    maxClause[i] = false;
	} else {
	    // tried all maxClauses
	    maxClause = null;
	}

	return t;
    }
}
