/* Copyright Hewlett-Packard, 2002 */

package escjava.pa.generic;

import java.util.Random;
import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;

import javafe.util.Set;
import javafe.util.Location;
import javafe.util.Assert;
import javafe.util.StackVector;

import mocha.wrappers.jbdd.*;

/* Abstractor that is like EnumMaxClausesFindMinAbstractor,
   but only enumerates enough maximal clauses to ensure that
   all invariants of length k are found.
*/



public class EnumNFindK implements Abstractor {

    private jbddManager bddManager;

    private int k;
    private jbdd R;
    private Vector clauses = new Vector();
    private Vector disj = new Vector();
    // invariant: R = conjunction of clauses
    // clauses are bdds, disj are Disjunctions, otherwise identical

    private Prover prover;

    private boolean noisy = Boolean.getBoolean("PANOISY");
    private static boolean invLeqK = Boolean.getBoolean("INVLEQK");

    private final long seed = 0xcafcaf;
    private Random random = new Random(seed);

    static {
	System.out.println("invLeqK="+invLeqK);
    }
    
    public EnumNFindK(jbddManager bddManager, int k) {
	if( k > bddManager.jbdd_num_vars() ) {
	    k = bddManager.jbdd_num_vars();
	}
	this.k = k;
	// bddManager.jbdd_num_vars
	this.bddManager = bddManager;
	R = bddManager.jbdd_zero();
	clauses.addElement( R );
	disj.addElement(new Disjunction()); // Yields Disjunction for false
    }
	
    public jbdd get() {
	return R;
    }

    public Vector getClauses() {
	return clauses;
    }

    private void add(Disjunction d, DisjunctionProver disjProver) {
	jbdd b = disjProver.disjToBdd(d);
	R = jbdd.jbdd_and( R, b, true, true );
	clauses.addElement(b);
	disj.addElement(d);
    }

    public boolean union(Prover prover) {

	int nclauses=0, kclauses=0;

	Vector oldDisj = disj;
	jbdd oldR = R;

	R = bddManager.jbdd_one();
	clauses = new Vector();
	disj = new Vector();

	this.prover = prover;
	DisjunctionProver disjProver = new DisjunctionProver( prover, bddManager );

	for(int i=0; i<oldDisj.size(); i++) {
	    Disjunction d = (Disjunction)oldDisj.elementAt(i);
	    if( disjProver.check(d) ) {
		if( noisy ) 
		    say("Invariant still valid: "+disjProver.printClause(d));
		add(d, disjProver);
	    } 
	}
	if( disj.size() == oldDisj.size() ) {
	    // all of the old invariants are still invariants
	    // so reachable space is not any bigger,
	    // and certainly cannot be any smaller
	    return true;
	}

	for(EnumKofN enum = new EnumKofN(k, bddManager.jbdd_num_vars());
	    enum.getNext(); ) {

	    kclauses++;
	    
	    if( disjProver.quickCheck(enum) == disjProver.UNKNOWN ) {
		
		if( noisy ) say("kbdd = "+disjProver.printClause( enum ));

		// Try to find extension to n-string that is unknown
		
		Disjunction nd = new Disjunction(enum);
		if( !extendToMaxDisjUnknown(nd,0,disjProver) ) {
		    Assert.fail("Problem extending "+disjProver.printClause( enum )
				+" to maximal disjunction of unknown validity");
		}

		Assert.notFalse(disjProver.quickCheck(nd) == disjProver.UNKNOWN );
		nclauses++;

		if( noisy ) {
		    say("nbdd = "+disjProver.printClause( nd) +" quickcheck "+ disjProver.quickCheck(nd));
		}

		Assert.notFalse( disjProver.quickCheck(nd) != disjProver.INVALID );

		if( disjProver.check(nd)) {
		    // nd valid, find subset that is valid
		    long usedBits = ~(-1L << bddManager.jbdd_num_vars());
		    findMinDisjValid( nd, disjProver, enum.stars & usedBits);
		    findMinDisjValid( nd, disjProver,~enum.stars & usedBits);

		    if( !invLeqK || size(nd) <= k ) {
			if( !disj.contains(nd) ) {
			    if( noisy ) 
				say("Invariant: "+disjProver.printClause(nd));
			    add(nd, disjProver);
			} else {
			    if( noisy ) 
				say("Repeated invariant: "+disjProver.printClause(nd));
			}
		    } else {
			if( noisy )
			    say("invariant too long: "
				+disjProver.printClause(nd));
		    }
		}
	    
		Assert.notFalse( disjProver.quickCheck(enum) != prover.UNKNOWN );
	    }
	}
	
	System.out.println("kClauses="+kclauses
			   +" nClauses="+nclauses);
	System.out.println("Prover: "+prover.report());

	return oldR.jbdd_equal( R );
    }

    // requires d valid, mutates d, leaves it valid
    private void findMinDisjValid( Disjunction d, 
				   DisjunctionProver disjProver, 
				   long dropWhich) {

	if( noisy ) 
	    say( "findMinClauseValid("+disjProver.printClause(d)
		 +", "+Long.toBinaryString(dropWhich)+")");

	for(int i=0; i<64; i++) {
	    long b = 1L<<i;

	    if( (dropWhich & b) == 0 ) continue;
	    if( (d.stars & b) != 0 ) continue;

	    long oldStars = d.stars;
	    long oldBits = d.bits;

	    d.stars |= b;
	    d.bits &= ~b;
	    
	    //if( noisy ) say( "findMinClauseValid checking "+disjProver.printClause(d));
	    if( disjProver.check(d) ) continue;

	    // could not drop, but it back
	    d.stars = oldStars;
	    d.bits = oldBits;
	}

	if( noisy ) 
	    say( "findMinClauseValid returning "+disjProver.printClause(d));

    }	

    private boolean extendToMaxDisjUnknown(Disjunction nd, int i, DisjunctionProver disjProver) {
	//say("extendToMaxDisjUnknown("+disjProver.printClause( nd)+","+i+")");
	
	Assert.notFalse(disjProver.quickCheck(nd) == disjProver.UNKNOWN );

	long bit = (1L<<i);
	if( i == bddManager.jbdd_num_vars() ) {
	    return true;
	} else if( (nd.stars & bit) == 0 ) {
	    // not a star at this bit, go to next
	    return extendToMaxDisjUnknown(nd, i+1, disjProver);
	} else {
	    nd.stars &= ~bit;
	    long r = random.nextLong();
	    for(int sign=0; sign<2; sign++) {
		nd.bits = (nd.bits & ~bit) | (r & bit);
		if( disjProver.quickCheck(nd) == disjProver.UNKNOWN &&
		   extendToMaxDisjUnknown(nd, i+1, disjProver)) {
		    // can extend
		    return true;
		}
		// failed to extend, try other choice for bit
		r = ~r;
	    }
	    // both choices did not work, but star back in and backtrack

	    nd.stars |= bit;
	    return false; // could not extend
	}
    }
    
    private void say(String s) {
	if( noisy ) {
	    System.out.println(s);
	}
    }

    // return size of a disjunction
    static private int size(jbdd b) {
	int s=0;
	while( !b.jbdd_is_tautology(true) && !b.jbdd_is_tautology(false)) {
	    jbdd t = b.jbdd_then();
	    if( t.jbdd_is_tautology(true)) {
		//t.jbdd_free();
		t = b.jbdd_else();
	    }
	    //if( s != 0 ) b.jbdd_free();
	    s++;
	    b = t;	    
	}
	return s;
    }

    // return size of a disjunction
    private int size(Disjunction d) {
	int s=0;
	for(int i=0; i<bddManager.jbdd_num_vars(); i++) {
	    if( (d.stars & (1L<<i)) == 0) s++;
	}
	return s;
    }

    jbdd longsToBdd(long stars, long bits) {
	// Put choice into a bdd, 
	jbdd t = bddManager.jbdd_zero();
	for( int i=bddManager.jbdd_num_vars()-1; i>=0; i-- ) {
	    long b = 1L<<i;
	    if( (stars & b) == 0 ) {
		// no star
		jbdd varBdd = bddManager.jbdd_get_variable( i );
		jbdd t2 = jbdd.jbdd_or( t, varBdd, true, (bits & b) != 0 );
		// t.jbdd_free();
		t = t2;
	    }
	}
	return t;
    }
}
