/* Copyright Hewlett-Packard, 2002 */

package escjava.pa;

import java.util.Hashtable;
import java.util.Enumeration;
import java.util.Vector;
import java.io.*;

import javafe.ast.*;
import javafe.util.Set;
import javafe.util.Location;
import javafe.util.Assert;
import javafe.util.StackVector;

import escjava.*;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.translate.*;
import escjava.sp.SPVC;
import escjava.sp.*;
import escjava.pa.generic.*;
import escjava.prover.*;

import mocha.wrappers.jbdd.*;

public class GCProver implements Prover {

    private jbddManager bddManager;
    private ExprVec loopPredicates;

    private boolean noisy = Boolean.getBoolean("PANOISY");

    public jbdd valid;
    public Vector validClauses = new Vector();
    private Vector allInvalidClauses = new Vector();
    private jbdd oldValid;
    
    private PrintStream ps = Main.prover.subProcessToStream();
    private VarMap subst;
    int queriesConsidered=0, queriesTried=0, queriesValid=0;
    long milliSecsUsed;

    public GCProver(jbddManager bddManager, 
		    ExprVec loopPredicates,
		    GuardedCmd g,
		    GCProver oldProver) {
	
	this.bddManager = bddManager;
	this.loopPredicates = loopPredicates;
	valid = bddManager.jbdd_one();

	oldValid =  oldProver == null ? bddManager.jbdd_zero() : oldProver.valid;

	VarMap[] out = new VarMap[2];
	GuardedCmd gc = DSA.dsa(g,out);
	Expr vc = SPVC.computeN(gc);
	subst = out[0];
	ps.print("\n(BG_PUSH ");
	VcToString.computePC(vc, ps);
	ps.println(")");
	Main.prover.sendCommands("");

    }

    public boolean check(jbdd query) {

	if( noisy ) say("query = "+printClause( query ));

	queriesConsidered++;

	switch( quickCheck( query )) {
	  case VALID:
	    return true;

	  case INVALID:
	    return false;

	  case UNKNOWN:
	    // query is possible invariant, call simplify
	    milliSecsUsed -= java.lang.System.currentTimeMillis();
	    Main.prover.startProve();
	    VcToString.compute( subst.apply(concretize( query )),
				ps);
	    Enumeration results = Main.prover.streamProve();
	    boolean queryvalid = processSimplifyOutput(results);
	    if( noisy ) say( "SIMPLIFY: "+(queryvalid ? "valid" : "invalid"));
	    queriesTried++;
	    milliSecsUsed += java.lang.System.currentTimeMillis();
	    
	    if( queryvalid ) {
		queriesValid++;
		
		validClauses.add( query );
		jbdd t = jbdd.jbdd_and( valid, query, true, true );
		// valid.jbdd_free();
		valid = t;
		return true;
	    } else {
		// query not tautology or contradictory
		// maybe some extension is an invariant
		allInvalidClauses.add( query );
		return false;
	    }

	  default:
	    Assert.fail("");
	    return false;
	}
    }

    public int quickCheck(jbdd query) {

	// check if query not implied by oldValid
	if( !oldValid.jbdd_leq( query, true, true ) ) {
	    say("not implied by oldValid");
	    return INVALID;
	};

	// check if query redundant
	if( valid.jbdd_leq( query, true, true ) ) {
	    say("redundant");
	    return VALID;
	}

	// check if query contradictory
	if( valid.jbdd_leq( query, true, false ) ) {
	    say("contradictory");
	    return INVALID;
	};

	// Chk if query an extension of something in validClauses

	boolean queryMaybeValid = true;

	// query is possible invariant, 
	// see if there exists d in allInvalidClauses such that
	// valid => (query => d)
	// if so, query is not valid
	// Optimized check is (valid /\ query) => d
	
	jbdd validAndQuery = jbdd.jbdd_and(valid, query, true, true);
	    
	for(Enumeration e2 = allInvalidClauses.elements(); 
	    e2.hasMoreElements() && queryMaybeValid; ) {
	    jbdd clause = (jbdd)e2.nextElement();
	    if( validAndQuery.jbdd_leq( clause, true, true ) ) {
		// query not a tautology
		say("invalid by earlier test");
		return INVALID;
	    }
	}
	
	// validAndQuery.jbdd_free();

	return UNKNOWN;
    }

    public String report() {
	return("Considered "+ queriesConsidered
	       +" tried "+queriesTried
	       +" valid "+queriesValid +
	       " simplify-time "+milliSecsUsed+"ms");
    }

    public void addPerfCounters(GCProver p) {
	queriesConsidered += p.queriesConsidered;
	queriesTried += p.queriesTried;
	queriesValid += p.queriesValid;
	milliSecsUsed += p.milliSecsUsed;
    }

    public void done() {
	Main.prover.sendCommand("(BG_POP)");
    }
	    
    public String printClause(jbdd b) {
	if (b.jbdd_is_tautology(true)) {
	    return "TRUE";
	}
	else if (b.jbdd_is_tautology(false)) {
	    return "";
	}
	else {
	    int n = b.jbdd_top_var_id();
	    if( b.jbdd_then().jbdd_is_tautology(true)) {
		// n is positive
		return "p"+n+"-1 "+ printClause( b.jbdd_else());
	    } else {
		// n is negative
		return "p"+n+"-0 "+ printClause( b.jbdd_then());
	    }
	}
    }

    private void say(String s) {
	if( noisy ) {
	    System.out.println(s);
	}
    }

    boolean processSimplifyOutput(Enumeration results) {
	boolean valid = false;
	while (results.hasMoreElements()) {
	    SimplifyOutput so = (SimplifyOutput)results.nextElement();
	    switch (so.getKind()) {
	    case SimplifyOutput.VALID: {
		valid = true;
		Assert.notFalse(!results.hasMoreElements());
		break;
	    }

	    case SimplifyOutput.INVALID: 
	    case SimplifyOutput.UNKNOWN:
	    case SimplifyOutput.COMMENT: 
	    case SimplifyOutput.COUNTEREXAMPLE:
	    case SimplifyOutput.EXCEEDED_PROVER_KILL_TIME:
	    case SimplifyOutput.EXCEEDED_PROVER_KILL_ITER:
	    case SimplifyOutput.REACHED_CC_LIMIT:
	    case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_TIME:
	    case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_ITER:
	    case SimplifyOutput.WARNING_TRIGGERLESS_QUANT:
		break;

	    default:
	      Assert.fail("unexpected type of Simplify output");
	      break;
	    }
	}
	return valid;
    }

    ExprVec concretize(Vector clauses) {
	ExprVec r = ExprVec.make( clauses.size() );
	for(int i=0; i<clauses.size(); i++) {
	    r.addElement( concretize( (jbdd)clauses.elementAt(i) ));
	}
	return r;
    }

    Expr concretize(jbdd b) {
	if (b.jbdd_is_tautology(true)) {
	    return GC.truelit;
	}
	else if (b.jbdd_is_tautology(false)) {
	    return GC.falselit;
	}
	else {
	    Expr e = loopPredicates.elementAt(b.jbdd_top_var_id());
	    jbdd thn = b.jbdd_then();
	    jbdd els = b.jbdd_else();

	    if( thn.jbdd_is_tautology(true)) {
		return GC.or( e, concretize( els ));
	    } else if( els.jbdd_is_tautology(true)) {
		return GC.or( GC.not(e), concretize( thn ));
	    } else {
		return GC.or( GC.and(e, concretize( thn )),
			      GC.and(GC.not(e), concretize( els )));
	    }
	}
    }

    public int size(jbdd b) {
	return size(b, bddManager.jbdd_num_vars());
    }

    public int size(jbdd b, int nbitsFree) {
	if( b.jbdd_is_tautology(false) ) {
	    return 0;
	}
	if( b.jbdd_is_tautology(true) ) {
	    int r=1;
	    while( nbitsFree-- >0 ) r = 2*r;
	    return r;
	}
	return size( b.jbdd_then(), nbitsFree-1 )
	    +  size( b.jbdd_else(), nbitsFree-1 );
    }

}
