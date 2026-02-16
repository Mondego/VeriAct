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
import escjava.prover.*;
import escjava.pa.generic.*;

import mocha.wrappers.jbdd.*;

public class PredicateAbstraction {

    public static ASTDecoration paDecoration = new ASTDecoration("paDecoration");
    
    static GuardedCmd abstractLoop(LoopCmd g, GuardedCmd context, Set env) {
	PredicateAbstraction pa = (PredicateAbstraction) paDecoration.get(g);	
	if (pa == null) {
	    pa = new PredicateAbstraction(g, env);
	    paDecoration.set(g, pa);
	}
	return pa.abstractLoopHelper(context, env);
    }

    private static boolean quantifyAssumptions = Boolean.getBoolean("PAquantifyAssumptions");
    ExprVec invariants = ExprVec.make();
    private jbddManager bddManager;
    public Abstractor abstractor;
    private LocalVarDeclVec skolemConstants;
    private ExprVec loopPredicates;
    private GuardedCmd body;
    private GuardedCmd bodyDesugared = GC.fail();
    private GuardedCmd havoc;
    private int startLoc;
    public int nQueries=0;
    long milliSecsUsed;
    GCProver perfCount;
    
    private final StackVector code = new StackVector();
    private final GenericVarDeclVec temporaries = GenericVarDeclVec.make();

    PredicateAbstraction(LoopCmd g, Set env) {

	body = GC.seq(g.guard, g.body);
	startLoc = g.getStartLoc();

	Set vds = Targets.normal(body);

	if( Main.inferPredicates ) {
	    inferPredicates(g, env, vds);
	}
	
	this.skolemConstants = g.skolemConstants;
	this.loopPredicates = g.predicates;
	this.bddManager = new jbddManager( loopPredicates.size() );

	if( System.getProperty("PABDT") != null ) {
	    this.abstractor = new BinaryDecisionTreeAbstractor(bddManager);
	} else if( System.getProperty("PA3n") != null ) {
	    this.abstractor = new EnumClausesAbstractor(bddManager);
	} else if( System.getProperty("PANK") != null ) {
	    this.abstractor = new EnumNFindK(bddManager,
					     Integer.parseInt(System.getProperty("PANK")));
	} else {
	    this.abstractor = new EnumMaxClausesFindMinAbstractor(bddManager);
	}

	code.push();
	for (Enumeration e = vds.elements(); e.hasMoreElements();) {
	    int loc = g.locStart;
	    GenericVarDecl at = (GenericVarDecl)e.nextElement();
	    VariableAccess va = VariableAccess.make(at.id, loc, at);	    
	    String s = va.id.toString();
	    Identifier idn = Identifier.intern(s);
	    VariableAccess result = GC.makeVar(idn, loc);
	    temporaries.addElement(result.decl);
	    result.loc = loc;
	    code.addElement(GC.gets(va, result));
	}
	for (Enumeration e = vds.elements(); e.hasMoreElements();) {
	    GenericVarDecl vd = (GenericVarDecl)e.nextElement();
	    
	    if (!vd.id.toString().endsWith("@init")) {
		code.addElement(GC.assume(TrAnExpr.targetTypeCorrect(vd, g.oldmap)));
	    }
	}	
	for (int i = 0; i < g.invariants.size(); i++) {
	    Condition cond = g.invariants.elementAt(i);
	    code.addElement(GC.assume(cond.pred));
	}

	havoc = GC.seq(GuardedCmdVec.popFromStackVector(code));
    }

    private GuardedCmd abstractLoopHelper(GuardedCmd context, Set env) {
	int step = 0;
	milliSecsUsed -= java.lang.System.currentTimeMillis();

	code.push();
	for(int j=0; j<skolemConstants.size();j++) {
	    LocalVarDecl sc = skolemConstants.elementAt(j);
	    code.addElement(GC.assume(TrAnExpr.typeAndNonNullCorrectAs(sc, sc.type, null, true, null)));
	}
	context = GC.seq( context, 
			  GC.seq( GuardedCmdVec.popFromStackVector(code)));


	System.out.println("Step " + step + ": Computing for loop at "
			   +Location.toString( startLoc )
			   +" num preds = "+loopPredicates.size()
			   + "....");

	perfCount = new GCProver(bddManager, loopPredicates, context, null);
	
	GCProver p = new GCProver(bddManager, loopPredicates, context, null);
	boolean atfixpoint = abstractor.union(p);
	p.done();
	perfCount.addPerfCounters(p);

	System.out.println("  reachable size: "+p.size(abstractor.get()));
	
	while (!atfixpoint) {
	    ExprVec invs = p.concretize(abstractor.getClauses());

	    if( quantifyAssumptions ) {
		invs = skolemQuantInvariants(skolemConstants, invs,
					     Location.NULL, Location.NULL);
	    }

	    System.out.println("Step " + ++step + ": Computing....");
	    GuardedCmd c = GC.seq( context, havoc, 
				   GC.assume(GC.and(invs)));
	    // from shaz
	    // c = GC.seq( context, havoc, GC.assume(e) );
	    milliSecsUsed += java.lang.System.currentTimeMillis();
	    bodyDesugared = Traverse.computeHelper(body, c, env);	
	    milliSecsUsed -= java.lang.System.currentTimeMillis();

	    if( Main.pgc ) {
		System.out.println("\n**** Guarded Command c:");
		((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, c);
		System.out.println("");
		System.out.println("\n**** Guarded Command body:");
		((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, body);
		System.out.println("");
		System.out.println("\n**** Guarded Command bodyDesugared:");
		((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, bodyDesugared);
		System.out.println("");
	    }
	    p = new GCProver(bddManager, loopPredicates, GC.seq(c, bodyDesugared), p);
	    atfixpoint = abstractor.union(p);
	    p.done();
	    perfCount.addPerfCounters(p);

	    System.out.println("  reachable size: "+p.size(abstractor.get()));
	}

	invariants = skolemQuantInvariants(skolemConstants,
					   p.concretize(abstractor.getClauses()),
					   Location.NULL,
					   Location.NULL);

	milliSecsUsed += java.lang.System.currentTimeMillis();
	System.out.println("Finished loop at "
			   +Location.toString( startLoc ) );
	return VarInCmd.make(temporaries, 
			     GC.seq(havoc, 
				    GC.assume(GC.and(invariants)),
				    bodyDesugared,
				    GC.fail()));
    }

    public static ExprVec skolemQuantInvariants(LocalVarDeclVec skolemConstants,
						ExprVec invs,
						int sloc, int eloc) {
				  
	// Now, assume the inferred invariants at the beginning of the loop,
	// universally quantified by the skolem constants
	    
	ExprVec r = ExprVec.make();

	for(int i=0; i<invs.size(); i++) {
	    Expr inv = invs.elementAt(i);
	    
	    GenericVarDeclVec decls = GenericVarDeclVec.make();
	    ExprVec ttc = ExprVec.make();
	    for(int j=0; j<skolemConstants.size();j++) {
		LocalVarDecl sc = skolemConstants.elementAt(j);
		if( mentions( inv, sc ) ) {
		    decls.addElement(sc);
		    ttc.addElement( TrAnExpr.typeAndNonNullCorrectAs(sc, sc.type, null, true, null) );
		}
	    }

	    Expr f = GC.quantifiedExpr( sloc, eloc,
					TagConstants.FORALL,
					decls,
					GC.implies( GC.and( ttc ),
						    inv ),
					    null );
	    r.addElement( f );
	}
	
	return r;
    }

    private static boolean mentions(Expr e, GenericVarDecl d) {
	if( e instanceof VariableAccess ) {
	    return ((VariableAccess)e).decl == d;
	} else {
	    for(int i=0; i<e.childCount(); i++) {
		Object c = e.childAt(i);
		if( c instanceof Expr && mentions((Expr)c,d) ) {
		    return true;
		}
	    }
	    return false;
	}
    }

    private void inferPredicates(LoopCmd g, Set env, Set targets) {
	Set t = new Set(targets.elements());
	t.intersect(env);

	for (Enumeration e = t.elements(); e.hasMoreElements();) {
	    GenericVarDecl vd = (GenericVarDecl)e.nextElement();
	    if( vd.type != null ) {
		if( escjava.tc.Types.isIntegralType( vd.type ) ) {
		    // guess vd >= 0
		    ExprVec vec = ExprVec.make();
		    int loc = g.getStartLoc();
		    vec.addElement( VariableAccess.make(vd.id, loc, vd) );
		    vec.addElement( LiteralExpr.make( TagConstants.INTLIT,
								   new Integer(0),
								   loc ));
		    
		    Expr pred = NaryExpr.make( loc, loc, TagConstants.INTEGRALGE, vec );
		    g.predicates.addElement( pred );
		}

		if ( escjava.tc.Types.isReferenceType( vd.type ) ) {
		    // guess vd != null
		    ExprVec vec = ExprVec.make();
		    int loc = g.getStartLoc();
		    vec.addElement( VariableAccess.make(vd.id, loc, vd) );
		    vec.addElement( LiteralExpr.make( TagConstants.NULLLIT,
								   null,
								   loc ));
		    
		    Expr pred = NaryExpr.make( loc, loc, TagConstants.REFNE, vec );
		    g.predicates.addElement( pred );		    
		}
	    }
	}
    }
    /*
    private void computeMentionsSet(ASTNode n, Set s) {
	if( n instanceof VariableAccess ) {
	    VariableAccess va = (VariableAccess)n;
	    if( n.decl != null ) {
		s.add(n.decl);
	    }
	}
	for(int i=0; i<n.childCount(); i++) {
	    Object c = n.childAt(i);
	    computeMentionsSet(c,s);
	}
    }
    */
	
}
