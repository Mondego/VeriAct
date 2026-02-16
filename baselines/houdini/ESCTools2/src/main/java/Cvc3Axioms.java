package escjava.sortedProver.cvc3;

import java.util.Iterator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;

import cvc3.*;


// tries to avoid asserting axioms not needed in a query.
//
// for exploration only, doesn't try to be efficient.
//
// on registerting an axiom P:
// - asserts
//     ProxyP : BOOLEAN = P;
//   to cvc
// - registers the symbols contained in P
public class Cvc3Axioms {

    // lazy or eager assertion of axioms?
    boolean lazy;

    // axioms in order of registration
    List axioms;

    // axiom -> contained symbols (HashSet)
    HashMap containedSymbols;
    // symbol -> containing it (HashSet)
    HashMap containingAxioms;

    public Cvc3Axioms(boolean _lazy) {
	lazy             = _lazy;
	axioms           = new LinkedList();
	containedSymbols = new HashMap();
	containingAxioms = new HashMap();
    }


    // register an axiom P:
    // - asserts ProxyP : BOOLEAN = P; to cvc
    // - registers the symbols contained in P
    public void register(cvc3.ValidityChecker prover, Expr axiom) throws Cvc3Exception {
	if (!lazy) {
	    prover.assertFormula(axiom);
	} else {
	    HashSet symbols = extractSymbols(axiom);
	    
	    // always assert symbol free axioms
	    if (symbols.isEmpty()) {
		prover.assertFormula(axiom);
	    } else {
		axioms.add(axiom);
		
		// axiom to symbols
		containedSymbols.put(axiom, symbols);
		
		// symbol to axioms
		Iterator i = symbols.iterator();
		while (i.hasNext()) {
		    Object symbol = i.next();
		    HashSet otherAxioms = (HashSet) containingAxioms.get(symbol);
		    if (otherAxioms == null) {
			otherAxioms = new HashSet();
			containingAxioms.put(symbol, otherAxioms);
		    }
		    otherAxioms.add(axiom);
		}
	    }
	}
    }

    // :TODO: more efficient
    // assert all axioms needed by the query
    public void assertRequired(cvc3.ValidityChecker prover, Expr query) throws Cvc3Exception {
	if (lazy) {
	    Iterator i;
	    Iterator j;

	    HashSet symbols = extractSymbols(query);
	    // :TODO: for readability order asserted axioms
	    //TreeSet requiredAxioms = new TreeSet();
	    HashSet requiredAxioms = new HashSet();

	    // :TODO: queue data structure for HashSet
	    HashSet processedSymbols = new HashSet();
	    // for each symbol the query depends on ...
	    while (!symbols.isEmpty()) {
		// move symbol from unprocessed to processed symbols
		Object symbol = symbols.iterator().next();
		symbols.remove(symbol);
		processedSymbols.add(symbol);

		// for each axiom containing symbol
		HashSet symbolAxioms = (HashSet) containingAxioms.get(symbol);
		if (symbolAxioms != null) {
		    i = symbolAxioms.iterator();
		    while (i.hasNext()) {
			Object axiom = i.next();

			// add axiom to set of required axioms
			requiredAxioms.add(axiom);

			// consider symbols contained in axiom
			HashSet axiomSymbols = (HashSet) containedSymbols.get(axiom);
			assert (axiomSymbols != null);
			j = axiomSymbols.iterator();
			while (j.hasNext()) {
			    Object next = j.next();
			    if (!processedSymbols.contains(next)) {
				symbols.add(next);
			    }
			}
		    }
		}
	    }

	    // assert reach required axiom
	    i = axioms.iterator();
	    while (i.hasNext()) {
		Expr axiom  = (Expr) i.next();
		if (requiredAxioms.contains(axiom)) {
		    prover.assertFormula(axiom);
		}
	    }

	}
    }



    protected String freshProxyName() {
	return ("escjava_proxy_" + new Integer(axioms.size()).toString());
    }



    protected void extractSymbolsChildren(HashSet symbols, Expr expr) throws Cvc3Exception {
	for (int i = 0; i < expr.arity(); ++i) {
	    extractSymbols(symbols, expr.getChild(i));
	}
    }

    protected void extractSymbols(HashSet symbols, Expr expr) throws Cvc3Exception {
	// uninterpreted constants
	if (!expr.isBoundVar() && expr.isVar()) {
	    symbols.add(expr);

	// uninterpreted functions
	} else if (expr.isApply()) {
	    symbols.add(expr.getOp());
	    extractSymbolsChildren(symbols, expr);

	// quantified expressions
	} else if (expr.isClosure()) {
	    extractSymbols(symbols, expr.getBody());

	// everything else
	} else { // if (expr.arity() > 0) {
	    extractSymbolsChildren(symbols, expr);
	}
    }

    protected HashSet extractSymbols(Expr axiom) throws Cvc3Exception {
	HashSet symbols = new HashSet();
	extractSymbols(symbols, axiom);

	// :DEBUG:
	System.out.println (axiom);
	System.out.println ("symbols:");
	Iterator i = symbols.iterator();
	while (i.hasNext()) {
	    System.out.println(i.next());
	}

	return symbols;
    }
}

