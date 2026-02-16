package escjava.sortedProver.cvc3;

import javafe.util.ErrorSet;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

public class Cvc3Term implements STerm {
    // wrapped cvc expr
    private final Expr expr;
    
    public Cvc3Term() {
	assert(false);
	ErrorSet.fatal("Cvc3Term: empty constructor: ");
	this.expr = null;
    }

    public Cvc3Term(Expr expr) {
	Cvc3Prover.print("Cvc3Term: " + expr);
	this.expr = expr;
    }

    public Expr getExpr() {
	return expr;
    }

    public String toString() {
	return getExpr().toString();
    }
    
}
