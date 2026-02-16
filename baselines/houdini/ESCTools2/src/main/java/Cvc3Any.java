package escjava.sortedProver.cvc3;

import javafe.util.ErrorSet;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

public class Cvc3Any extends Cvc3Term implements SAny {
    public Cvc3Any() {
	ErrorSet.fatal("Cvc3Any: empty constructor: ");
    }

    public Cvc3Any(Expr any) {
	super(any);
    }

    public boolean isSubSortOf(Sort s) {
	//:TODO:
	return true;
    }
}
