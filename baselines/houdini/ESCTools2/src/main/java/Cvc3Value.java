package escjava.sortedProver.cvc3;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

// a symbol?
public class Cvc3Value extends Cvc3Any implements SValue {

    public Cvc3Value() {
    }
		      
    public Cvc3Value(Expr value) {
	super(value);
    }
}
