package escjava.sortedProver.cvc3;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

public class Cvc3Bool extends Cvc3Value implements SBool {
    public Cvc3Bool() {
    }

    public Cvc3Bool(Expr e) {
	super(e);
    }
}
