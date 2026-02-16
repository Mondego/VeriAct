package escjava.sortedProver.cvc3;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

// any reference value: object, array
public class Cvc3Ref extends Cvc3Value implements SRef {

    public Cvc3Ref() {
    }

    public Cvc3Ref(Expr e) {
	super(e);
    }
}
