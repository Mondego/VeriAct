package escjava.sortedProver.cvc3;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

public class Cvc3Int extends Cvc3Value implements SInt {

    public Cvc3Int() {
    }

    public Cvc3Int(Expr e) {
	super(e);
    }
}
