package escjava.sortedProver.cvc3;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

public class Cvc3Map extends Cvc3Ref implements SMap {
    
    public Cvc3Map() {
    }

    public Cvc3Map(Expr expr) {
	super(expr);
    }
}
