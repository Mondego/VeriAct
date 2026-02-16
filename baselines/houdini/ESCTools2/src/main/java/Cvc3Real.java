package escjava.sortedProver.cvc3;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

// implementing SInt is not really ok,
// but this allows to map the type Time to Real in the cvc3 encoding,
// but map it to Int in the general type inference.
// And Int is more efficient for Simplify,
// as otherwise
//  (EQ |alloc@pre| alloc)
// can be mapped to
//  (EQ (floatingEQ |alloc@pre| alloc) |@true|)
public class Cvc3Real extends Cvc3Value implements SReal, SInt {

    public Cvc3Real() {
    }

    public Cvc3Real(Expr e) {
	super(e);
    }
}
