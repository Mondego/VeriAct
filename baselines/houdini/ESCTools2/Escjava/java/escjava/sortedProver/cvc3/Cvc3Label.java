package escjava.sortedProver.cvc3;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

// a labeled expression
public class Cvc3Label extends Cvc3Pred {
    // pos. or neg. label
    private final boolean pos;
    // name of label
    private final String name;
    // the labelled boolean expression
    private final Cvc3Pred pred;
    // is this a major label?
    private final boolean major;

    public Cvc3Label(boolean pos, String name, Cvc3Pred pred) throws Cvc3Exception {
	super(pred.getExpr());
	markLabeled();
        pred.markLabeled();

	this.pos = pos;
	this.name = name;
	this.pred = pred;

	this.major = isMajor(name);

	this.addChild(pred);
    }
    
    public boolean isPos() {
	return pos;
    }

    public String getName() {
	return name;
    }

    public boolean isMajor() {
	return major;
    }

    public static boolean isMajor(String name) {
	return name.indexOf('@') != -1;
    }

    public Cvc3Pred getPred() {
	return pred;
    }
    

    public String toString() {
	if (isPos()) {
	    return ("LBLPOS: " + getName() + " " + getPred());
	} else {
	    return ("LBLNEG: " + getName() + " " + getPred());
	}
    }
}
