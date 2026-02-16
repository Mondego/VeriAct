package escjava.sortedProver.cvc3;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import escjava.sortedProver.NodeBuilder.*;

import cvc3.*;

public class Cvc3Pred extends Cvc3Term implements SPred {
    // all the children of this node
    private final List children;
    // true if some node in the branch under this node is labeled
    private boolean labeled;

    public Cvc3Pred() {
	this.children = new ArrayList();
	this.labeled = false;
    }

    public Cvc3Pred(Expr pred) throws Cvc3Exception {
	super(pred);
	assert(pred.getType().isBoolean());
	this.children = new ArrayList();
	this.labeled = false;
    }



    public boolean hasChild(Cvc3Pred child) {
	Iterator i = getChildren().iterator();
	while (i.hasNext()) {
	    if (i.next() == child) return true;
	}
	return false;
    }

    public void addChild(Cvc3Pred child) {
	getChildren().add(child);
	if (child.isLabeled()) {
	    markLabeled();
	}
    }

    public List getChildren() {
	return children;
    }


    public boolean isLabeled() {
	return labeled;
    }

    public void markLabeled() {
	labeled = true;
    }


//     public boolean hasLabel(Cvc3Label label) {
// 	Iterator i = getLabels().iterator();
// 	while (i.hasNext()) {
// 	    if (i.next() == label) return true;
// 	}
// 	return false;
//     }
    
//     public void addLabel(Cvc3Label label) {
// 	assert(!hasLabel(label));
// 	getLabels().add(label);
// 	label.extendPath(this);
//     }

//     public List getLabels() {
// 	return labels;
//     }
}
