package escjava.sortedProver.cvc3;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import cvc3.*;

/*@ non_null_by_default @*/

// :TODO: Assumption: no labels in ite condition
public class Cvc3Labels {
    // the cvc instance
    final private cvc3.ValidityChecker prover;

    // mapping from a label (name) to all expressions labeled with it
    // String -> List<Cvc3Label>
    final private HashMap labels = new HashMap();

    // mapping from the labels occurring in the current query
    // to their paths in the query, i.e. all occurrences
    // of the labeled expression in the query.
    // Cvc3Label -> List<List<Expr>>
    final private HashMap paths = new HashMap();



    public Cvc3Labels(cvc3.ValidityChecker prover) throws cvc3.Cvc3Exception {
	this.prover = prover;
    }



    protected HashMap getLabels() {
	return labels;
    }

    protected HashMap getPaths() {
	return paths;
    }


    // returns all expressions for the given label
    public List getLabeled(String label) {
	return (List) getLabels().get(label);
    }


    // register label occurrences in current query
    protected void registerLabel(Cvc3Label label, ArrayList path) {
	// Note: same label name might be used for several expressions
	List currentLabels = (List) getLabels().get(label.getName());

	// first occurrence
	if (currentLabels == null) {
	    currentLabels = new ArrayList();
	    getLabels().put(label.getName(), currentLabels);
	}

	currentLabels.add(label);



	//System.out.println("registerLabel:");
	//System.out.println(label);
	List currentPaths = (List) getPaths().get(label);

	// first occurrence
	if (currentPaths == null) {
	    currentPaths = new ArrayList();
	    getPaths().put(label, currentPaths);
	}

	currentPaths.add(path);
    }


    
    // compute for each label occurring in the predicate expression
    // the branch from the root of the query to the labeled expression.
    // this is done by a depth first traversal of the query,
    // where path represents the (reversed) path from the root to the current node.
    protected void computeLabelContexts(Cvc3Pred pred, List path) throws cvc3.Cvc3Exception {
	if (!(pred.isLabeled())) {
	    return;
	}

	assert(!pred.getExpr().isTerm());

	path.add(0, pred);

	if (pred instanceof Cvc3Label) {
	    registerLabel((Cvc3Label) pred, new ArrayList(path));
	}
	
        // :TODO: we do not look inside quantified expressions ...
	Iterator i = pred.getChildren().iterator();
	while (i.hasNext()) {
	    computeLabelContexts((Cvc3Pred)i.next(), path);
	}

	path.remove(0);
    }
    

    public void computeLabelContexts(Cvc3Pred pred) throws cvc3.Cvc3Exception {
	getLabels().clear();
	getPaths().clear();
	computeLabelContexts(pred, new LinkedList());
    }



    // checks if an expressions is true in the current context
    protected boolean isTrue(Expr query) throws cvc3.Cvc3Exception  {
	Cvc3Prover.print("isTrue: " + query);

	cvc3.FormulaValue value = prover.getValue(query);

	if (value == cvc3.FormulaValue.TRUE) {
	    Cvc3Prover.print("TRUE");
	} else if (value == cvc3.FormulaValue.FALSE) {
	    Cvc3Prover.print("FALSE");
	} if (value == cvc3.FormulaValue.UNKNOWN) {
	    Cvc3Prover.print("UNKNOWN");
	}

	//assert(isEntailed(query) == (value == FormulaValue.TRUE));

	return (value == FormulaValue.TRUE);
    }

    // checks if an expressions is true in the current context
    protected boolean isFalse(Expr query) throws cvc3.Cvc3Exception  {
	Cvc3Prover.print("isFalse: " + query);

	cvc3.FormulaValue value = prover.getValue(query);

	if (value == cvc3.FormulaValue.TRUE) {
	    Cvc3Prover.print("TRUE");
	} else if (value == cvc3.FormulaValue.FALSE) {
	    Cvc3Prover.print("FALSE");
	} if (value == cvc3.FormulaValue.UNKNOWN) {
	    Cvc3Prover.print("UNKNOWN");
	}

	//assert(isEntailed(prover.notExpr(query)) == (value == FormulaValue.FALSE));

	return (value == FormulaValue.FALSE);
    }


    // checks if an expressions is entailed in the current context
    protected boolean isEntailed(Expr query) throws cvc3.Cvc3Exception  {
	Cvc3Prover.print("isEntailed: " + query);

	prover.push();	
	QueryResult result = prover.query(query);
	prover.pop();
	return (result == QueryResult.VALID);
    }


    // label relevance rules:
    //
    // the relevance of a label depends on the formula it is contained in.
    //
    // notation: rule
    //   ... LBLPOS P ... -> ...
    // means: the label is relevant in the current model
    // if the right hand side holds.
    //
    // negative labels: LBLNEG P
    // LBLNEG P         -> P is false
    // NOT LBLNEG P     -> LBLPOS NOT P
    // (LBLNEG P) AND Q -> LBLNEG (P AND Q)
    // Q AND (LBLNEG P) -> Q is true and LBLNEG (Q AND P)
    // (LBLNEG P) OR Q  -> LBLNEG (P OR Q)
    // Q OR (LBLNEG P)  -> LBLNEG (Q OR P)

    // positive labels are dual: LBLPOS P
    // LBLPOS P         -> P is true
    // NOT LBLPOS P     -> LBLNEG NOT P
    // (LBLPOS P) AND Q -> LBLPOS (P AND Q)
    // Q AND (LBLPOS P) -> LBLPOS (Q AND P)
    // (LBLPOS P) OR Q  -> LBLPOS (P OR Q)
    // Q OR (LBLPOS P)  -> Q is false and LBLPOS (Q OR P)

    // sense: interested in true or false expression (pos or neg label)?
    // label: the label we are interested in
    // child: the previous relevant expression in the path
    // path: the remainder of the path from the labeled expression to the root
    protected boolean isRelevant(boolean sense, Cvc3Label label, Cvc3Pred child, Iterator path)
	throws cvc3.Cvc3Exception {
	if (!path.hasNext()) {
	    return true;
	} else {
	    Cvc3Pred pred = (Cvc3Pred) path.next();

	    // this is itself a label, thus retrieve its predicate
	    if (pred instanceof Cvc3Label) {
		return isRelevant(sense, label, ((Cvc3Label)pred).getPred(), path);
		//return isRelevant(sense, label, pred, path);
	    }

	    Expr expr = pred.getExpr();
	    Cvc3Prover.print("isRelevant: " + expr);

	    // NOT
	    if (expr.isNot()) {
		return isRelevant(!sense, label, pred, path);
	    }

	    // AND
	    else if (expr.isAnd()) {
		List children = pred.getChildren();
		assert(children.size() == expr.arity());

		// LBLNEG
		if (!sense) {
		    Iterator i = children.iterator();
		    while (i.hasNext()) {
			Cvc3Pred childi = (Cvc3Pred) i.next();
			// child is the first false conjunct and thus relevant
			if (childi.getExpr().equals(child.getExpr())) {
			    assert(isFalse(childi.getExpr()));
			    return isRelevant(sense, label, pred, path);
			}
			// previous conjunct is false, thus child not relevant
			else if (isFalse(childi.getExpr())) {
			    Cvc3Prover.print("isRelevant: NO");
			    return false;
			}
		    }
		    assert(false);
		}
		// all conjuncts are true and relevant
		else {
		    return
			isTrue(pred.getExpr())
			&&
			isRelevant(sense, label, pred, path);
		}
	    }

	    // OR
	    else if (expr.isOr()) {
		List children = pred.getChildren();
		if (children.size() != expr.arity()) {
		    System.out.println(expr);
		    System.out.println(pred);
		}
		assert(children.size() == expr.arity());

		// LBLPOS
		if (sense) {
		    Iterator i = children.iterator();
		    while (i.hasNext()) {
			Cvc3Pred childi = (Cvc3Pred) i.next();
			// child is the first true disjunct and thus relevant
			if (childi.getExpr().equals(child.getExpr())) {
			    assert(isTrue(childi.getExpr()));
			    return isRelevant(sense, label, pred, path);
			}
			// previous conjunct is true, thus child not relevant
			else if (isTrue(childi.getExpr())) {
			    Cvc3Prover.print("isRelevant: NO");
			    return false;
			}
		    }
		    assert(false);
		}
		// all disjuncts are false and relevant
		else {
		    return
			isFalse(pred.getExpr())
			&&
			isRelevant(sense, label, pred, path);
		}
	    }

	    else {
		Cvc3Prover.print("isRelevant: NOT IMPLEMENTED");
		assert(false);	
		return false;
	    }
	}

	assert(false);
	return false;
    }

    // checks if the label is relevant in the given context
    protected boolean isRelevant(Cvc3Label label) throws cvc3.Cvc3Exception {
	Cvc3Prover.print("isRelevant LABEL: " + label.getName());
	List paths = (List) getPaths().get(label);

	if (paths == null || paths.size() == 0) {
	    Cvc3Prover.print("Label does not occur in formula");
	    assert(false);
	    return false;
	}

	Iterator i = paths.iterator();
	while (i.hasNext()) {
	    Iterator path = ((List) i.next()).iterator();
	    Cvc3Pred pred = (Cvc3Pred) path.next();
	    Expr query = pred.getExpr();

	    if (
		((label.isPos() && isTrue(query))
		 ||
		 (!label.isPos() && isFalse(query)))
		&&
		isRelevant(label.isPos(), label, pred, path)) {
		Cvc3Prover.print("isRelevant: YES");
		return true;
	    }
	}

	Cvc3Prover.print("isRelevant: NO");
	return false;
    }

    // returns the names of the labels which are relevant
    // in the current (satisfiable) context
    public List relevantLabels() throws cvc3.Cvc3Exception {
	// the relevant labels
	// consider each label name as relevant only once,
	// even if it labels several relevant expressions
	HashSet relevant = new HashSet();

	Iterator i = getPaths().keySet().iterator();
	while (i.hasNext()) {
	    Cvc3Label label = (Cvc3Label) i.next();
	    if (!relevant.contains(label.getName()) && isRelevant(label)) {
		relevant.add(label.getName());
	    }
	}

	return new ArrayList(relevant);
    }
}
