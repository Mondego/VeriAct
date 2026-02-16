package escjava.vcGeneration.coq.visitor.simplifiers;

import java.io.IOException;
import java.util.Iterator;

import escjava.vcGeneration.TBoolAnd;
import escjava.vcGeneration.TBoolImplies;
import escjava.vcGeneration.TDisplay;
import escjava.vcGeneration.TNode;

/**
 * Transform as much as possible ands of the proof into
 * implies. Useful not to have to split the proof obligation
 * afterward.
 * @author J. Charles
 */
public class TAndRemover  extends ATSimplifier{

	/**
	 * If the first son of the implication is an and,
	 * it tries to turn it into implications.
	 */
	/*
	 * (non-Javadoc)
	 * @see escjava.vcGeneration.TVisitor#visitTBoolImplies(escjava.vcGeneration.TBoolImplies)
	 */
	public void visitTBoolImplies(/*@non_null@*/ TBoolImplies n) {
		if(n.sons.size() != 2)
			  TDisplay.err(n.sons.size() +"sons, that's suspicious");

		TNode noddy = (TNode)n.sons.get(0);
		if(noddy instanceof TBoolAnd) {
			TBoolAnd andy = (TBoolAnd)noddy;
			and2imply(andy, n);
		}
		((TNode)n.sons.get(1)).accept(this);
		
	}
	
	/**
	 * Turn a boolean and, which was the child of the
	 * implication into a serie of implications and 
	 * glue it back to the proof obligation.
	 * @param andy the and to split
	 * @param n the node to glue everything up
	 */
	private void and2imply(TBoolAnd andy, TBoolImplies n) {
		Iterator iter = andy.sons.iterator();
		TNode next = (TNode)n.sons.get(1);
		TBoolImplies tmp = n;
		while(iter.hasNext()) {
			addSon(n, 0, (TNode)iter.next());
			tmp = n;
			n = new TBoolImplies();
			
			addSon(tmp, 1, n);
		}
		addSon(tmp, 1,  next);
	}


}
