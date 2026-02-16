package escjava.vcGeneration.coq.visitor.simplifiers;

import java.io.IOException;

import escjava.vcGeneration.TBoolImplies;
import escjava.vcGeneration.TBoolNot;
import escjava.vcGeneration.TBoolean;
import escjava.vcGeneration.TDisplay;
import escjava.vcGeneration.TNode;

/**
 * Removes the last not and turn it into (drum-roll) <code>implies False</code>!
 * @author J. Charles
 */
public class TNotRemover  extends ATSimplifier{
	/**
	 * When it arrives at the last implies,
	 * if it points to a negated expression, it
	 * is turned into implies false. 
	 * i.e.: <code>(not e)</code> ===> <code>(e -> False)</code> 
	 */
	public void visitTBoolImplies(/*@non_null@*/ TBoolImplies n) {
		if(n.sons.size() != 2)
			  TDisplay.warn(n.sons.size() +"sons, that's suspicious");

		TNode noddy = (TNode)n.sons.get(1);
		
		if(noddy.parent == null) {
			TDisplay.warn("Nodes parents should not be null!");
			noddy.parent = n;
		}
		
		if(noddy instanceof TBoolNot) {
			TBoolNot nonot = (TBoolNot)noddy;
			TBoolImplies tbi = new TBoolImplies();
			addSon(tbi, 0, (TNode)nonot.sons.get(0));
			addSon(tbi, 1, new TBoolean(false));
			addSon(n, 1, tbi);
		}
		else {
			noddy.accept(this);
		}
	}

}
