package escjava.vcGeneration.coq.visitor.simplifiers;

import java.io.IOException;
import java.util.Iterator;

import escjava.vcGeneration.TBoolImplies;
import escjava.vcGeneration.TBoolOr;
import escjava.vcGeneration.TDisplay;
import escjava.vcGeneration.TFunction;
import escjava.vcGeneration.TNode;
import escjava.vcGeneration.TRoot;


/**
 * Split the proof using the <code>or</code>s to find out the cutting points.
 * It does not duplicate the whole proof, it only duplicate the main
 * implies path.
 * After this visitor the {@link TNode#parent} field is inconsistant
 * apart from the main 
 * <code>TRoot -> TBoolImplies ... -> TBoolImplies -> False</code> path.
 * The split is done recursively over the ors but for a finite number of time
 * to avoid explosion of the cases. This limmit is given by the 
 * <code>morbidity</code> of the splitter instance.
 * @author J. Charles
 */
public class TProofSplitter extends ATSimplifier {
	
	/** tells how much the splitter shall go recursively over the <code>or</code>s */
	private int morbidity;
	
	/** when the degeneration become unviable: equals 5 */
	private final static int MORBIDITY_THRESHOLD = 5;
	
	
	private TRoot currentRoot;
	
	/**
	 * Construct a new splitter object with a given morbidity of 0.
	 */
	public TProofSplitter() {
		morbidity = 0;
	}
	/**
	 * Construct a new splitter object with a given morbidity.
	 * @param currentMorbidity the morbidity of the new instance
	 */
	public TProofSplitter(int currentMorbidity) {
		morbidity = currentMorbidity;
	}


	/**
	 * Visit the children of the root node if the morbidity is
	 * different from 0.
	 */
	/*
	 * (non-Javadoc)
	 * @see escjava.vcGeneration.TVisitor#visitTRoot(escjava.vcGeneration.TRoot)
	 */
	public void visitTRoot(/*@ non_null @*/ TRoot n) {
		
		if(morbidity == MORBIDITY_THRESHOLD)
			return;
		currentRoot = n;
		super.visitTRoot(n);
	}
	
	
	/**
	 * Visit an implication.
	 * If the implication's first member is a or, it generates
	 * several proof obligations, one for each children of the or.
	 * These proof obligations can be obtained with {@link #getListTerms()}.
	 */
	/*
	 * (non-Javadoc)
	 * @see escjava.vcGeneration.TVisitor#visitTBoolImplies(escjava.vcGeneration.TBoolImplies)
	 */
	public void visitTBoolImplies(/*@ non_null @*/ TBoolImplies n) {
		if(n.sons.size() != 2) {
			  TDisplay.err(n.sons.size() +"sons, that's suspicious");
		}
		
		TNode noddor = (TNode)n.sons.get(0);
		TNode noddy = (TNode)n.sons.get(1);
		
		if(noddor instanceof TBoolOr) {
			TBoolOr tbo = (TBoolOr) noddor;
			Iterator iter = tbo.sons.iterator();
			if (!(n.parent.parent == null)) {
				currentRoot.sons.remove(findRoot(n));
				while(iter.hasNext()) {
					TNode nod = cloneAndReplace((TBoolImplies)n.parent,(TNode)iter.next(), noddy);
					
					TProofSplitter tps = new TProofSplitter(this.morbidity + 1);
					TRoot root = new TRoot();
					root.sons.add(nod);
					root.accept(tps);
					if(tps.currentRoot == null || tps.currentRoot.sons.size() == 0) {
						currentRoot.sons.add(nod);
					}
					else {
						currentRoot.sons.addAll(tps.currentRoot.sons);
					}
				}
				return;
			}
		}
		
		noddy.accept(this);
	}
	
	
	/**
	 * Shallow clone the parents term, the children term
	 * and glue it with node through an implication.
	 * Like: <code>parent -> (node -> noddy)</code>. It
	 * returns the root of the generated term.
	 * @param parent the parent to clone and glue
	 * @param node the node to glue
	 * @param noddy the children to clone and glue
	 * @return returns the root node of the clone
	 */
	private TNode cloneAndReplace(TBoolImplies parent, TNode node, TNode noddy) {
		TNode noddyclone = cloneChildren(noddy);
		TBoolImplies tbi =new TBoolImplies();
		addSon(tbi, 0, node);
		addSon(tbi, 1, noddyclone);
		TFunction parentclone = cloneParents(parent);
		addSon(parentclone, 1, tbi);
		TFunction tr = findRoot(parentclone);
		return tr;
	}

	/**
	 * Clone a term and its parents.
	 * @param par the term to clone
	 * @return the clone term
	 */
	private TFunction cloneParents(TBoolImplies par) {
		TBoolImplies res = new TBoolImplies();
		TBoolImplies oldtmp = res;
		TNode parent = par;
		TBoolImplies tmp = null;
		while (parent.parent != null) {
			tmp = cloneof((TBoolImplies)parent);
			addSon(tmp, 1, oldtmp);
			oldtmp = tmp;
			parent = parent.parent;
		}
		TRoot tr = new TRoot();
		tr.type = parent.type;
		tr.sons.add(tmp);
		tmp.parent = tr;
		return res.parent;
	}

	/**
	 * Shallow clone the children of a node.
	 * It only clone the implications.
	 * @param noddy the node to clone
	 * @return the cloned node
	 */
	private TNode cloneChildren(TNode noddy) {
		if(! (noddy instanceof TBoolImplies))
			return noddy;
		TBoolImplies nod = (TBoolImplies) noddy;
		TBoolImplies newnod = cloneof(nod);
		TNode nnn = cloneChildren((TNode)nod.sons.get(1));
		nnn.parent = newnod;
		return newnod;
	}


}

