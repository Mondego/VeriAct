/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


/**
 * A ExtTree is a HashTree that starts out as just a root node, but may
 * be extended at any time by (recursively) adding children.<p>
 *
 * Edges and nodes cannot be deleted once added, however.<p>
 */

class ExtTree extends HashTree {

    /*
     * Object invariant: All elements in the edges HashTable are
     * ExtTrees.
     */


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** Create a root node: */
    public ExtTree(Object data) {
	super(data);
    }

    /** Create a non-root node: */
    //@ requires parent != null && label != null;
    protected ExtTree(Tree parent, String label, Object data) {
	super(parent, label, data);
    }


    /***************************************************
     *                                                 *
     * Adding children:				       *
     *                                                 *
     **************************************************/

    /**
     * Create a new direct child of us with label label and data newData.
     * The new child will have no direct children.<p>
     *
     * If a child by the name of label already exists, then this
     * routine leaves the tree unchanged.<p>
     *
     * In either case, the (resulting) child with label label is returned.<p>
     */
    //@ requires label != null;
    //@ ensures \result != null;
    public ExtTree addChild(String label, Object newData) {
	/* Handle case where child already exists: */
	ExtTree child = (ExtTree)getChild(label);	//@ nowarn Cast;
	if (child != null) {
	    return child;
	}

	child = new ExtTree(this, label, newData);
	edges.put(label, child);
	return child;
    }


    /**
     * This is an extended version of addChild that takes a path (a
     * list of labels) instead of a single label.  It returns the child
     * located by following path from this node.  It creates any
     * necessary nodes along the way using addChild with null for
     * newData.  Path must be non-null.
     */
    /*@ requires path != null &&
	(\forall int i; (0<=i && i<path.length) ==> path[i] != null); */
    //@ ensures \result != null;
    public ExtTree addPath(String[] path) {
	ExtTree currentNode = this;

	for (int i=0; i<path.length; i++)
	    currentNode = currentNode.addChild(path[i], null);
	
	return currentNode;
    }
}
