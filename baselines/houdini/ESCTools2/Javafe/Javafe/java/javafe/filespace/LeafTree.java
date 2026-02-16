/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;


/**
 * A LeafTree is a degenerate form of Tree that never contains
 * children.  It is used for the leaf nodes of a Tree.
 */

class LeafTree extends Tree {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** Create a root node: */
    public LeafTree(Object data) {
	super(data);
    }

    /** Create a non-root node: */
    //@ requires parent != null && label != null;
    /* package */ LeafTree(Tree parent, String label, Object data) {
	super(parent, label, data);
    }


    /***************************************************
     *                                                 *
     * Fetching and counting children:		       *
     *                                                 *
     **************************************************/

    /*
     * Hardwire the fact that we never have children.  New definitions
     * of isLeaf and getChildrenCount are provided for efficiency
     * reasons.
     */

    public final Enumeration children() {
	Enumeration empty = new EmptyEnum();
	//@ set empty.elementType = \type(Tree);

	return empty;
    }

    public final boolean isLeaf() {
	return true;
    }

    public final int getChildrenCount() {
	return 0;
    }
}
