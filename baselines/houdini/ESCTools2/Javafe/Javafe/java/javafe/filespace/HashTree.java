/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;
import java.util.Hashtable;


/**
 * A HashTree is a Tree that uses a Hashtable to store the map between
 * labels and its direct children.<p>
 *
 * This abstract class leaves how the Hashtable should be filled up to its
 * subclasses.<p>
 */

abstract class HashTree extends Tree {

    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     **************************************************/

    /**
     * The mapping between our outgoing edge's labels and the subTrees
     * they point to.  No entry for a label means no edge with that
     * label exists.<p>
     *
     * Invariant: all elements of edges are Trees and all keys are
     * Strings.<p>
     */
    //+@ invariant edges.keyType == \type(String);
    //+@ invariant edges.elementType == \type(Tree);
    protected /*@ non_null @*/ Hashtable edges = new Hashtable(5);


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** Create a root node: */
    public HashTree(Object data) {
	super(data);

	//+@ set edges.keyType = \type(String);
	//+@ set edges.elementType = \type(Tree);
    }

    /** Create a non-root node: */
    protected HashTree(/*@ non_null @*/ Tree parent, 
                       /*@ non_null @*/ String label, 
                       Object data) {
	super(parent, label, data);

	//+@ set edges.keyType = \type(String);
	//+@ set edges.elementType = \type(Tree);
    }


    /***************************************************
     *                                                 *
     * Fetching children:			       *
     *                                                 *
     **************************************************/

    /**
     * An enumeration of this node's direct children.  Each child
     * occurs exactly once in the enumeration.  The order is
     * unspecified and may differ from call to call.<p>
     *
     * Note: The Objects returned by the resulting enumeration's
     * nextElement method are guaranteed to be of type Tree and non-null.<p>
     */
    public Enumeration children() {
	return edges.elements();
    }


    /**
     * Fetch our direct child along the edge labelled label.  Iff there
     * is no such child, return null.
     */
    public Tree getChild(String label) {
	return (Tree)edges.get(label);
    }
}
