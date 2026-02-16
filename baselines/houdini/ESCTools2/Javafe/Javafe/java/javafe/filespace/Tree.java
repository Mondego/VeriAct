/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;


/**
 * A Tree is a n-ary tree with data at each node; the direct children of a
 * node are unordered but distinguished via String labels on the edges
 * to them.<p>
 *
 * All labels must be non-null.<p>
 */

public abstract class Tree {

    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     **************************************************/

    /** Our parent or null if we have no parent (aka, we are a root) */
    //@ spec_public
    private Tree parent = null;

    /**
     * The label on the edge leading to us from our parent, or null if
     * we have no parent.
     */
    //@ invariant (label == null) == (parent == null);
    //@ spec_public
    private String label = null;

    /*
     * Note: the existence of these two fields implies that any given
     * Tree has at most 1 parent and can be reached via at most 1 edge.
     */

    /** The data, if any, associated with this node: */
    public Object data;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** Create a root node: */
    protected Tree(Object data) {
	this.data = data;
    }

    /** Create a non-root node: */
    //@ requires parent != null && label != null;
    protected Tree(Tree parent, String label, Object data) {
	this.parent = parent;
	this.label = label;
	this.data = data;
    }
	

    /***************************************************
     *                                                 *
     * Simple public accessors:			       *
     *                                                 *
     **************************************************/

    /** Return our parent node or null if we have no parent */
    public final Tree getParent() {
	return parent;
    }

    /**
     * Return the label on the edge to us from our parent or null if we
     * have no parent.
     */
    public final String getLabel() {
	return label;
    }


    /***************************************************
     *                                                 *
     * Fetching and counting children:		       *
     *                                                 *
     **************************************************/

    /**
     * An enumeration of this node's direct children.  Each child
     * occurs exactly once in the enumeration.  The order is
     * unspecified and may differ from call to call.<p>
     *
     * Note: The Objects returned by the resulting enumeration's
     * nextElement method are guaranteed to be of type Tree,
     * non-null, and have non-null labels.<p>
     */
    //@ ensures \result != null;
    //@ ensures !\result.returnsNull;
    //@ ensures \result.elementType == \type(Tree);
    public abstract Enumeration children();


    /**
     * Fetch our direct child along the edge labelled label.  Iff there
     * is no such child, return null.
     */
    //@ requires label != null;
    public Tree getChild(String label) {
	/*
	 * Stupid & slow default implementation using children()
	 * enumeration:
	 */
	Enumeration C = children();

	while (C.hasMoreElements()) {
	    Tree next = (Tree)C.nextElement();

	    if (label.equals(next.label))
		return next;
        }

	return null;
    }


    /** Return true iff we have no direct children */
    public boolean isLeaf() {
	return (!children().hasMoreElements());
    }


    /** Return a count of how many direct children we have: */
    //@ ensures \result >= 0;
    public int getChildrenCount() {
	/*
	 * Stupid & slow default implementation using children()
	 * enumeration:
	 */
	int count = 0;
	for (Enumeration C = children(); C.hasMoreElements(); C.nextElement())
	    count++;

	return count;

    }


    /***************************************************
     *                                                 *
     * Utility functions:			       *
     *                                                 *
     **************************************************/

    /**
     * The same as getLabel, except that it returns "" instead of null
     * for the top node.
     */
    //@ ensures \result != null;
    public final String getSimpleName() {
	if (label == null)
	    return "";
	else
	    return label;
    }

    /**
     * Return a fully qualified name for this node using a specified
     * separator String.<p>
     *
     * If the sequences of labels on the edges from the root of the
     * tree this node belongs to til this node is X, Y, ... Z, then
     * this routine returns the string X!Y! ... !Z, where ! is the
     * separator.  Normal usage is to use "." as the separator.<p>
     *
     * Root nodes are thus named "" and their direct children have
     * names of the form X, where X is their label.<p>
     * 
     * For the resulting name to be useful, labels should never contain
     * the separator or be the empty string.<p>
     */
    //@ ensures \result != null;
    public final String getQualifiedName(String separator) {
	if (parent == null)
	     return "";

	if (parent.parent == null)
	    return label;

	return parent.getQualifiedName(separator) + separator + label;
    }

    /** Return the root node for the tree we belong to.  */
    public final Tree getRootNode() {
	if (parent == null)
	    return this;

	return parent.getRootNode();
    }

    /**
     * Return the child with a given partially qualified name or null
     * if no such node exists; if this node is X.Y and name is Z!W,
     * where ! is the specified separator, then this routine attempts
     * to find the child with fully qualified name X.Y.Z.W.<p>
     *
     * See getQualifiedName for more information on qualified names.
     * Name must be non-null.<p>
     */
    //@ requires name != null;
    public final Tree getQualifiedChild(String name, char separator) {
	String[] path = StringUtil.parseList(name, separator);

	Tree currentNode = this;
	for (int i=0; i<path.length; i++) {
	    currentNode = currentNode.getChild(path[i]);
	    if (currentNode == null)
		return null;
        }

	return currentNode;
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /**
     * Print out on System.out a human-readable representation of this
     * tree.<p>
     *
     * Included is our fully qualified name, our data, and information
     * about each of our children.<p>
     */
    public void print(String prefix) {
	System.out.print(prefix + "Node '" + getQualifiedName(".") + "'");
	printDetails(prefix + "  ");
	if (isLeaf())
	    System.out.println();
	else {
	    System.out.println(":");
	    for (Enumeration E = children(); E.hasMoreElements(); )
		((Tree)E.nextElement()).print(prefix + "  ");
        }
    }

    public void printDetails(String prefix) {
	System.out.print(", data=" + data);
    }
}
