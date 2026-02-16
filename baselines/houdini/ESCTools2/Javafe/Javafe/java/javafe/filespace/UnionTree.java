/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;
import java.io.File;


/**
 * A UnionTree is a Tree which represents the union of a series of
 * Tree's.<p>
 *
 * A node exists in a UnionTree iff a corresponding node (i.e., same
 * fully qualified name) exists in any of the underlying Trees.  Its
 * data field is copied at creation time from the first such
 * corresponding node (i.e., the one whose Tree is first in the list
 * of underlying Trees).<p>
 *
 * Exception: if the underlying list contains 0 Trees, then the
 * UnionTree contains exactly 1 node, the root node, which will have
 * data equal to null.<p>
 *
 * The other corresponding nodes of a node X can be accessed by calling
 * X.duplicates(), which returns a list of all the corresponding nodes
 * (including the first one) in the same order as the original input
 * list of Trees.<p>
 * 
 * The behavior of a UnionTree is undefined if the underlying Trees it
 * depends on are altered after it has been created.<p>
 */

public class UnionTree extends PreloadedTree {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * The list of Trees we represent the union of:<p>
     *
     * Invariant: contains no nulls and is non-null.<p>
     */
    //@ invariant \nonnullelements(roots);
    protected Tree[] roots;

    /**
     * Create a new Tree that represents the union of the Trees in
     * roots.<p>
     *
     * roots must be non-null and contain no nulls.<p>
     */
    //@ requires \nonnullelements(roots);
    public UnionTree(Tree[] roots) {
	super(null);

	this.roots = roots;
	if (roots.length>0)
	    data = roots[0].data;
    }

    /**
     * Create a non-root node:<p>
     *
     * roots must be non-null and contain no nulls.<p>
     */
    //@ requires \nonnullelements(roots);
    //@ requires parent != null && label != null;
    protected UnionTree(Tree parent, String label, Tree[] roots) {
	super(parent, label, null);

	this.roots = roots;
	if (roots.length>0)
	    data = roots[0].data;
    }

    /***************************************************
     *                                                 *
     * Accessors:				       *
     *                                                 *
     **************************************************/

    /**
     * Return a list of all the nodes that correspond to this one in
     * the underlying Trees in the same order as the original list of
     * Trees.  This ist is never null and never contains any nulls.
     */
    //@ ensures \nonnullelements(\result);
    public Tree[] duplicates() {
	return roots;
    }

    /**
     * Return the number of nodes corresponding to this one there are
     * in the underlying Trees.
     */
    public int countDuplicates() {
	return roots.length;
    }


    /***************************************************
     *                                                 *
     * Loading the edges map:			       *
     *                                                 *
     **************************************************/

    /** Load the edges map for use.  */
    protected void loadEdges() {
	/*
         * Make sure all our direct children are loaded by trying to
	 * load all the direct edges of each of our roots:
         */
	for (int i=0; i<roots.length; i++)
	    for (Enumeration C=roots[i].children(); C.hasMoreElements(); ) {
		Tree next = (Tree)C.nextElement();
		loadEdge(next.getLabel());		//@ nowarn Pre;
	    }
    }

    /* Load the direct edge labeled label if it is not already loaded */
    //@ requires label != null;
    protected void loadEdge(String label) {
	if (edges.containsKey(label))
	    return;			// Edge already loaded...

	// Get a list and count of the corresponding nodes:
	int count = 0;
	Tree[] directChildren = new Tree[roots.length];
	for (int i=0; i<roots.length; i++) {
	    Tree directChild = roots[i].getChild(label);
	    if (directChild != null)
		directChildren[count++] = directChild;
        }

	// If there are no corresponding nodes, then do nothing;
	// this case should not occur.
	if (count == 0)
	    return;

	// Shrink the resulting array to remove the extra space then use
	// the list of nodes to create a subUnionTree to be installed as
	// the child for this edge.
	Tree[] childRoots = new Tree[count];
	for (int i=0; i<count; i++)
	    childRoots[i] = directChildren[i];

	edges.put(label, new UnionTree(this, label, childRoots));
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    public void printDetails(String prefix) {
	System.out.print(" [" + countDuplicates() + "]");

//	System.out.println();
//	for (int i=0; i<roots.length; i++)
//	    roots[i].print(prefix + "  >> ");
    }

    /** A simple test driver */
    //@ requires args != null;
    /*@ requires (\forall int i; (0<=i && i<args.length)
		==> args[i] != null); */
    public static void main(String[] args) {
	/*
	 * Create a list of FileTree's using the paths we're passed in:
         */
	Tree[] roots = new Tree[args.length];
	for (int i=0; i<args.length; i++)
	    roots[i] = new FileTree(new File(args[i]));

	/*
	 * Create a new UnionTree that is a union of those trees then
	 * print it out.
	 */
	Tree T = new UnionTree(roots);
	T.print("");
    }
}
