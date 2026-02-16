/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.*;
import java.io.File;


/**
 * This class provides a way to enumerate all the nodes of a
 * given Tree in depth-first pre-order using lexical ordering on
 * siblings.  I.e., X preceeds X.A, which in turn preceeds X.B.
 * The first node in the list will be the root note of the Tree.
 * It also allows enumerating a Tree's direct children in sorted order
 * (based on their labels).<p>
 *
 * Guarantee: Returned elements are always non-null Trees.<p>
 */

public final class TreeWalker extends LookAheadEnum {

    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     **************************************************/

    //@ invariant elementType == \type(Tree);
    //@ invariant !returnsNull;


    /**
     * The remaining children we have yet to start processing:
     */
    //@ invariant remainingChildren != null;
    //@ invariant !remainingChildren.returnsNull;
    //@ invariant remainingChildren.elementType == \type(Tree);
    protected Enumeration remainingChildren;

    /**
     * The remaining nodes from the child we are currently
     * processing:
     */
    //@ invariant remainingNodes != null;
    //@ invariant !remainingNodes.returnsNull;
    //@ invariant remainingNodes.elementType == \type(Tree);
    protected Enumeration remainingNodes;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * From a Tree create an enumeration that enumerates
     * all of the Tree's nodes (including the root node first).
     * The nodes are produced in depth-first lexical pre-order.
     */
    //@ requires T != null;
    public TreeWalker(Tree T) {
	// First element is the tree itself:
	super(T);

	remainingChildren = sortedChildren(T);
	remainingNodes    = new EmptyEnum();
	//@ set remainingNodes.elementType = \type(Tree);

        //@ set elementType = \type(Tree);
        //@ set returnsNull = false;
    }


    /***************************************************
     *                                                 *
     * Calculating the next element:		       *
     *                                                 *
     **************************************************/

    /*
     * This returns the next element in the enumeration or null if there
     * are no more nodes left.
     */
    //@ also
    //@ assignable remainingNodes;
    protected Object calcNextElement() {
        for (;;) {
	    // First exhaust the nodes from the current child:
	    if (remainingNodes.hasMoreElements())
		return remainingNodes.nextElement();

	    // Advance to next child:
	    if (!remainingChildren.hasMoreElements())
		return null;	// no nodes left...
	    Tree nextChild = (Tree)remainingChildren.nextElement();

	    // Process all of its nodes:
	    remainingNodes = new TreeWalker(nextChild);
        }
    }


    /***************************************************
     *                                                 *
     * Enumerating children in order:		       *
     *                                                 *
     **************************************************/

    /**
     * Enumerate a Tree's direct children in sorted order (of labels).<p>
     *
     * Guarantee: The resulting enumeration never yields null as an
     * element.<p>
     */
    //@ requires T != null;
    //@ ensures \result != null;
    //@ ensures !\result.returnsNull;
    //@ ensures \result.elementType == \type(Tree);
    //@ ensures ((Object)\result).owner == null;
    public static Enumeration sortedChildren(Tree T) {
	return new TreeWalker_ArrayEnum(getSortedChildren(T));
    }


    /** Return a sorted list of a Tree's direct children: */
    //@ requires T != null;
    //@ ensures \result != null;
    //@ ensures \elemtype(\typeof(\result)) == \type(Tree);
    private static Tree[] getSortedChildren(Tree T) {
	Tree[] directChildren = new Tree[T.getChildrenCount()];

	// Copy list of T's direct children into an array:
	int c = 0;
	for (Enumeration C=T.children(); C.hasMoreElements(); ) {
	    directChildren[c++] = (Tree)C.nextElement(); //@nowarn IndexTooBig;
	}
	//@ assume \nonnullelements(directChildren);

	// Sort the array then return it:
	sort(directChildren);
	return directChildren;
    }

    /*
     * A simple, stupid (aka slow) sorting routine.
     *
     * precondition: a[i] is not a root node.
     */
    //@ requires \nonnullelements(a);
    private static void sort(Tree[] a) {
	// Only arrays of length>2 need to be sorted:
	if (a.length<2)
	    return;

	// Bubble sort...
	for (;;) {
	    boolean sorted = true;
	    for (int i=0; i<a.length-1; i++) {
		if (a[i].getLabel().compareTo	//@ nowarn Null;
		        (a[i+1].getLabel())>0) {  //@ nowarn NonNull;
		    Tree tmp = a[i];
		    a[i] = a[i+1];
		    a[i+1] = tmp;
		    sorted = false;
		}
	    }
	    if (sorted)
		return;
	}
    }


    /***************************************************
     *                                                 *
     * Debugging:				       *
     *                                                 *
     **************************************************/

   /** A simple test driver. */
    //@ requires \nonnullelements(args);
   public static void main(String[] args) {
       	if (args.length != 1) {
	    System.out.println("usage: TreeWalker <pathname>");
	    return;
	}
	String treeName = args[0];

	// Find the tree in question:
	Tree top = new FileTree(new File(treeName));

	// Enumerate its subtrees:
	for (Enumeration E = new TreeWalker(top); E.hasMoreElements();) {
	    Tree subtree = (Tree)E.nextElement();
	    System.out.println(subtree.getQualifiedName(".") + ":");
        }
    }
}


/**
 * A Enumeration for enumerating the members of an array of Objects.
 *
 * This filter is for the use of the TreeWalker class only; if inner
 * classes were available, it would be expressed as an anonymous class.
 */
class TreeWalker_ArrayEnum extends LookAheadEnum {

    //@ private invariant list != null;
    //@ private invariant \elemtype(\typeof(list)) == elementType;
    private Object[] list;

    //@ private invariant index+1>=0;
    private int index = -1;


    //@ private normal_behavior
    //@ requires list != null;
    //@ assignable \not_specified;
    //@ ensures elementType == \elemtype(\typeof(list));
    TreeWalker_ArrayEnum(Object[] list) {
	this.list = list;

	//@ set elementType = \elemtype(\typeof(list));
    }

    //@ also private normal_behavior
    //@ assignable index;
    public Object calcNextElement() {
	if (++index>=list.length)
	    return null;
	else
	    return list[index];
    }
}
