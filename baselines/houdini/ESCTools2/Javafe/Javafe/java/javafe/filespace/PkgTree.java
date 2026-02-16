/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;
import java.io.IOException;

import javafe.genericfile.*;


/**
 * A PkgTree is a filtered representation of a filespace {@link Tree}
 * (cf {@link PathComponent}) where some files and directories that
 * are clearly not part of the Java namespace have been filtered out;
 * the remaining nodes can be divided into two categories: (a)
 * (usually interior) nodes that correspond to potential Java
 * packages, and (b) exterior nodes that correspond to files that
 * reside in one of the potential Java packages and that have an
 * extension (e.g., .java).<p>
 *
 * A function, {@link #isPackage(Tree)}, is provided to distinguish
 * the two categories.  A convenience function, {@link
 * #packages(Tree)}, is provided to enumerate all the potential Java
 * packages in a PkgTree.<p>
 *
 * {@link #isPackage(Tree)} depends only on a node's label; this
 * ensures that a {@link UnionTree} of several PkgTree's never
 * combines package and non-package nodes into a single node.  (This
 * is why (b) excludes files without extensions.)  Accordingly,
 * PkgTree's accessors and enumerators can also be used on {@link
 * UnionTree}s of PkgTrees.<p>
 *
 * This module is meant to do a reasonable job of identifying potential
 * packages.  It is not 100% accurate, however, erring on the side of
 * admitting too many packages.  For example, it does not attempt to
 * disallow package names containing Java keywords.<p>
 */

public class PkgTree extends PreloadedTree {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/


    /** The non-null filespace Tree we are filtering */
    //@ invariant underlyingTree != null;
    protected Tree underlyingTree;


    /**
     * Filter a non-null filespace Tree, leaving potential Java
     * packages and files.
     */
    //@ requires underlyingTree != null;
    public PkgTree(Tree underlyingTree) {
	super(underlyingTree.data);
	this.underlyingTree = underlyingTree;
    }

    /**
     * Create a non-root node.  underlyingTree must be a non-null
     * filespace Tree.
     */
    //@ requires underlyingTree != null;
    //@ requires parent != null && label != null;
    protected PkgTree(Tree parent, String label, Tree underlyingTree) {
	super(parent, label, underlyingTree.data);
	this.underlyingTree = underlyingTree;
    }


    /***************************************************
     *                                                 *
     * Deciding what nodes to filter out:	       *
     *                                                 *
     **************************************************/

    /*
     * Status codes returned by getStatus:
     */

    /** ignore the node and its children */
    protected static final int IGNORE = 0;

    /** include the node but not its children */
    protected static final int INCLUDE_NODE = 1;

    /** include the node and its children */
    protected static final int INCLUDE_TREE = 2;


    /**
     * Decide what to do with a node of the underlying filespace, returning
     * one of the following codes: IGNORE, INCLUDE_NODE, or INCLUDE_TREE.
     */
    protected static int getStatus(/*@ non_null @*/ Tree node) {
	String label = node.getSimpleName();
	String extension = Extension.getExtension(label);

	/*
         * Ignore all files beginning with "META-" (used by the .jar
	 * format for meta-data):
         */
	if (label.startsWith("META-"))
	    return IGNORE;

	/*
	 * For now, potential packages are directories without
	 * extensions:
	 */
	if (((GenericFile)node.data).isDirectory()	//@ nowarn Cast,Null;
		&& extension.equals(""))
	    return INCLUDE_TREE;

	/* Directories in jars do not appear as directories */
	if ((node.data instanceof ZipGenericFile)
		&& extension.equals(""))
	    return INCLUDE_TREE;

	/*
	 * Non-package files include only those with extensions.
	 *
	 * Note that directories with extensions are treated here as
	 * if they were ordinary files (i.e., ignore their children
	 * and treat them as non-packages).  This is necessary to
         * minic javac's behavior and to allow the checker to complain
	 * about such files.
         */
	if (extension.equals(""))
	    return IGNORE;
	else
	    return INCLUDE_NODE;
    }

    /***************************************************
     *                                                 *
     * Loading the edges map:			       *
     *                                                 *
     **************************************************/

    /** Load the edges map for use.  */
    protected void loadEdges() {
	if (getStatus(underlyingTree) != INCLUDE_TREE)
	    return;	// We are ignoring this tree or its children

	for (Enumeration E=underlyingTree.children(); E.hasMoreElements(); ) {
	    Tree child = (Tree)E.nextElement();
	    if (getStatus(child) != IGNORE) {
		String label = child.getLabel();
		//@ assume label != null;
		edges.put(label, new PkgTree(this, label, child));
	    }
	}
    }


    /***************************************************
     *                                                 *
     * Accessors:				       *
     *                                                 *
     **************************************************/

    /**
     * Is a node of a PkgTree (or a union of PkgTree's) a potential
     * Java package?
     */
    public static boolean isPackage(/*@ non_null @*/ Tree node) {
	return Extension.getExtension(node.getSimpleName()).equals("");
    }

    /** The name to use for root packages */
    public static String rootPackageName = "<the unnamed package>";

    /**
     * Return the human-readable name of a package.  Uses rootPackageName
     * as the name of root packages.<p>
     *
     * Note: the resulting name will only make sense if node is a
     * package.<p>
     */
    public static String getPackageName(/*@ non_null @*/ Tree node) {
	if (node.getParent() == null)
	    return rootPackageName;

	return node.getQualifiedName(".");
    }


    /***************************************************
     *                                                 *
     * Enumerators:				       *
     *                                                 *
     **************************************************/

    /**
     * Enumerate all the potential packages of a PkgTree (or a union of
     * PkgTree's) in depth-first pre-order using lexical ordering on
     * siblings (cf. TreeWalker).
     */
    //@ ensures !\result.returnsNull;
    //@ ensures \result.elementType == \type(Tree);
    public static /*@ non_null @*/ Enumeration packages(/*@ non_null @*/ Tree node) {
	Enumeration allNodes = new TreeWalker(node);

	return new FilterEnum(allNodes, new PkgTree_PackagesOnly());
    }

    /**
     * Enumerate all the components of package P with extension E in
     * sorted order (of labels).<p>
     *
     * For a PkgTree (or a union of PkgTrees), if E is "", then all
     * direct potential subpackages will be selected.  Otherwise, only
     * non-subpackages will be selected.<p>
     */
    //@ ensures !\result.returnsNull;
    //@ ensures \result.elementType == \type(Tree);
    public static /*@ non_null @*/ Enumeration components(/*@ non_null @*/ Tree P, 
                                                          /*@ non_null @*/ String E) {
	Enumeration allComponents = TreeWalker.sortedChildren(P);

	return new FilterEnum(allComponents,
			new PkgTree_MatchesExtension(E));
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /** Extend printDetails to include our isPackage status */
    public void printDetails(String prefix) {
	super.printDetails(prefix);
	if (isPackage(this))
	    System.out.print(" [P]");
    }

    /** A simple test driver */
    //@ requires (\forall int i; (0 <= i && i < args.length) ==> args[i] != null);
    public static void main(/*@ non_null @*/ String[] args) throws IOException {
	if (args.length != 1 && args.length != 3) {
	    System.out.println("PkgTree: usage <path component>"
		+ " [<package name> <extension>]");
	    return;
	}

	/*
	 * Convert the path component to a filespace then filter it via
	 * PkgTree.
	 */
	Tree T = PathComponent.open(args[0], false);
	T = new PkgTree(T);

	// If a package name is provided, list all its components with
	// the given extension:
	if (args.length==3) {
	    Tree P = T.getQualifiedChild(args[1], '.');
	    if (P==null) {
		System.out.println("No such package: " + args[1]);
		return;
	    };

	    System.out.println(getPackageName(P) + ":");
	    Enumeration E = components(P, args[2]);
	    while (E.hasMoreElements()) {
		Tree next = (Tree)E.nextElement();
		System.out.println(next.getSimpleName());
	    }
	    return;
        };

	T.print("");
	System.out.println();

	// Enumerate all the potential packages also to test packages():
	Enumeration E = packages(T);
	while (E.hasMoreElements()) {
	    Tree next = (Tree)E.nextElement();
	    System.out.println(getPackageName(next));
	}
    }
}


/**
 * A filter for accepting only packages:
 *
 * This filter is for the use of the PkgTree class only; if inner
 * classes were available, it would be expressed as an anonymous class.
 */
class PkgTree_PackagesOnly implements Filter {
    //@ invariant acceptedType == \type(Tree);

    public PkgTree_PackagesOnly() {
	//@ set acceptedType = \type(Tree);
    }

    public boolean accept(Object node) {
	return PkgTree.isPackage((Tree)node);
    }
}


/**
 * A filter for accepting only node's with a particular extension:
 *
 * This filter is for the use of the PkgTree class only; if inner
 * classes were available, it would be expressed as an anonymous class.
 */
class PkgTree_MatchesExtension implements Filter {

    //@ invariant acceptedType == \type(Tree);

    //@ invariant targetExtension != null;
    String targetExtension;

    //@ requires targetExtension != null;
    PkgTree_MatchesExtension(String targetExtension) {
	this.targetExtension = targetExtension;
	//@ set acceptedType = \type(Tree);
    }

    public boolean accept(Object node) {
	return Extension.hasExtension(
	    ((Tree)node).getSimpleName(), targetExtension);
    }

}
