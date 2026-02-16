/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.io.*;

import java.util.Enumeration;
import java.util.zip.*;

import javafe.genericfile.*;


/**
 * A ZipTree is a Tree that mirrors the contents of a zipfile;
 * the constructor takes in a pathname and returns a tree
 * representing the filesystem contained in that zipfile.<p>
 *
 * ZipTree works by scanning the zipfile at object creation time,
 * building up a hierarchical list of all the files in the zipfile.
 * Later modifications to the zipfile will not be reflected in the
 * ZipTree.<p>
 *
 * The data field of every (sub)node in a ZipTree contains a
 * non-null ZipGenericFile representing the file it mirrors (or would
 * mirror if a corresponding entry existed) in the underlying zipfile.<p>
 *
 * 
 * Note: ZipTree interior or root nodes (and only interior or root
 * nodes) may contain ZipGenericFiles that do not have a corresponding
 * zip entry because there may be no zip entries for those directories.
 * E.g., the zipfile might contain a file for X/Y but not for X<p>
 *
 * We use "./" as the ZipGenericFile name to represent the missing
 * root directory (no ZipTree actually has a root directory as far as
 * we can tell).
 *
 * This gives the right isDirectory behavior (since it ends with a
 * "/"), but the wrong local name (".").  This is the best we can do,
 * though, since the other alternative ("") gives the reverse
 * behavior, which is worse for us since we rely on the isDirectory
 * bit and only use local names for non-directories.
 */

class ZipTree extends ExtTree {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** The zipfile we are a snapshot of */
    //@ invariant zip != null;
    protected ZipFile zip;


    /**
     * Initialize a node's data field to a ZipGenericFile that
     * represents the file that it would correspond to if the tree it
     * belongs to mirrored zip.<p>
     */
    //@ requires node != null && zip != null;
    protected static void missingEntry(Tree node, ZipFile zip) {
	String name = node.getLabel();
	if (name==null)
	    name = "./";

	for (Tree ptr = node.getParent(); ptr != null; ptr=ptr.getParent()) {
	    if (ptr.getLabel() != null)
		name = ptr.getLabel() + "/" + name;
	}
	
	node.data = new ZipGenericFile(zip, new ZipEntry(name));
    }


    /**
     * Create a ZipTree to mirror a zipfile's contents.<p>
     *
     * May throw an IOException (e.g., file doesn't exist) or a
     * ZipException (e.g., file is not a properly formatted
     * zipfile).
     **/
    public ZipTree(/*@ non_null @*/ File zipfile) throws IOException, ZipException {
	super(null);

	// Open the file as a ZipFile:
	zip = new ZipFile(zipfile);

	// Then initialize us using the ZipFile's table of contents:
	loadZipData();
    }

    /**
     * Create a tree of ZipEntry's from the pathnames of the ZipEntry's
     * in zip
     */
    private void loadZipData() {
	// data may be overwritten later if an entry exists for us:
	missingEntry(this, zip);		

	for (Enumeration E=zip.entries(); E.hasMoreElements(); ) {
	    ZipEntry next = (ZipEntry)E.nextElement();
	    addZipEntry(next);
	}
    }

    /** Add a ZipEntry to this tree according to its pathname: */
    //@ requires z != null;
    private void addZipEntry(ZipEntry z) {
	String pathname = z.getName();

//	System.out.println("entry: " + pathname + (z.isDirectory()?" D":""));

	//
	// Note: ZipEntry's always use '/' as their separator!
	//

	// Strip off any leading file separators:
	while (pathname.length()>0 && pathname.charAt(0)=='/')
	    pathname = pathname.substring(1);

	// Strip off any trailing file separators:
	while (pathname.length()>0 &&
		pathname.charAt(pathname.length()-1)=='/')
	    pathname = pathname.substring(0,pathname.length()-1);

	// Locate/create a node corresponding to path:
	String[] path = StringUtil.parseList(pathname, '/');
	Tree fileNode = addPath(path);

	fileNode.data = new ZipGenericFile(zip, z);

	// Initialize any new interior nodes as missing entries;
	for (; fileNode != null; fileNode = fileNode.getParent())
	    if (fileNode.data==null)
		missingEntry(fileNode, zip);
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /** A simple test driver */
    //@ requires args != null;
    /*@ requires (\forall int i; (0<=i && i<=args.length)
		==> args[i] != null); */
    public static void main(String[] args) {
	if (args.length != 1) {
	    System.out.println("ZipTree: usage <zipfile>");
	    return;
	}

	/*
	 * Create a new ZipTree from the file args[0] and then 
	 * print it out.
	 */
	try {
	    Tree T = new ZipTree(new File(args[0]));
//	    T.print("");

	    // Enumerate its subtrees:
	    for (Enumeration E = new TreeWalker(T);
		E.hasMoreElements();) {
	        Tree subtree = (Tree)E.nextElement();
		if (subtree.getChildrenCount() == 0) {
		    System.out.println("  " + subtree.getLabel());
		} else
		    System.out.println(subtree.getQualifiedName(".") + ":");

		/*
		System.out.println("[local name = "
			    +((GenericFile)subtree.data).getLocalName()+"]");
		System.out.println("[entry name = "
			    +((GenericFile)subtree.data).getHumanName()+"]");
		*/
	    }

	} catch (Exception E) {
	    System.out.println("Caught " + E);
	};

    }
}
