/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.io.*;

import javafe.genericfile.*;


/**
 * A FileTree is a Tree that mirrors the contents of a disk filesystem;
 * the constructor takes in a directory and returns a tree representing
 * the filesystem rooted at that directory.<p>
 *
 * FileTree works by scanning directories the first time clients
 * ask for information about that part of the tree.  If the filesystem
 * is changed afterwards, the changes will not be visible in the
 * FileTree.<p>
 *
 * The data field of every (sub)node in a FileTree contains a
 * non-null NormalGenericFile representing the file it mirrors on disk.<p>
 */

class FileTree extends PreloadedTree {

    /** The directory we are a snapshot of */
    protected File dir;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** Create a root node: */
    //@ requires dir != null;
    public FileTree(File dir) {
	super(new NormalGenericFile(dir));
	this.dir = dir;
    }

    /** Create a non-root node: */
    //@ requires dir != null;
    //@ requires parent != null && label != null;
    protected FileTree(Tree parent, String label, File dir) {
	super(parent, label, new NormalGenericFile(dir));
	this.dir = dir;
    }


    /***************************************************
     *                                                 *
     * Loading the edges map:			       *
     *                                                 *
     **************************************************/

    /** Load the edges map for use.  */
    protected void loadEdges() {
	/*
	 * If our directory is null or not an actual existing directory
	 * on disk, we have no children:
	 */
	if (dir == null || !dir.isDirectory())
	    return;

	/*
	 * Get all the filenames in our directory.  If the directory is
	 * unreadable (permission errors, etc.), we have no children:
	 */
	String[] filenames = dir.list();
	if (filenames == null)
	    return;

	/*
	 * Add all of them as either sub-FileTree's (if a directory) or
	 * LeafTree's (if not a directory):
	 */
	for (int i=0; i<filenames.length; i++) {
	   if (filenames[i]==null)
		continue;

	    File file = new File(dir, filenames[i]);

	    Tree subtree;
	    if (file.isDirectory())
		subtree = new FileTree(this, filenames[i], file);
	    else
		subtree = new LeafTree(this, filenames[i],
					new NormalGenericFile(file));

	    edges.put(filenames[i], subtree);
        }
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /** A simple test driver */
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
	if (args.length>1) {
	    System.out.println("FileTree: usage [<directory>]");
	    return;
	}
	String dirName = ".";		// Default directory = .
	if (args.length==1)
	    dirName = args[0];

	/*
	 * Create a new FileTree rooted at the given directory name then
	 * print it out.
	 */
	Tree T = new FileTree(new File(dirName));
	T.print("");
    }
}
