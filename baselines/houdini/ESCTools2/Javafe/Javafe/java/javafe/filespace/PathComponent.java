/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.io.*;
import java.util.zip.*;
import java.util.Enumeration;

import javafe.genericfile.*;


/**
 * This module encapsulates how to convert from a Java path-component
 * name to the hierarchical filespace it denotes.<p>
 *
 * Filespaces are represented by Trees whose node's data fields contain
 * (non-null) GenericFiles that represent the corresponding files.
 * E.g., the node at X.Y.Z represents a file with pathname ./X/Y/Z in
 * the hierarchy of the path component.<p>
 *
 * Changes made to the underlying file system may or may not be
 * reflected in the returned filespaces.<p>
 */

public class PathComponent {

    /***************************************************
     *                                                 *
     * Creation of filespaces:			       *
     *                                                 *
     **************************************************/

    /**
     * Create an empty filespace, containing only a root directory
     */
    //@ ensures \result != null;
    public static Tree empty() {
	return new LeafTree(
		     new UnopenableFile("<root of the empty filesystem>",
					true));
    }


    /**
     * Convert from a path-component name to the filespace it denotes.<p>
     *
     * Throws an IOException if any errors occur while initially
     * scanning the component.  Component must be non-null.  The result
     * is always a non-null filespace.<p>
     *
     *
     * If complain is set, open will throw IOExceptions if the
     * path component does not exist, or if it is not a directory or a
     * zipfile.  If complain is not set, then an empty filespace will
     * be returned in these situations.  Either way, IOExceptions will
     * still be thrown if an error occurs reading an actual file or
     * directory.<p>
     *
     *
     * Note: changes to the filesystem named by component may or may
     * not be reflected in the returned Tree.<p>
     */
    //@ requires component != null;
    //@ ensures \result != null;
    public static Tree open(String component, boolean complain)
				 throws IOException {
	// Make sure component refers to an existing file:
	File root = new File(component);
	if (!root.exists()) {
	    if (complain)
		throw new IOException("no such file: " + component);
	    else
		return empty();
        }

	// Attempt to get the filespace contained in root
	Tree filespace = null;
	try {
	    if (isZipFilename(component))
		filespace = new ZipTree(root);
	    else if (root.isDirectory())
		filespace = new FileTree(root);
	} catch (ZipException E) {
	     throw new IOException(component + ": unable to process zipfile: "
		+ E.getMessage());
        } catch (IOException E) {
	     throw new IOException("unable to open/read file: "
		+ E.getMessage());
	}

	// Complain if component is not a directory or a zipfile:
	if (filespace == null) {
	    if (complain)
		throw new IOException("invalid classpath component: "
					+ component);
	    else
		return empty();
        }

	return filespace;
    }

    /**
     * Does a filename indicate that it is in zip format? <p>
     *
     */
    //@ requires name != null; 
    protected static boolean isZipFilename(String name) {
	return name.endsWith(".zip") || name.endsWith(".jar");
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /** A simple test driver */
    //@ requires \nonnullelements(args);
    public static void main(String[] args) throws IOException {
	// Check usage:
	if (args.length<1 || args.length>2) {
	    System.out.println("PathComponent: usage <path component> "
		+ "[<pathname>]");
	    return;
	}

	/*
	 * Create a new filespace from the path component args[0]:
	 */
	Tree T;
	try {
	    T = open(args[0], false);
	} catch (IOException E) {
	    System.out.println("Caught " + E);
	    return;
	};

	// If pathname given, try to print out that file then exit:
	if (args.length==2) {
	    String pathname = args[1];

	    // Strip off any leading '/'s:
	    while (pathname.length()>0 && pathname.charAt(0)=='/')
	        pathname = pathname.substring(1,pathname.length());

	    // Strip off any trailing '/'s:
	    while (pathname.length()>0 &&
		    pathname.charAt(pathname.length()-1)=='/')
	        pathname = pathname.substring(0,pathname.length()-1);

	    Tree F = T.getQualifiedChild(pathname, '/');
	    if (F==null) {
		System.out.println("No such file: " + args[1]);
		return;
	    }

	    InputStream I = ((GenericFile)F.data).getInputStream(); //@ nowarn Cast,Null;
	    for (;;) {
		int next = I.read();
		if (next<0)
		    return;

                System.out.write(next);
            }
        }

//	T.print("");

	// Otherwise, list all the files in the filespace by their
	// distinctive names, indicating which are directories and
	// giving their modification times
	Enumeration E = new TreeWalker(T);
	while (E.hasMoreElements()) {
	    Tree node = (Tree)E.nextElement();
	    GenericFile file = (GenericFile)node.data;	//@ nowarn Cast;
	    //@ assume file != null;

	    System.out.print(file.getHumanName() + " ");
	    if (file.lastModified()==0L)
		System.out.print("(unknown) ");
	    else
		System.out.print("(" + file.lastModified() + ") ");
	    if (file.isDirectory())
	        System.out.print("[D] ");

            System.out.println();
        }
    }
}
