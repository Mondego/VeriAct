/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.io.IOException;
import java.util.Enumeration;

import javafe.genericfile.*;


/**
 * This module implements the Query "interface" by using the Java
 * filespace classes (ClassPath, PathComponent, etc.) provided by the
 * javafe.filespace package.
 */

public class SlowQuery extends Query {

    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * The Java file space that corresponds to our classpath.
     */
    private Tree javaFileSpace;


    /**
     * Create an query engine that may be queried about packages and
     * classes in the classpath classpath.
     *
     */
    //@ requires classpath != null;
    public SlowQuery(String classpath) throws java.io.IOException {
	javaFileSpace = ClassPath.open(classpath, false);
    }

    /**
     * Create an query engine that may be queried about packages and
     * classes in the current Java classpath (cf. ClassPath) at the
     * time the engine was created.  (I.e., later changes to the
     * current classpath have no effect on the query engine.)
     */
    public SlowQuery() throws java.io.IOException {
	javaFileSpace = ClassPath.open(false);
    }


    /***************************************************
     *                                                 *
     * Locating files:				       *
     *                                                 *
     **************************************************/

    /**
     * Return true iff the package P in the Java filespace is
     * "accessible".<p>
     *
     * Warning: the definition of accessible is host system dependent
     * and may in fact be defined as always true.<p>
     */
    //@ also
    //@   requires P != null;
    public boolean accessable(String[] P) {
	return (getPackage(P) != null);
    }


    /**
     * Attempt to locate the file typename+"."+extension in the package
     * P in the Java filespace.<p>
     *
     * If such a file is found, then a (non-null) GenericFile
     * representing it is returned.  Otherwise, null is returned.<p>
     */
    //@ also
    //@   requires P != null;
    //@   requires typename != null;
    //@   requires extension != null;
    public GenericFile findFile(String[] P, String typename,
					String extension) {
	return findFile(P,typename+"."+extension);
    }

    //@ also
    //@   requires P != null;
    //@   requires filename != null;
    public GenericFile findFile(String[] P, String filename) {
	Tree Package = getPackage(P);
	if (Package==null)
	    return null;

	Tree node = Package.getChild(filename);
	if (node==null)
	    return null;

	return (GenericFile)node.data;		//@ nowarn Cast;
    }

    //@ also
    //@   requires P != null;
    //@   requires typename != null;
    //@ requires extensions != null;
    public GenericFile findFile(String[] P, String typename,
					String[] extensions) {
// FIXME - only utilizes the first package
	Tree Package = getPackage(P);
	if (Package==null)
	    return null;

        for (int i=0; i<extensions.length; ++i) {
	    String extension = extensions[i];
	    Tree node = Package.getChild(typename+"."+extension);
	    if (node != null) return (GenericFile)node.data; //@ nowarn Cast;
	}
	return null;
     }

    //@ also
    //@   requires P != null;
    public Enumeration findFiles(String[] P) {
	Tree Package = getPackage(P);
	if (Package==null)
	    return null;

	return Package.children();
    }

    /*
     * Helper function: return the node corresponding to package P in
     * the Java filespace or null if there is no such corresponding
     * package.
     */
    //@ requires P != null;
    //@ requires \nonnullelements(P);
    private Tree getPackage(String[] P) {
	Tree Package = javaFileSpace;

	for (int i=0; i<P.length; i++) {
	    if (Package != null)
		Package = Package.getChild(P[i]);
	}

	return Package;
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /** A simple test driver */
    //@ requires args != null;
    //@ requires \nonnullelements(args);
    public static void main(String[] args) throws IOException {
	/*
	 * Parse command arguments:
	 */
	if (args.length != 3) {
	    System.out.println(
		"Query: usage <package name> <typename> <extension>");
	    return;
	}

	Query Q = new SlowQuery();	
	String[] P = StringUtil.parseList(args[0],'.');
	
	if (!Q.accessable(P))
	    System.out.println("Package not accessible.");

	GenericFile result = Q.findFile(P, args[1], args[2]);
	if (result==null)
	    System.out.println("Type not found.");
	else
	    System.out.println("Type found @ " + result.getHumanName());
    }
}
