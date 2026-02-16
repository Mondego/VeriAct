/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.io.*;
import java.util.Properties;
import java.util.Enumeration;

import javafe.genericfile.*;


/**
 * Functions for dealing with classpaths.
 */

public class ClassPath {

    /************************************************************
     *								*
     * Accessing our current classpath:				*
     *								*
     ***********************************************************/

    /**
     * Return our current classpath; if the Java system property
     * <code>java.class.path.skip</code> is set to <var>n</var>, we
     * ignore the first <var>n</var> components of the path. <p>
     *
     * This makes it easier to write Java applications that use the
     * classpath because we can append the path component containing
     * the application binaries to the start of the classpath then
     * remove them here, making it look like the classpath was not
     * disturbed.  (In particular, the automatic addition of the system
     * libraries is not disturbed.)
     */
    //@ ensures \result != null;
    public static String current() {
	String arg = System.getProperty("java.class.path.skip", "0");
	int skip = 0;
	try {
	    skip = new Integer(arg).intValue();
	} catch (NumberFormatException e) {}

	String path = System.getProperty("java.class.path", ".");

	for (int i=0; i<skip; i++) {
	    int sep = path.indexOf(File.pathSeparatorChar);
	    if (sep == -1 || sep == path.length()-1)
		return ".";
	    path = path.substring(sep + 1);
	}

	return path;
    }

    /**
     * Set our current classpath by changing the property
     * <code>java.class.path</code>.<p>
     *
     */
    //@ requires newClassPath != null;
    public static void set(String newClassPath) {
	Properties P = System.getProperties();
	P.put("java.class.path", newClassPath);
	System.setProperties(P);
    }


    /************************************************************
     *								*
     * Obtaining the namespace associated with a classpath:	*
     *								*
     ***********************************************************/

    /**
     * Get the filtered filespace (cf {@link PathComponent})
     * specified by a classpath.  (Filtering is performed using
     * {@link PkgTree} on each of the path components before they are
     * union'ed together).<p>
     *
     * All {@link PkgTree} accessors and enumerators can be used on
     * the resulting filespace.<p>
     *
     * May throw an {@link IOException} if errors occur.<p>
     *
     * Iff complain is set, we throw {@link IOException}s if
     * non-existent or ill-formed path components are present in the
     * classpath.<p>
     *
     */
    //@ requires classpath != null;
    //@ ensures \result != null;
    public static Tree open(String classpath, boolean complain)
				throws IOException {
	if (classpath.length()==0) {
	    throw new IOException("Empty classpath");
	}

	String[] pathnames = StringUtil.parseList(classpath,
						  File.pathSeparatorChar);

	Tree[] components = new Tree[pathnames.length];
	for (int i=0; i<components.length; i++) {
	    components[i] = new PkgTree(PathComponent.open(pathnames[i],
							   complain));
	}

	return new UnionTree(components);
    }


    /**
     * Get the namespace specified by the current classpath using open;
     * this is a convenience function.<p>
     *
     * Iff complain is set, we throw {@link IOException}s if
     * non-existent or ill-formed path components are present in the
     * classpath.<p>
     */
    //@ ensures \result != null;
    public static Tree open(boolean complain) throws IOException {
	return open(current(), complain);
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /**
     * A nicer, formatted version of print.<p>
     *
     * @param P must be a filespace filtered via {@link PkgTree};
     * moreover <code>PkgTree.isPackage(P)</code> should be true.<p>
     */
    //@ requires P != null;
    public static void displayPackage(Tree P) {
	// Enumerate P's subpackages:
	for (Enumeration E = PkgTree.packages(P); E.hasMoreElements();) {
	    Tree SP = (Tree)E.nextElement();

	    // Display the current package's full name:
	    System.out.println(PkgTree.getPackageName(SP) + ":");

	    // List the sources of the current subpackage:
	    Enumeration S = PkgTree.components(SP, ".java");
	    while (S.hasMoreElements())
		System.out.println("  S> "
			+ ((Tree)S.nextElement()).getSimpleName());

	    // List the binaries of the current subpackage:
	    S = PkgTree.components(SP, ".class");
	    while (S.hasMoreElements())
		System.out.println("  B> "
			+ ((Tree)S.nextElement()).getSimpleName());
	}
    }


    /** A simple test driver */
    //@ requires args != null;
    /*@ requires (\forall int i; (0<=i && i<args.length)
		==> args[i] != null); */
    public static void main(String[] args) throws IOException {
	/*
	 * Parse command arguments:
	 */
	if (args.length>2) {
	    System.out.println("ClassPath: usage "
		+ "[<package name> [<class or interface name>]]");
	    return;
	}
	String packageName = (args.length>0 ? args[0] : "");
	String typeName   = (args.length>1 ? args[1] : null);

//	set("Tmp2/new.zip");
//	set("");

	System.out.println("classpath=" + current());

	// Get the indicated package:
	Tree P = open(false).getQualifiedChild(packageName, '.');
	if (P==null) {
	    System.out.println("No such package: " + packageName);
	    return;
        }

	// If no type given, just display the indicated package and exit:
	if (typeName==null) {
	    displayPackage(P);
	    return;
	}

	// Get the indicated source:
	Tree source = P.getChild(typeName+".java");
	if (source == null) {
	    System.out.println("No source is available for type "
			+ PkgTree.getPackageName(P) + "." + typeName);
	    return;
        }
	
	// Dump the source in question to standard out:
	GenericFile sourceFile = (GenericFile)source.data;	//@ nowarn Cast;
	//@ assume sourceFile != null;
	InputStream I = sourceFile.getInputStream();
	System.out.println(sourceFile.getHumanName() + ":");
	System.out.println("------------------------------ " +
	    typeName + ".java: ------------------------------ ");

	for (int next=I.read(); next>=0; next=I.read())
            System.out.write(next);
    }
}
