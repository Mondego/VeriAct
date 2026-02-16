/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.util.Enumeration;
import javafe.genericfile.*;


/**
 * This module defines a very simple query interface for use in
 * locating Java files according to a classpath.
 */

public abstract class Query {

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
    //@ requires P != null;
    //@ requires \nonnullelements(P);
    public boolean accessable(String[] P) {
	return true;
    }


    /**
     * Attempt to locate the file typename+"."+extension in the package
     * P in the Java filespace.<p>
     *
     * If such a file is found, then a (non-null) GenericFile
     * representing it is returned.  Otherwise, null is returned.<p>
     */
    //@ requires P != null;
    //@ requires typename != null;
    //@ requires extension != null;
    //@ requires \nonnullelements(P);
    public abstract GenericFile findFile(String[] P, String typename,
					String extension);

    //@ requires P != null;
    //@ requires filename != null;
    public abstract GenericFile findFile(String[] P, String filename);

    /** Locates a file with given package, typename, and one of the given
      * extensions; the first directory on the search path containing a 
      * candidate file is used - within that directory, extensions near the
      * beginning of the extensions Vector take precedence.
      */
    //@ requires P != null;
    //@ requires typename != null;
    //@ requires extensions != null;
    public abstract GenericFile findFile(String[] P, String typename,
					String[] extensions);


    /** Returns an Enumeration containing GenericFile objects representing
	all the files in the given package P.
    */
    //@ requires P != null;
    public abstract Enumeration findFiles(String[] P);

    /***************************************************
     *                                                 *
     * Checking class/interface existence:	       *
     *                                                 *
     **************************************************/

    /**
     * Return true iff the fully-qualified outside type P.T exists in
     * our Java file space.
     */
    //@ requires P != null;
    //@ requires \nonnullelements(P);
    public boolean exists(String[] P, String T) {
	return (findFile(P, T, "java") != null)
	    || (findFile(P, T, "class") != null);
    }
}
