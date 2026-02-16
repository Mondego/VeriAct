/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.genericfile;


import java.io.IOException;
import java.io.InputStream;


/**
 * A GenericFile is a least-common-denominator representation of a
 * read-only file (or directory).<p>
 *
 * Note: The existence of GenericFile does not imply the actual
 * existence of an underlying file.<p>
 * 
 *
 * Currently, GenericFile's can be created from normal files
 * (java.io.File) and zipfile entries (java.util.zip.ZipEntry).<p>
 *
 * Additional operations on GenericFiles will be added later as needed.<p>
 */

public interface GenericFile {

    /***************************************************
     *                                                 *
     * Operations on GenericFiles:		       *
     *                                                 *
     **************************************************/

    /**
     * Return a name that uniquely identifies us to the user.
     *
     * Warning: the result may not be a conventional filename or use
     * the system separators.
     */
    //@ ensures \result != null;
    public abstract String getHumanName();


    /**
     * Return a String that canonically represents the identity of our
     * underlying file.
     *
     * This function must be defined such that if two GenericFiles
     * return non-null canonical ID's then the IDs are the same
     * (modulo .equals) => the GenericFiles represent the same
     * underlying file.  Ideally, under normal circumstances, the =>
     * is actually a <=>.
     *
     * This function should only return null in exceptional
     * cases, such as when an I/O error in the underlying storage media
     * prevents construction of a canonical ID.
     *
     * Convention: Canonical IDs start with <X> where X is the
     * fully-qualified name of the class that mediates I/O to the
     * underlying file.  E.g., java.io.File for a normal disk file.
     */
    public abstract String getCanonicalID();


    /**
     * Return our local name, the name that distinguishes us
     * within the directory that contains us.
     *
     * E.g., "/a/b/c" has local name "c", "/e/r/" has local name "r", and
     * "/" has local name "".  (assuming "/" is the separator char)
     */
    //@ ensures \result != null;
    public abstract String getLocalName();


    /**
     * Do we represent a directory?
     */
    public abstract boolean isDirectory();


    /**
     * Open the file we represent as an InputStream.<p>
     *
     * java.io.IOEXception may be thrown for many reasons, including no
     * such file and read permission denied.<p>
     */
    //@ modifies objectState;
    //@ ensures \fresh(\result);
    //@ ensures \result != null;
    //@ ensures \result.isOpen;
    public abstract InputStream getInputStream() throws IOException;


    /**
     * Returns the time that the file represented by us was last
     * modified.<p>
     *
     * The return value is system dependent and should only be used to
     * compare with other values returned by last modified. It should
     * not be interpreted as an absolute time.<p>
     *
     * If a last-modified time is not available (e.g., underlying file
     * doesn't exist, no time specified in a zipentry, etc.), then 0L
     * is returned.<p>
     */
    public abstract long lastModified();


    /**
     * Attempt to return a GenericFile that describes the file in the
     * same "directory" as us that has the local name <code>n</code>. <p>
     *
     * No attempt is made to verify whether or not that file exists.<p>
     *
     * In cases where the notion of "containing directory" makes no
     * sense (e.g., streams or root directories), null is returned.
     */
    public abstract GenericFile getSibling(/*@ non_null @*/ String n);
}
