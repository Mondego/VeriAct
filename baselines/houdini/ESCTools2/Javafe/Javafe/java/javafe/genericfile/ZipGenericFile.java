/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.genericfile;


import java.io.InputStream;
import java.io.IOException;

import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;


/**
 * A ZipGenericFile represents a zipfile-entry file
 * (java.util.zip.ZipEntry) as a GenericFile.
 *
 * WARNING: ZipEntry's (but not ZipFile's) always use "/" as their
 * separator.
 */

public class ZipGenericFile implements GenericFile {

    //@ invariant underlyingZipFile != null;
    public ZipFile  underlyingZipFile;

    //@ invariant underlyingZipEntry != null;
    public ZipEntry underlyingZipEntry;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /** Create a generic file representing a ZipEntry in a ZipFile: */
    //@ requires file != null && entry != null;
    public ZipGenericFile(ZipFile file, ZipEntry entry) {
	underlyingZipFile = file;
	underlyingZipEntry = entry;
    }


    /***************************************************
     *                                                 *
     * Operations on ZipGenericFiles:		       *
     *                                                 *
     **************************************************/

    /**
     * Return a name that uniquely identifies us to the user.
     *
     * Warning: the result may not be a conventional filename or use
     * the system separators.
     */
    public String getHumanName() {
	return underlyingZipFile.getName() + ":"
		+ underlyingZipEntry.toString();
    }


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
    public String getCanonicalID() {
	/*
	 * WARNING: this doesn't quite implement the spec.  In
	 * particular, we can't canonicalize underlyingZipFile's
	 * pathname since we can't convert it to an absolute pathname
	 * (don't know current directory when it was created.)
	 */
	return "<java.util.zip.ZipEntry>("
	    + underlyingZipFile.getName().length() + ")"
	    + underlyingZipFile.getName()
		+ ":" + underlyingZipEntry.getName();
    }


    /**
     * Return our local name, the name that distinguishes us
     * within the directory that contains us.
     *
     * E.g., "/a/b/c" has local name "c", "/e/r/" has local name "r", and
     * "/" has local name "".  (assuming "/" is the separator char)
     */
    public String getLocalName() {
	String path = underlyingZipEntry.getName();
	while (path.endsWith("/"))
	    path = path.substring(0,path.length()-1);

	int index = path.lastIndexOf('/');
	if (index== -1)
	    return path;
	else
	    return path.substring(index+1, path.length());
    }


    /**
     * Do we represent a directory?
     */
    public boolean isDirectory() {
	return underlyingZipEntry.isDirectory();
    }


    /**
     * Open the file we represent as an InputStream.<p>
     *
     * java.io.IOEXception may be thrown for many reasons, including no
     * such file and read permission denied.<p>
     */
    public InputStream getInputStream() throws IOException {
	return underlyingZipFile.getInputStream(underlyingZipEntry);
    }


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
    public long lastModified() {
	long time = underlyingZipEntry.getTime();

	/*
	 * getTime returns -1L if no time was specified in the zipfile;
	 * convert this to 0L so we behave the same as
	 * File.lastModified():
	 */
	if (time<0)
	    time = 0;

	return time;
    }


    /**
     * Attempt to return a GenericFile that describes the file in the
     * same "directory" as us that has the local name <code>n</code>. <p>
     *
     * No attempt is made to verify whether or not that file exists.<p>
     *
     * In cases where the notion of "containing directory" makes no
     * sense (e.g., streams or root directories), null is returned.
     */
    public GenericFile getSibling(/*@non_null*/String n) { 
	String name = underlyingZipEntry.getName();

	// Root directory (never appears in real zipfiles) has no siblings:
	if (name.equals(""))
	    return null;

	// get "parent", including trailing separator if one:
	int index = name.lastIndexOf('/');
	name = name.substring(0,index+1) + n;

	ZipEntry siblingEntry = underlyingZipFile.getEntry(name);
	if (siblingEntry == null) {
	    // Return something that will cause a file-not-found error
	    // to be raised when an attempt is made to open the entry
	    siblingEntry = new ZipEntry(name);
	}
	return new ZipGenericFile(underlyingZipFile, siblingEntry);
    }
}
