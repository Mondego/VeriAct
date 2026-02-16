/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.genericfile;


import java.io.IOException;
import java.io.InputStream;


/**
 * Instances of UnopenableFile are {@link GenericFile}s that cannot
 * be opened. <p>
 *
 * Their value lies solely in their associated naming, etc., info.<p>
 *
 * Example: {@link javafe.util.CorrelatedReader}'s keeps an open
 * {@link InputStream} and an associated {@link GenericFile}.  In the
 * case of unreopenable streams like stdin, the associated {@link
 * GenericFile} is an UnopenableFile with the name "stdin".
 */

public class UnopenableFile implements GenericFile {

    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     **************************************************/

    //* Our human readable name:
    /*@ non_null @*/ String humanName;

    //* Are we a directory?
    boolean isDir;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * Create a ordinary (aka, non-directory) UnopenableFile with
     * human-name name.
     *
     * The resulting file has no modification date available and a
     * local name of "".
     */
    public UnopenableFile(/*@ non_null @*/ String name) {
	this(name, false);
    }


    /**
     * Create an UnopenableFile with human-name name that is a
     * directory iff isDir.
     *
     * The resulting file has no modification date available and a
     * local name of "".
     */
    public UnopenableFile(/*@ non_null @*/ String name, boolean isDir) {
	humanName = name;
	this.isDir = isDir;
    }


    /***************************************************
     *                                                 *
     * GenericFile interface implementation:	       *
     *                                                 *
     **************************************************/

    public String getHumanName() { return humanName; }

    public String getCanonicalID() {
	return "<javafe.filespace.UnopenableFile>" + humanName;
    }

    public String getLocalName() { return ""; }

    public boolean isDirectory() { return isDir; }

    public InputStream getInputStream() throws IOException {
	throw new IOException(
	    "Attempt to open an unopenable genericfile");
    }

    public long lastModified() { return 0L; }

    public GenericFile getSibling(/*@non_null*/String n) { return null; }
}
