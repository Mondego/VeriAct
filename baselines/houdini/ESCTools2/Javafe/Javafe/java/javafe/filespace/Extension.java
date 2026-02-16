/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;

/**
 * This module contains functions for decomposing filenames (Strings)
 * into a basename and an extension.  I.e., "foo.java" -> "foo",
 * ".java" and "bar" -> "bar", "".
 *
 * <p> Extensions include the '.' if present so that no extension can
 * be distinguished from a blank one (i.e., "foo."). </p>
 *
 * <p> This also has the property that concatenating a filename's
 * basename with its extension always gives the original
 * filename. </p>
 */

public class Extension
{
    /**
     * Return the extension of a filename (including the ".") or "" if
     * it has none.  The extension is defined as the substring
     * starting with the last "." and ending at the end of the
     * filename.
     */
    public static /*@ non_null @*/ String 
            getExtension(/*@ non_null @*/ String filename) {
	int lastDot = filename.lastIndexOf(".");

	if (lastDot == -1)
	    return "";		// no extension present...

	return filename.substring(lastDot);
    }

    /**
     * Return the basename of a filename -- the part of a filename
     * that preceeds its extension (if any).  More precisely, the
     * prefix of the filename preceeding the last "." or the entire
     * filename if no "." is present.
     */
    public static /*@ non_null @*/ String 
            getBasename(/*@ non_null @*/ String filename) {
	int lastDot = filename.lastIndexOf(".");

	if (lastDot == -1)
	    return filename;		// no extension present...

	return filename.substring(0,lastDot);
    }

    /**
     * Return true iff a given filename has a particular extension.<p>
     *
     * It is faster to use endsWith for non-empty extensions; use this
     * function when extension may be empty ("").<p>
     */
    public static boolean hasExtension(/*@ non_null @*/ String filename,
                                       /*@ non_null @*/ String extension) {
	if (!filename.endsWith(extension))
	    return false;

	if (extension.equals(""))
	    return filename.lastIndexOf(".") == -1;

	return true;
    }
}
