/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import java.io.*; 

import javafe.genericfile.GenericFile;

/** 
 * A reader (aka input stream) that provides an associated
 * <I>location</I> with each character read.
 *
 * <p> See javafe.util.Location for the interpretation of these locations.
 *
 * <p> This class also takes care of unicode conversion.  A unicode
 * character sequence consists of a backslash that is preceded by an
 * even number of backslashes, followed by one or more 'u's, followed
 * by four hexadecimal digits. This class converts each unicode character
 * sequence into a single character.
 *
 * <p> This class also provides the ability to mark a specific point
 * in the input stream, and to reposition the stream at the marked
 * position. We also provide a method to create a new
 * <code>CorrelatedReader</code> for the text between the marked
 * position and the current point in the stream. Marking is also
 * allowed on the new <code>CorrelatedReader</code> object. For
 * efficiency, we also provide the facility to remove the mark from
 * the stream.<p>
 *
 * @see javafe.util.Location
 * @author Cormac Flanagan
 */

public abstract class CorrelatedReader 
/* may later want "extends Reader" */
{
    // Creation

    /** Simple constructor.  Pretty boring, really. */
    //@ ensures !marked;

    protected CorrelatedReader() {}

    // Misc

    /** Returns the file underlying this correlated reader. */
    public abstract /*@non_null*/GenericFile getFile();

    // Getting Locations

    /**
     * Returns the location of the last character read.  If no
     * character has been read yet, returns a whole-file location for
     * this file.
     */
    //@ ensures \result != Location.NULL;
    public abstract int getLocation();

    // [Un]marking

    /**
     * True iff a mark has been set.
     */

    /*@ spec_public */ protected boolean marked = false;

    /**
     * Marks the position of the current character in this input
     * stream.  A subsequent call to the <code>reset</code> method
     * repositions this stream at the last marked position. <p>
     *
     * This method differs from
     * <code>java.io.InputStream.readlimit</code> in that it does not
     * take a <code>readlimit</code> argument. <p>
     *
     * @see javafe.util.CorrelatedReader#reset()
     * @see javafe.util.CorrelatedReader#clearMark()
     * @see javafe.util.CorrelatedReader#createReaderFromMark
     */
    //-@ modifies marked;
    //@ ensures marked;

    public abstract void mark();

    /**
     * Removes the mark (if any) from the input stream.
     *
     * @see javafe.util.CorrelatedReader#mark()
     */
    //@ modifies marked;
    //@ ensures !marked;

    public abstract void clearMark();

    /**
     * Repositions this stream to the position at the time the
     * <code>mark</code> method was last called on this input stream. <p>
     *
     * Requires that the input stream had been previously marked 
     * via the <code>mark()</code> method. <p>
     *
     * If mark() is called before read(), then the mark will be
     * restored to its previous value (and not the preceeding
     * character().<p>
     *
     * @exception IOException  if this stream is not marked.
     * @see javafe.util.CorrelatedReader#mark()
     */
    //@ requires marked;
    //-@ modifies marked;
    //@ ensures !marked;

    public abstract void reset() throws IOException;

    /**
     * Creates a <code>CorrelatedReader</code> object for the input
     * text from the marked position, to the current position. <p>
     *
     * Calls to <code>getLocation()</code> for characters from the
     * new <code>CorrelatedReader</code> yield the same locations as
     * calls to <code>getLocation()</code> for the same characters on
     * this <code>CorrelatedReader</code>. <p>
     *
     * The <code>discard</code> argument specifies the number of
     * characters to discard from the end of the
     * sub-<code>CorrelatedReader</code>. <p>
     *
     * Clears the mark as a side-effect. <p>
     *
     * Requires that the input stream had been previously marked via
     * the <code>mark()</code> method and that we have not been
     * closed. <p>
     *
     * @exception IndexOutOfBoundsException if <code>discard</code> is
     * negative or exceeds the number of marked characters
     *
     * @see javafe.util.BufferedCorrelatedReader#mark()
     */
    //@ requires 0 <= discard;
    //@ requires marked;
    //@ modifies marked;
    //@ ensures !marked;

    public abstract /*@non_null*/CorrelatedReader createReaderFromMark(int discard)
            throws IndexOutOfBoundsException;

    // I/O

    /**
     * Closes us.  No other I/O operations may be called on us after
     * we have been closed.
     */

    public void close() {}

    /**
     * Reads the next character from this input stream.  Does unicode
     * conversion. <p>
     *
     * Requires we are open.<p>
     * 
     * @return A unicode character, or -1.<p>
     */
    public abstract int read() throws IOException;
}
