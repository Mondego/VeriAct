/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import java.io.IOException;
import javafe.genericfile.GenericFile;

/** 
 * This CorrelatedReader is built on top of another, given CorrelatedReader.
 * All of the methods in this class simply call the corresponding methods
 * on the underlying CorrelatedReader.
 *
 * @see javafe.util.CorrelatedReader
 * @author Rustan Leino
 */

public class FilterCorrelatedReader extends CorrelatedReader
{
    // Creation

    protected /*@ non_null */ CorrelatedReader child;

    /* The following invariant mentions "marked", which is not declared in
     * this class.  The justification for why it's still okay to declare
     * this invariant here is that all of the methods that modify "this.marked"
     * are overridden in the current class and, furthermore, the client is
     * not supposed to use "child" (hence, not "child.marked" either) after
     * passing it to the constructor.
     */
    //@ invariant marked == child.marked;

    /** Constructs a FilterCorrelatedReader with <code>child</code> as the
     * underlying CorrelatedReader.  After calling this constructor, the
     * caller should no longer use <code>child</code> directly.
     */

    protected FilterCorrelatedReader(/*@ non_null */ CorrelatedReader child) {
        child.clearMark();
        this.child = child;
    }

    // Misc

    /** Returns the file underlying this correlated reader.
     */

    public /*@non_null*/GenericFile getFile() {
        return child.getFile();
    }


    // Getting Locations

    /**
     * Returns the location of the last character read.  If no
     * character has been read yet, returns a whole-file location for
     * this file.
     */

    public int getLocation() {
        return child.getLocation();
    }

    // [Un]marking

    /** See documentation in superclass.
     */

    public void mark() {
        child.mark();
        marked = true;
    }

    /** See documentation in superclass.
     */

    public void clearMark() {
        child.clearMark();
        marked = false;
    }

    /** See documentation in superclass.
     */

    public void reset() throws IOException {
        child.reset();
        marked = false;
    }

    public /*@non_null*/CorrelatedReader createReaderFromMark(int discard)
            throws IndexOutOfBoundsException {
        return child.createReaderFromMark(discard);
    }

    // I/O

    /**
     * Closes us.  No other I/O operations may be called on us after
     * we have been closed.
     */

    public void close() {
        child.close();
        super.close();
    }

    /**
     * Reads the next character from this input stream. 
     * Does unicode conversion. <p>
     *
     *
     * Requires we are open.<p>
     * 
     * @return   A unicode character, or -1.<p>
     */
    public int read() throws IOException {
        return child.read();
    }
}
