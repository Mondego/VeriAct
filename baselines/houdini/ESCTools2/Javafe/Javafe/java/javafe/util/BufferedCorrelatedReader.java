/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import java.io.IOException;

/**
 * Instances of this <code>CorrelatedReader</code> contain a buffer.
 */

public abstract class BufferedCorrelatedReader extends CorrelatedReader
{
  /**
   * All locations are allocated at or above a medium-large int
   * STARTFREELOC to make it less likely that other program ints can
   * be interpreted as locations.
   */

  /*@ spec_public */ static final int STARTFREELOC = 1000000;

  /**
   * The allocated location range for this CorrelatedReader instance
   * is [minLoc, maxLoc). <p>
   *
   * Note that in practice only a prefix,
   *   [minLoc, min(beforeBufLoc + curNdx, maxLoc-1)],
   * of the range representing the characters read so far are actually valid
   * locations at any given time.
   */
  //@ invariant STARTFREELOC <= minLoc;
  //@ invariant minLoc <= maxLoc;
  
  protected int minLoc;
  protected int maxLoc;

  /**
   * While open, we buffer input in the buffer <code>buf</code>, which
   * grows on demand.
   *
   * We are closed iff <code>buf</code> is non-null.
   */

  /*@spec_public*/ protected /*@nullable*/byte[] buf;

  /**
   * The location of the first character in the buffer, minus 1. <p>
   *
   * The locations of later valid characters are obtained by adding
   * their index to this loc. <p>
   */
  //@ invariant minLoc-1 <= beforeBufLoc;

  protected int beforeBufLoc;

  /**
   * The first unused entry in the buffer.  Aka,
   * buf[0]...buf[endBufNdx-1] contains valid data (if buf != null).
   */
  //@ invariant buf != null ==> endBufNdx <= buf.length;

  protected int endBufNdx;

  /**
   * The index of the next character to be read from the buffer.  If
   * equal to endBufNdx, then we need to read more from the stream.
   */
  //@ invariant 0 <= curNdx;
  //@ invariant curNdx <= endBufNdx;

  protected int curNdx = 0;

  /**
   * The value of curNdx for the last character read, or, if no
   * characters have been read yet, the value of curNdx.
   *
   * Note that because of unicode conversion, this may not be just
   * curNdx-1.  This exists because mark() marks from just before
   * the last character read, not the current position.
   */
  //@ invariant 0 <= lastCharNdx;
  //@ invariant lastCharNdx <= curNdx;

  protected int lastCharNdx = curNdx;

  /**
   * For unicode conversion, we need to know if we have just seen
   * an even or odd number of backslashes.  Accordingly, we record
   * the location of the last odd backslash read (Location.NULL if one
   * has not been seen yet).
   */

  protected int oddSlashLoc = Location.NULL;


    /**
     * This constructor leaves <code>buf</code> as null.
     */
    //@ requires STARTFREELOC <= minLoc;
    //@ requires minLoc <= maxLoc;
    //@ requires minLoc-1 <= beforeBufLoc;
    //@ ensures curNdx == 0;
    //@ ensures !marked;
    //@ ensures buf == null;
    BufferedCorrelatedReader(int minLoc, int beforeBufLoc, int maxLoc) {
	this.minLoc = minLoc;
	this.beforeBufLoc = beforeBufLoc;
	this.maxLoc = maxLoc;
    }


  /**
   * Closes us.  No other I/O operations may be called on us after
   * we have been closed.
   */
  //@ also modifies buf;
  //@      ensures buf == null;

  public void close() {
    buf = null;
    super.close();
  }

  /**
   * Returns the location of the last character read.  If no
   * character has been read yet, returns a whole-file location for
   * this file.
   */

  public int getLocation() {
    int loc = beforeBufLoc + curNdx;

    if (loc<minLoc) { // no characters read yet
      return FileCorrelatedReader.createWholeFileLoc(getFile());
    }

    if (loc>maxLoc) {
      ErrorSet.fatal("Input file `" + getFile().getHumanName() +
		     "' is too large (maximum file size is " +
		     LocationManagerCorrelatedReader.MAXFILESIZE + " bytes).");
    }

    return loc;
  }

  /**
   * Refills the buffer. <p>
   *
   * In doing so, may reallocate the buffer. <p>
   *
   * Returns true iff not end-of-file, and at least one character
   * was read from the file.  Throws an IOException if no
   * characters could be read from the stream.<p>
   *
   * Requires we are open.<p>
   */
  //@ requires buf != null;
  //@ requires curNdx == endBufNdx;
  //@ ensures \result ==> curNdx < endBufNdx;

  protected abstract boolean refillBuf() throws IOException;

  /**
   * Peeks the next character from this input stream. 
   * Does no unicode conversion. <p>
   *
   * Requires we are open.<p>
   */
  //@ requires buf != null;
  //@ ensures 0 <= \result ==> curNdx < endBufNdx;
  protected int peek() throws IOException {
    if (curNdx == endBufNdx) {
      // Refill buffer:
      if (!refillBuf()) {
	// EOF
	return -1;
      }
    }

    return (/*+@(non_null)*/buf)[curNdx];
  }

  /**
   * Reads the next character from this input stream. 
   * Does no unicode conversion. <p>
   *
   * Requires we are open.<p>
   */
  //@ requires buf != null;
  protected int readRaw() throws IOException {
    if (curNdx == endBufNdx) {
      // Refill buffer:
      if (!refillBuf()) {
	// EOF
	return -1;
      }
    }

    return (/*+@(non_null)*/buf)[curNdx++];
  }

  /* ************************************************
   *                                                 *
   * Marks:                                          *
   *                                                 *
   * ************************************************/

  /**
   * The value of curNdx at the mark point (if marked is true).
   */
  //@ invariant marked ==> 0 <= markNdx && markNdx <= curNdx;

  protected int markNdx; /*@ readable markNdx if marked; */

  /**
   * Marks the position of the current character in this input
   * stream.  A subsequent call to the <code>reset</code> method
   * repositions this stream at the last marked position.
   *
   * This method differs from
   * <code>java.io.InputStream.readlimit</code> in that it does not
   * take a <code>readlimit</code> argument. <p>
   *
   * @see     javafe.util.BufferedCorrelatedReader#reset()
   * @see     javafe.util.BufferedCorrelatedReader#clearMark()
   * @see     javafe.util.BufferedCorrelatedReader#createReaderFromMark
   */

  public void mark() {
    markNdx = curNdx;
    marked = true;
  }

  /**
   * Removes the mark (if any) from the input stream.
   *
   * @see     javafe.util.BufferedCorrelatedReader#mark()
   */
  public void clearMark() {
    marked = false;
  }

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
   * @see     javafe.util.BufferedCorrelatedReader#mark()
   */

  public void reset() throws IOException {
    if (!marked) {
      throw new IOException("mark: CorrelatedReader not marked");
    }

    curNdx = markNdx;
    lastCharNdx = curNdx;

    // System.out.println("Reset to "+(beforeBufLoc + curNdx)+" - "
    // +getLocation());

    marked = false;
  }

  /** Returns the location of the character before the mark.
   */
  //@ requires marked;
  //@ ensures STARTFREELOC-1 <= \result;
  //@ pure
  public int getBeforeMarkLocation() {
    Assert.notFalse(marked);

    return beforeBufLoc + markNdx;
  }

  /** Returns a new buffer containing the contents of this BufferedCorrelatedReader's
   * buffer, from the marked position to the current position less
   * <code>discard</code> bytes (not characters).
   *
   * Clears the mark as a side-effect. <p>
   *
   * @exception IndexOutOfBoundsException if <code>discard</code> is negative or exceeds the number of marked characters
   */
  //@ requires 0 <= discard;
  //@ requires marked;
  //@ modifies marked;
  //@ ensures !marked;
  public /*@non_null*/byte[] getBufferFromMark(int discard)
      throws IndexOutOfBoundsException {

    if (buf == null) {
      Assert.precondition("getBufferFromMark called on a closed CorrelatedReader");
    }
    if (!marked) {
      Assert.precondition("getBufferFromMark called on a " +
			  "CorrelatedReader that has not been marked");
    }
    if (discard < 0 || curNdx-markNdx < discard) {
      throw new IndexOutOfBoundsException();
    }

    int start = markNdx;
    int len = curNdx-markNdx - discard;

    byte[] newBuffer = new byte[len];
    System.arraycopy((/*+@(non_null)*/buf), start, newBuffer, 0, len);

    marked = false;

    return newBuffer;
  }

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
   * @exception IndexOutOfBoundsException if <code>discard</code> is negative or exceeds the number of marked characters
   *
   * @see javafe.util.BufferedCorrelatedReader#mark()
   */

  //@ also requires 0 <= discard;
  public /*@non_null*/CorrelatedReader createReaderFromMark(int discard)
	throws IndexOutOfBoundsException {
    int b4markLoc = getBeforeMarkLocation();
    byte[] subBuffer = getBufferFromMark(discard);
    return new SubCorrelatedReader(getFile(), subBuffer, b4markLoc);
  }
}
