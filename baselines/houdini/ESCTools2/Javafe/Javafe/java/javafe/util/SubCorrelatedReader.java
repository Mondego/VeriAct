/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import javafe.genericfile.GenericFile;
import java.io.IOException;

/** 
 * A reader (aka input stream) that provides an associated
 * <i>location</i> with each character read.
 *
 * <p>See javafe.util.Location for the interpretation of these
 * locations.
 *
 * <p> We also provide a method to create a new
 * <code>CorrelatedReader</code> for the text between the marked
 * position and the current point in the stream.  Marking is also
 * allowed on the new <code>CorrelatedReader</code> object.
 *
 * @see javafe.util.Location
 * @author Cormac Flanagan
 */

public class SubCorrelatedReader extends BufferedCorrelatedReader
{
  private /*@ non_null */ GenericFile file;

  /** Returns the file underlying this correlated reader.
   */

  public /*@non_null*/GenericFile getFile() {
    return file;
  }

    /**
     * Creates a sub-reader. <p>
     *
     * This method captures the given <code>buf</code>, that is,
     * callers should no longer use <code>buf</code> after passing it
     * in to this constructor.
     */
    //@ requires STARTFREELOC-1 <= beforeBufLoc;
    public SubCorrelatedReader(/*@ non_null @*/ GenericFile file,
			       /*@ non_null @*/ byte[] buf,
			       int beforeBufLoc) {

	// Our locations are exactly [beforeBufLoc+1, beforeBufLoc+1+len):
	super(/*minLoc*/ beforeBufLoc+1, beforeBufLoc,
	      /*maxLoc*/ beforeBufLoc+1+buf.length);
	this.file = file;
	this.buf = buf;
	this.endBufNdx = buf.length;
    }

  /* ************************************************
   *                                                 *
   * The methods:                                    *
   *                                                 *
   **************************************************/

  /** See spec in the abstract <code>CorrelatedReader</code> class.
   */

  public int read() throws IOException {
    if (buf == null) {
      Assert.precondition("read() called on a closed CorrelatedReader");
    }

    if (curNdx == endBufNdx) {
      // refill buffer
      if (!refillBuf()) {
	// EOF
	// save position of last character
	lastCharNdx = curNdx;
	return -1;
      }
    }

    // save the position of this character, before it is read:
    lastCharNdx = curNdx;

    int c = buf[curNdx++];

    if (c=='\\' && oddSlashLoc+1 != getLocation()) {
      // Can be a unicode escape sequence (is an odd slash!)
      if (peek() == 'u') {
	// is a unicode escape sequence
	while (peek() == 'u') {
	  curNdx++;
	}
	char s[] = new char[4];
	for (int i=0; i<4; i++) {
	  s[i] = (char)readRaw(); 
	  if (Character.digit(s[i],16)==-1) {
	    throw new IOException("Bad unicode character sequence");
	  }
	}
	c = Integer.parseInt( new String(s), 16 );
      } else {
	// not a unicode
	oddSlashLoc = getLocation();
      }
    }

    return c;
  }

  protected boolean refillBuf() throws IOException {
    // the buffer of a SubCorrelatedReader cannot be refilled
    return false;
  }
}
