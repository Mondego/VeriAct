/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import javafe.genericfile.GenericFile;

import java.util.Vector;
import java.io.IOException;

/**
 * This <code>CorrelatedReader</code> class manages the allocation of
 * location numbers.
 */

public abstract class LocationManagerCorrelatedReader 
    extends BufferedCorrelatedReader
{
  // Represents the internal state - changes to this should be
  // benevolent side effects.
  //@ public model JMLDataGroup internalState; 

  /* ************************************************
   *                                                 *
   * Class variables & constants:                    *
   *                                                 *
   * ************************************************/

  /**
   * The next location to be allocated to a LocationManagerCorrelatedReader
   * instance.  (The next instance's stream will have locations in a prefix of
   * [freeLoc, freeLoc+MAXFILESIZE).) <p>
   *
   * We do not use ints below STARTFREELOC to denote locations.  If
   * freeLoc+MAXFILESIZE results in an overflow, then too many
   * LocationManagerCorrelatedReader instances have been created and an
   * assertion failure occurs.
   */
  //@ invariant STARTFREELOC <= freeLoc;

  //@ spec_public
  private static int freeLoc = STARTFREELOC;

  /**
   * A location is an integer that encodes file/line/column/offset
   * information.  When a LocationManagerCorrelatedReader is created,
   * we allocate MAXFILESIZE locations for its stream.  An assertion
   * failure will occur if a location is requested for a character
   * located at an offset greater than or equal to MAXFILESIZE.
   */

  static final int MAXFILESIZE = 30000000;

  /**
   * Each LocationManagerCorrelatedReader has a "new line offset
   * array", or NLOA, that contains the offset of the beginning of
   * each line. This array allows us to map an offset to line or
   * column information. <p>
   *
   * NLOA is defined for indexes in [0..curLineNo). <p>
   *
   * If NLOA[i], i in [0..curLineNo), = j, then the characters at
   * offset j and j-1 were on different lines. Furthermore, either
   *   (a) i=j=0, or
   *   (b) i>0, j>0, and the character at offset j-1 is a newline. <p>
   *
   * Note: this means a line ends just *after* a newline, not before. <p>
   */
  /*@ invariant curLineNo <= NLOA.length
    @	&& (\forall int i; 0 <= i && i < curLineNo ==> 0 <= NLOA[i]); */

  /*@ spec_public */ private /*@ non_null */ int[] NLOA; //@ in internalState;

  /** The default initial size to allocate an NLOA array */
  //@ invariant NLOA_DEFAULT_SIZE <= NLOA.length;

  private static final int NLOA_DEFAULT_SIZE = 200;

  protected int streamid;

    /**
     * This constructor allocates a range of location for use by the
     * CorrelatedReader.
     */
    //@ ensures curNdx == 0;
    //@ ensures !marked;
    //@ ensures !isWholeFile;
    //@ ensures buf == null;
    LocationManagerCorrelatedReader() {
	// Allocate locations for us:
	super(/*minLoc*/ freeLoc, /*beforeBufLoc*/ freeLoc-1,
	      /*maxLoc*/ freeLoc+MAXFILESIZE);
	freeLoc += MAXFILESIZE;

	if (freeLoc<0) {
	    long limit = (1L<<32 - STARTFREELOC) / MAXFILESIZE;
	    ErrorSet.fatal("javafe.util.LocationManagerCorrelatedReader "
			   + "maximum # of files limit (~"+limit+") exceeded");
	}
	
	this.NLOA = new int[NLOA_DEFAULT_SIZE];
	this.NLOA[0] = 0;
	
        streamid = allCorrStreams.size();
	allCorrStreams.addElement(this);
    }


  /**
   * Closes us.  No other I/O operations may be called on us after
   * we have been closed.
   */

    //@ also protected normal_behavior
    //@ assignable maxLoc;
  public void close() {
    if (maxLoc == freeLoc) {
      maxLoc = beforeBufLoc + curNdx + 1;
      freeLoc = maxLoc;
    }
    super.close();
  }

  /** Records a newline at the current location.
   */
  //@ modifies NLOA;
  //@ ensures curLineNo < NLOA.length;
  //@ ensures 0 <= NLOA[curLineNo];

  protected void recordNewLineLocation() {
    if (curLineNo == NLOA.length) {
      int[] new_NLOA = new int[ 2*NLOA.length ];
      System.arraycopy( NLOA, 0, new_NLOA, 0, NLOA.length );
      NLOA = new_NLOA;
    }
    //@ assert curLineNo < NLOA.length;
    NLOA[ curLineNo ] = beforeBufLoc + curNdx + 1 - minLoc;
  }

  /**
   * A static Vector containing all LocationManagerCorrelatedReader
   * instances, in the order they were allocated, which is used to
   * map a given location to a corresponding LocationManagerCorrelatedReader
   * instance. <p>
   *
   * A location's streamId is its index in allCorrStreams.  See
   * locToStreamId(int) for the algorithm mapping locations to
   * streamId's.
   */
  //@ invariant !allCorrStreams.containsNull;
  //@ invariant allCorrStreams.elementType == \type(LocationManagerCorrelatedReader);

  /*@spec_public*/ private static /*@ non_null @*/ Vector allCorrStreams = new Vector();

  /**
   * Creates and returns a vector that associates file numbers 
   * to file names.
   */
  //@ ensures \result != null;
  //@ ensures \result.elementType == \type(String);
  //@ ensures !\result.containsNull;

  public static Vector fileNumbersToNames() {
    Vector v = new Vector();
    //@ set v.elementType = \type(String);
    //@ set v.containsNull = false;
    for(int i = 0; i < allCorrStreams.size(); i++) {
      LocationManagerCorrelatedReader s = getCorrStreamAt(i);
      v.addElement(s.getFile().getHumanName());
    }
    return v;
  }

  /* ************************************************
   *                                                 *
   * Inspecting Locations:                           *
   *                                                 *
   *                                                 *
   * These methods are mostly package-protected.     *
   * The Location class provides access to these     *
   * methods.                                        *
   *                                                 *
   * ************************************************/

  /**
   * Return the LocationManagerCorrelatedReader associated with
   * streamId i. <p>
   *
   * If i is not a valid streamId (aka, i not in
   * [0, allCorrStreams.size()) ), an assertion failure occurs. <p>
   */
  //@ requires 0 <= i && i < allCorrStreams.elementCount;
  //@ modifies \nothing;
  //@ ensures \result != null;

  protected static LocationManagerCorrelatedReader getCorrStreamAt(int i) {
    try {
      LocationManagerCorrelatedReader c =
        (LocationManagerCorrelatedReader)allCorrStreams.elementAt(i);

      return c;
    } catch (ArrayIndexOutOfBoundsException e) {
      Assert.precondition("invalid streamId " + i);
      return null;    // dummy return
    }
  }

  /**
   * Attempt to return the valid regular location associated with a
   * given streamId, line #, and col #. <p>
   *
   * If there is no such location currently in existence, an
   * assertion failure will occur. <p>
   *
   * This method is intended mainly for debugging purposes. <p>
   */
  //@ requires 0 <= streamId && streamId < allCorrStreams.elementCount;
  //@ requires 0 < line;
  //@ requires 0 <= col;
  //@ ensures \result != Location.NULL;

  static int makeLocation(int streamId, int line, int col) {
    LocationManagerCorrelatedReader s = getCorrStreamAt(streamId);
	
    if (s.isWholeFile) {
      return s.minLoc;
      //Assert.fail("streamId denotes a whole file location");
    }

    if (line>s.curLineNo) {
	System.out.println("INTERNAL ERROR: invalid request to form a location (out of range line number): " + streamId + " " + line + " " + col + " " + streamIdToFile(streamId).getHumanName());
	line = 1; col = 0;
    }

    if ((line==s.curLineNo && col+1>s.curNdx) ||
	(line != s.curLineNo && col+1>(s.NLOA[line]-s.NLOA[line-1]))) {
	System.out.println("INTERNAL ERROR: invalid request to form a location (out of range column number): " + streamId + " " + line + " " + col + " " + streamIdToFile(streamId).getHumanName());
	col = 0;
    }

    int loc;
    if (line == 0) {
      loc = s.minLoc + col;
    } else {
      loc = s.minLoc + s.NLOA[line-1] + col;
    }

    // Verify we got the right location:
    if (locToStreamId(loc) != streamId ||
	locToColumn(loc) != col ||
	locToLineNumber(loc) != line) {
      Assert.fail("bug found in makeLocation");
    }

    return loc;
  }

  /**
   * Return the LocationManager CorrelatedReader instance associated
   * with location loc. <p>
   *
   * Requires that loc be a valid location. <p>
   */
  //@ requires loc != Location.NULL;
  //@ modifies \nothing;
  //@ ensures \result != null;
  //@ ensures \result.minLoc <= loc && loc <= \result.beforeBufLoc + \result.curNdx;

  protected static LocationManagerCorrelatedReader locToStream(int loc) {
    int i = locToStreamId(loc);
    LocationManagerCorrelatedReader s = getCorrStreamAt(i);
    Assert.notFalse(s.minLoc <= loc && loc <= s.beforeBufLoc + s.curNdx); //@ nowarn Pre;
    return s;
  }

  /**
   * Returns the internal stream ID used for the stream associated
   * with location <code>loc</code>.
   *
   * Requires that loc be a valid location. <p>
   */
  //@ normal_behavior
  //@ requires loc != Location.NULL;
  //@ modifies \nothing;
  //@ ensures 0 <= \result && \result < allCorrStreams.elementCount;
  static int locToStreamId(int loc) {
    // This is somewhat inefficient:
    for(int i = 0; i < allCorrStreams.size(); i++) {
      LocationManagerCorrelatedReader s = getCorrStreamAt(i);
      if (s.minLoc <= loc && loc <= s.beforeBufLoc + s.curNdx) {
	return i;
      }
    }

    // Bad location:
    Assert.precondition("Bad location "+loc);
    return -1; // dummy return
  }

  /**
   * Is a location a whole file location?
   *
   * Requires that loc be a valid location or NULL. <p>
   */

  //@ modifies \nothing;
  static boolean isWholeFileLoc(int loc) {
    if (loc == Location.NULL) {
      return false;
    }
    return locToStream(loc).isWholeFile;
  }

  /**
   * Returns the offset associated with location 
   * <code>loc</code>. <p>
   *
   * Requires that loc be a valid regular location (regular ==> not
   * a whole-file location). <p>
   *
   * Note: offsets start with 1 (a la emacs). <p>
   */
  //@ requires loc != Location.NULL;

  static int locToOffset(int loc) {
    if (isWholeFileLoc(loc)) {
      Assert.precondition("locToOffset passed a whole-file location");
    }

    LocationManagerCorrelatedReader s = locToStream(loc);
    return loc - s.minLoc + 1;
  }

  /**
   * Returns the line number associated with location 
   * <code>loc</code>. <p>
   *
   * Requires that loc be a valid regular location (regular ==> not
   * a whole-file location). <p>
   *
   * Note: line numbers start with 1 (a la emacs). <p>
   */
  //@ requires loc != Location.NULL;
  //@ ensures 1 <= \result;

  static int locToLineNumber(int loc) {
    return locToColOrLine(loc, false);
  }

  /**
   * Returns the column number associated with location 
   * <code>loc</code>. <p>
   *
   * Requires that loc be a valid regular location (regular ==> not
   * a whole-file location). <p>
   *
   * Note: column numbers start with 0. <p>
   */
  //@ requires loc != Location.NULL;
  //@ ensures 0 <= \result;

  static int locToColumn(int loc) {
    return locToColOrLine(loc, true);
  }

  /**
   * Returns the column number (if wantColumn) or line number (if
   * !wantColumn) associated with location <code>loc</code>. <p>
   *
   * Requires that loc be a valid regular location (regular ==> not
   * a whole-file location). <p>
   *
   * Note: line numbers start with 1 (a la emacs), while column
   * numbers start with 0. <p>
   */
  //@ requires loc != Location.NULL;
  //@ ensures 0 <= \result;
  //@ ensures !wantColumn ==> 1 <= \result;

  private static int locToColOrLine(int loc, boolean wantColumn) {
    LocationManagerCorrelatedReader s = locToStream(loc);
    int offset = loc - s.minLoc;

    for (int i = s.curLineNo-1; /* Have sentinel s.NLOA[0]==0 */; i--) {
      if( s.NLOA[i] <= offset ) {
	// Line i+1 begins before offset
	return wantColumn ? offset-s.NLOA[i] : i+1;
      }
    }
  }

  /**
   * Returns the GenericFile associated with stream id
   * <code>id</code>, where <code>id</code> has previously been
   * returned by <code>locToStreamId</code>.
   *
   * Requires that id be a valid streamId.
   */
  //@ requires 0 <= id && id < allCorrStreams.elementCount;

  static GenericFile streamIdToFile(int id) {
    return getCorrStreamAt(id).getFile();
  }

  /**
   * Returns the GenericFile associated with location
   * <code>loc</code>. <p>
   *
   * Requires that id be a valid streamId of a FileCorrelatedReader. <p>
   */
  //@ requires loc != Location.NULL;
  //@ ensures \result != null;

  static GenericFile locToFile(int loc) {
    return locToStream(loc).getFile();
  }


  /* ************************************************
   *                                                 *
   * Whole-file correlated readers:                  *
   *                                                 *
   * ************************************************/

  /**
   * Are all of our locations whole-file locations?
   */
  /*@spec_public*/ protected boolean isWholeFile = false;

  //@ invariant isWholeFile ==> buf==null;


  /* ************************************************
   *                                                 *
   * Stuff related to counting lines:                *
   *                                                 *
   * ************************************************/

  /**
   * The total # of lines that have been read so far by all
   * FileCorrelatedReaders. <p>
   *
   * This is not used internally, and is kept only for interested
   * clients.
   */

  public static int totalLinesRead = 0;

  /**
   * The current line number; that is, the number of <newlines>
   * we've read from our stream so far + 1. <p>
   *
   * (Line numbers are counted from 1 not 0.) <p>
   */
  //@ invariant 1 <= curLineNo;

  protected int curLineNo = 1;

  /**
   * The value of curLineNo at the mark point (if marked is true). <p>
   *
   * The justification for why it's okay to place the following invariant
   * in this class, even though <code>marked</code> is declared in the
   * superclass, is that this class overrides the methods that set
   * <code>marked</code> to <code>true</code>.  (But there's no mechanical
   * check that this justification is upheld, so it needs to be upheld
   * manually.)
   */
  //@ invariant marked ==> 0 < markLineNo && markLineNo <= curLineNo;
  
  //@ spec_public
  private int markLineNo; /*@ readable markLineNo if marked; */

  public void mark() {
    markLineNo = curLineNo;
    super.mark();
  }

  public void reset() throws IOException {
    curLineNo = markLineNo;
    super.reset();
  }

  /* ************************************************
   *                                                 *
   * Misc:                                           *
   *                                                 *
   * ************************************************/

  public /*@non_null*/String toString() {
    StringBuffer r = new StringBuffer("LocationManagerCorrelatedReader: \"");
    r.append(getFile().getHumanName());
    r.append("\" ");

    if (buf == null) {
      r.append("closed");
    } else {
      r.append("buf[");
      for(int i=curNdx; i<endBufNdx; i++ )
	r.append( ""+(char)buf[i] );
      r.append("] "+marked);
    }

    return r.toString();
  }

  static public void clear() {
	freeLoc = STARTFREELOC;
	totalLinesRead = 0;
	allCorrStreams = new Vector();
  }
}
