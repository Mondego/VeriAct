/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import javafe.genericfile.*;


/** 
 * A <I>location</I> is an integer that identifies the position in a
 * file of a particular piece of program source text.  This class is
 * never instantiated. It contains static functions for manipulating
 * locations. Locations are produced by the
 * <code>CorrelatedReader</code> class (and by this class), they are
 * stored in ASTs, and they are passed to <code>ErrorSet</code> to
 * identify the program source that caused a particular error or
 * warning.

 * <p> There are three kinds of locations.
 * <ol>
 * <li> <i>Regular locations</i> encode the file, line, column, and
 * offset of a character read by <code>CorrelatedReader</code>.  A
 * call to the <code>getLocation()</code> method of a
 * <code>CorrelatedReader</code> object returns the regular location
 * of the last character read from that <code>CorrelatedReader</code>
 * object.  The file/line/column/offset of that location can be
 * extracted via the methods described below.

 * <br> Following emacs, line numbers begin at 1, column numbers at 0,
 * and offsets at 1.  A newline character is considered the last
 * character on a line.

 * <li> <i>Whole file locations</i> encode just file information.
 * They are currently used for error messages that are global to a
 * file (eg. the error message given when a file that is expected to
 * contain a package declaration does not have one). We expect they
 * will also be used in ASTs produced from class files (since there is
 * no meaningful line or column information, and offset information
 * would not be useful in an error message).

 * <li><i>The null location</i> is a constant that plays a similar
 * role for locations that null plays for reference types.
 * </ol>
 *
 * <p> Interface reviewed at Sparta Meeting 1/8/97.
 *
 * @see javafe.util.CorrelatedReader
 * @see javafe.util.ErrorSet 
 */

public class Location
{
  //@ public model JMLDataGroup internalState;

  /** Private constructor. Never called. */

  private Location() {}


  /** The null location, is the constant 0. */

  public static final int NULL = 0;


  /**********************************************************************
   * Check if a location is a whole file location.

   * <p>Precondition: loc should be a valid location (ie a regular,
   * whole file, or dummy location).
   *********************************************************************/

    //@ pure
    public static boolean isWholeFileLoc(int loc) {
	return LocationManagerCorrelatedReader.isWholeFileLoc(loc);
    }


  /**********************************************************************
   * Extracts the file corresponding to a location. 

   * <p>Precondition: loc should be a regular location or a whole file
   location.
   *********************************************************************/

    //@ requires loc != Location.NULL;
    //@ modifies \nothing;
    //@ ensures \result != null;
    public static GenericFile toFile(int loc) {
	return LocationManagerCorrelatedReader.locToFile(loc);
    }


  /**********************************************************************
   * Extracts the filename corresponding to a location. 

   * <p>Precondition: loc should be a regular location or a whole file
   location.
   *********************************************************************/

    //@ requires loc != Location.NULL;
    //@ modifies \nothing;
    //@ ensures \result != null;
    public static String toFileName(int loc) {
	return LocationManagerCorrelatedReader.locToFile(loc).getHumanName();
    }


  /**********************************************************************
   * Extracts the offset corresponding to a location.

   * The first character in a stream is at offset 1.

   * <p>Precondition: loc should be a regular location.
   *********************************************************************/

    //@ requires loc != Location.NULL;
    //@ modifies \nothing;
    public static int toOffset(int loc) {
	return LocationManagerCorrelatedReader.locToOffset(loc);
    }

  /**********************************************************************
   * Extracts the line number corresponding to a location.  

   * The first line in a stream is numbered 1.

   * <p>Precondition: loc should be a regular location.
   *********************************************************************/

    //@ requires loc != Location.NULL;
    //@ modifies \nothing;
    //@ ensures \result >= 1;
    public static int toLineNumber(int loc) {
	return LocationManagerCorrelatedReader.locToLineNumber(loc);
    }

  /**********************************************************************
   * Extracts the column corresponding to a location.  

   * The first column on each line is numbered 0.

   * <p>Precondition: loc should be a regular location.
   *********************************************************************/

    //@ requires loc != Location.NULL;
    //@ modifies \nothing;
    //@ ensures \result >= 0;
    public static int toColumn(int loc) {
	return LocationManagerCorrelatedReader.locToColumn(loc);
    }

  /**********************************************************************
   * Convert a location into a printable description suitable for use
   * in a warning or error message.

   * <p>Precondition: loc should be a valid location (ie a regular,
   * whole file, or dummy location).
   *********************************************************************/

  public static /*@non_null*/String toString(int loc) {

    if( loc == NULL )
      return "<dummy location>";

    String name = "\"" + toFileName(loc) + "\"";

    if (isWholeFileLoc(loc))
      return name;
    
    return name + ", line " + toLineNumber(loc)
   	         +", col "  + toColumn(loc);
  }

  public static String toFileLineString(int loc) {
    String s = Location.toFileName(loc);
    if (!Location.isWholeFileLoc(loc))
	s = s + ":" + Location.toLineNumber(loc);
    return s;
  }

    /**********************************************************************
     * Create a whole file location corresponding to the given GenericFile.
     * Calls to <code>toFile</code> on that location will return
     * this file.
     *********************************************************************/

    //@ ensures \result != Location.NULL;
    public static int createWholeFileLoc(/*@ non_null @*/ GenericFile file) {
	return FileCorrelatedReader.createWholeFileLoc(file);
    }


    /**
     * Create a fake location described by description.<p>
     *
     * This should only be used by debugging code and in places where
     * it should never see the light of day.
     *
     * The resulting location is a whole-file location associated
     * with an unopenable file with human-name description.
     */
    //@ ensures \result != Location.NULL;
    public static int createFakeLoc(/*@ non_null @*/ String description) {
	return FileCorrelatedReader.createWholeFileLoc(
		   new UnopenableFile(description));
    }



    //@ requires 0 <= streamId;
    //@ requires streamId < LocationManagerCorrelatedReader.allCorrStreams.elementCount;
    //@ requires line > 0;
    //@ requires col >= 0;
    //@ ensures \result != Location.NULL  ;
    public static int make(int streamId, int line, int col) {
	return LocationManagerCorrelatedReader.makeLocation(streamId, line, col);
    }

    /**
     * Attempts to return a location <code>n</code> characters further
     * to the right of <code>loc</code> on the same line.  Returns the
     * same location if it is not a regular location.
     *
     * Produces an assertion failure if that location does not exist
     * (e.g., the line is too short.).
     */
    //@ requires n >= 0;
    //@ ensures loc != NULL ==> \result != NULL;
    public static int inc(int loc, int n) {
	if (isWholeFileLoc(loc) || loc==NULL)
	    return loc;

	// Should be a regular location here:
        // This assertion is commented out because when we translate
        // Java's assert construct into a conditional throws clause,
        // under some circumstances, the translated (fictional) IfStmt
        // does not actually fit on the original line.  E.g.,
        //   assert false;
        // becomes
        // if !false throw new java.lang.AssertionError();
        // which obviously is longer than the original statement.
	// Assert.notFalse(toLineNumber(loc) == toLineNumber(loc+n)); //@ nowarn pre
	return loc+n;
    }                        //@ nowarn Post;


  /**
    * Returns the internal stream ID used for the stream associated
    * with location <code>loc</code>.
    */
    //@ requires loc != Location.NULL;
  public static int toStreamId(int loc) {
    return LocationManagerCorrelatedReader.locToStreamId(loc);
  }

    /**
     * Returns the file associated with stream id <code>id</code>,
     * where <code>id</code> has previously been returned by
     * <code>toStreamId</code>.
     */
    //@ requires 0 <= id && id < LocationManagerCorrelatedReader.allCorrStreams.elementCount;
    public static GenericFile streamIdToFile(int id) {
	return LocationManagerCorrelatedReader.streamIdToFile(id);
    }
}
