/* Copyright Hewlett-Packard, 2002 */

package escjava.parser;

import java.io.IOException;
import javafe.util.CorrelatedReader;
import javafe.util.FilterCorrelatedReader;
import javafe.util.Assert;


/** 
 ** This FilterCorrelatedReader creates the illusion that the additional
 ** @-signs, etc. allowed in the JML annotation syntax are really just
 ** whitespace.
 **
 ** @see      javafe.util.FilterCorrelatedReader
 ** @author   Rustan Leino
 **/

public class JmlCorrelatedReader extends FilterCorrelatedReader {

  /* *************************************************
   *                                                 *
   * Creation:                                       *
   *                                                 *
   * *************************************************/

  public static final int EOL_COMMENT = 0;
  public static final int C_COMMENT = 1;
  public static final int JAVADOC_COMMENT = 2;
  public static final int COMMENTS_KINDS = 3;

  /** Constructs a JmlCorrelatedReader with <code>child</code> as the
   ** underlying CorrelatedReader.  After calling this constructor, the
   ** caller should no longer use <code>child</code> directly. <p>
   **
   ** <code>commentKind</code> is <code>EOL_COMMENT</code> to
   ** indicate a slash-slash comment, <code>C_COMMENT</code> to
   ** indicate an ordinary slash-star comment, and
   ** <code>JAVADOC_COMMENT</code> to indicate that
   ** the portion to be read resides inside a (pair of tags in a) Javadoc
   ** comment.
   **/

  //@ requires 0 <= commentKind && commentKind < COMMENTS_KINDS;
  protected JmlCorrelatedReader(/*@ non_null */ CorrelatedReader child,
				int commentKind) {
    super(child);
    switch (commentKind) {
      case EOL_COMMENT:
	prefixMode = 2; // disallow initial @-signs
	specialCharacter = '@';
	allowSpecialSuffix = false;
	break;
      case C_COMMENT:
	specialCharacter = '@';
	allowSpecialSuffix = true;
	break;
      default:
	Assert.precondition("invalidate value for commentKind");
	// fall-through (to please compiler)
      case JAVADOC_COMMENT:
	specialCharacter = '*';
	allowSpecialSuffix = false;
	break;
    }
  }

  /* *************************************************
   *                                                 *
   * Reading and filtering:                          *
   *                                                 *
   * *************************************************/

  /** The lines of the input consist of (0) a number of whitespace
    * characters, (1) a number of special characters ('@' in ordinary
    * comments and '*' inside Javadoc comments), and followed by (2)
    * the "real meat".  Variable <code>prefixMode</code> indicates
    * which of these three segments are currently being scanned.
    **/

  private int prefixMode = 1;  /* 0-whitespace, 1-special, 2-meat */
  //@ invariant 0 <= prefixMode && prefixMode < 3;

  /** Indicates the special character. **/

  private final int specialCharacter;

  /** This variable is included so that @-signs at the end of a
    * pragma-containing comment can be ignored.  In particular,
    * it counts the number of characters read but not reported.
    * These characters will be a number of @-signs followed by one
    * more character (or -1, if at end-of-pragma).  @-signs that
    * immediately follow the first white space characters
    * on a line are always "reported" (since these are actually
    * reported as spaces anyway).  In the steady state of this correlated
    * reader, <code>unreturnedChars</code> is non-zero only when
    * <code>lastUnreturnedChar</code> is not an @-sign. <p>
    *
    * <code>unreturnedChars</code> is not used (actually, is 0) when
    * scanning Javadoc comments.
    **/

  private int unreturnedChars = 0;
  //@ invariant 0 <= unreturnedChars;
  //@ invariant prefixMode < 2 ==> unreturnedChars == 0;
  //@ invariant specialCharacter != '@' ==> unreturnedChars == 0;

  /** If there are unreturned characters, <code>lastUnreturnedChar</code>
    * indicates the last of these characters.
    */

  private int lastUnreturnedChar /*@ readable_if unreturnedChars != 0; */;
  //@ invariant lastUnreturnedChar != '@';

  /** Indicates whether or not a sequence of consecutive special characters
    * at the end of the comment are to be treated as white space.  This is
    * only done for the special character '@'.
    **/

  private final boolean allowSpecialSuffix;
  //@ invariant allowSpecialSuffix ==> specialCharacter == '@';

  /**
   ** Reads the next character from this input stream. 
   ** Does unicode conversion and JML filtering. <p>
   **
   ** Requires we are open.<p>
   ** 
   ** @return   A unicode character, or -1.<p>
   **/
  public int read() throws IOException {
    /*@ uninitialized */ int ch = 0; // dummy assignment
    if (unreturnedChars == 0) {
      ch = child.read();
      if (ch == '\n') {
	prefixMode = 0;
      } else if (prefixMode == 0 && Character.isWhitespace((char)ch)) {
	// remain in this mode
      } else if (prefixMode < 2) {
	if (ch == specialCharacter) {
	  ch = ' ';  // treat the special character as a space
	  prefixMode = 1;
	} else {
	  prefixMode = 2;  // enter meat mode
	}
      } else if (allowSpecialSuffix && ch == specialCharacter) {
	unreturnedChars++;
	do {
	  lastUnreturnedChar = child.read();
	  unreturnedChars++;
	} while (lastUnreturnedChar == specialCharacter);
      }
    }
    if (unreturnedChars > 0) {
      if (unreturnedChars == 1) {
	ch = lastUnreturnedChar;
      } else if (lastUnreturnedChar == -1) {
	ch = ' ';
      } else {
	ch = specialCharacter;
      }
      unreturnedChars--;
    }
    return ch;
  }

  /* *************************************************
   *                                                 *
   * Getting Locations:                              *
   *                                                 *
   * *************************************************/

  public int getLocation() {
    return super.getLocation() - unreturnedChars;
  }

  /* *************************************************
   *                                                 *
   * [Un]marking:                                    *
   *                                                 *
   * *************************************************/

  private int prefixModeAtMark /*@ readable_if marked; */;
  //@ invariant 0 <= prefixModeAtMark && prefixModeAtMark <= 2;
  private int unreturnedCharsAtMark /*@ readable_if marked; */;
  //@ invariant 0 <= unreturnedCharsAtMark;
  //@ invariant prefixModeAtMark < 2 ==> unreturnedCharsAtMark == 0;
  //@ invariant specialCharacter != '@' ==> unreturnedCharsAtMark == 0;
  private int lastUnreturnedCharAtMark /*@ readable_if marked; */;
  //@ invariant lastUnreturnedCharAtMark != '@';

  /** See documentation in superclass.
   **/

  public void mark() {
    super.mark();
    prefixModeAtMark = prefixMode;
    unreturnedCharsAtMark = unreturnedChars;
    lastUnreturnedCharAtMark = lastUnreturnedChar; //@ nowarn Unreadable
  }

  /** See documentation in superclass.
   **/

  public void reset() throws IOException {
    prefixMode = prefixModeAtMark;
    unreturnedChars = unreturnedCharsAtMark;
    lastUnreturnedChar = lastUnreturnedCharAtMark;
    super.reset();
  }
}
