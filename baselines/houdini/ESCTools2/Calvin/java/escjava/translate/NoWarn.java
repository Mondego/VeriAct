/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;


import javafe.ast.*;
import javafe.util.*;

import escjava.ast.*;
import escjava.ast.TagConstants;


/**
 ** Handles turning off warnings.
 **/

public class NoWarn {

    /***************************************************
     *                                                 *
     * Global nowarns:				       *
     *                                                 *
     ***************************************************/

  static private int chkStatus[] 
  = new int[TagConstants.LASTESCCHECKTAG - 
	   TagConstants.FIRSTESCCHECKTAG];

  static {
    setAllChkStatus(TagConstants.CHK_AS_ASSERT);
  }

  public static void setAllChkStatus(int status) {
    for (int i=TagConstants.FIRSTESCCHECKTAG;
	 i < TagConstants.LASTESCCHECKTAG; i++) {
      setChkStatus(i, status);
    }

    // We never check Free because we know they always hold, even if we
    // can't prove them:
    setChkStatus( TagConstants.CHKFREE, TagConstants.CHK_AS_SKIP );
  }

    // if this boolean is set to true, all checks will use the
    // the globalStatus check tag
    public static boolean useGlobalStatus = false;
    // this will be set to one of the three kinds of checking
    //  (CHK_AS_ASSERT/ASSUME/SKIP)
    public static int globalStatus;

  /** Sets how the check tag should be interpreted.  tag should be one
      of the CHK... constants defined in TagConstants, and status
      should be one of CHK_AS_ASSUME/CHK_AS_ASSERT/CHK_AS_SKIP */

  public static void setChkStatus( int tag, int status ) {

    Assert.notFalse( TagConstants.FIRSTESCCHECKTAG <= tag
		     && tag < TagConstants.LASTESCCHECKTAG );

    Assert.notFalse( status == TagConstants.CHK_AS_ASSUME
		     || status == TagConstants.CHK_AS_ASSERT
		     || status == TagConstants.CHK_AS_SKIP );

    chkStatus[ tag - TagConstants.FIRSTESCCHECKTAG ] = status;
  }


  /** Returns how the check tag should be interpreted.  tag should be
      one of the CHK... constants defined in TagConstants. The result
      is be one of CHK_AS_ASSUME/CHK_AS_ASSERT/CHK_AS_SKIP. */

  public static int getChkStatus( int tag ) {

    Assert.notFalse( TagConstants.FIRSTESCCHECKTAG <= tag
		     && tag <= TagConstants.LASTESCCHECKTAG );
    // Use the globalStatus if the flag is set
    if (useGlobalStatus)
	return globalStatus;
    else
	return chkStatus[ tag - TagConstants.FIRSTESCCHECKTAG ];
  }


     /**
      ** Convert a nowarn category to its tag.  Returns 0 if the String
      ** is not a valid nowarn category.
      **/
     public static int toNoWarnTag(String name) {
	for (int i = TagConstants.FIRSTESCCHECKTAG;
		i<TagConstants.LASTESCCHECKTAG;
		i++) {
	    if (TagConstants.toString(i).equals(name)
		&& i!=TagConstants.CHKFREE)
		return i;
	}

	return 0;
     }


     /*
      * The line # and streamId to nowarn before (cf. setStartLine).
      */
     private static int noWarnStreamId = -1;
     private static int startLine = -1;	// no nowarn by default

    /**
     ** Set a nowarn to ignore all lines before a given line in a given
     ** CompilationUnit.<p>
     **
     ** Future calls to this routine remove any previous nowarns
     ** established via this routine.<p>
     **
     ** Passing a line # of -1 acts as a no-op nowarn. <p>
     **/
    //@ requires cu!=null
    public static void setStartLine(int line, CompilationUnit cu) {
    	startLine = line;
	noWarnStreamId = Location.toStreamId(cu.loc);
    }


    /***************************************************
     *                                                 *
     * Registering nowarns annotations and checking    *
     * that they are legal ones.                       *
     *                                                 *
     ***************************************************/

  static private LexicalPragmaVec nowarns = LexicalPragmaVec.make();

  public static void registerNowarns( LexicalPragmaVec v ) {
    if (v!=null)
      nowarns.append(v);
  }

  /** Type checks the registered nowarn pragmas, reporting errors to ErrorSet
    * appropriately.
    **/

  public static void typecheckRegisteredNowarns() {
    for (int i = 0; i < nowarns.size(); i++) {
      LexicalPragma lp = nowarns.elementAt(i);
      if (lp instanceof NowarnPragma) {
	NowarnPragma np = (NowarnPragma)lp;
	IdentifierVec iv = np.checks;
	for (int j = 0; j < iv.size(); j++) {
	  String nowarnName = iv.elementAt(j).toString();
	  if (toNoWarnTag(nowarnName) == 0) {
	    ErrorSet.error(np.loc, "'" + nowarnName +
			   "' is not a legal warning category");
	  }
	}
      }
    }
  }


    /***************************************************
     *                                                 *
     * Comparing locations:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Is <code>loc</code> on a given line number in a given stream? <p>
     **
     ** <code>loc</code> may be <code>Location.NULL</code>, in which
     ** case <code>false</code> is returned.
     **/
    static boolean onLine(int loc, int lineNo, int streamId) {
	if (loc==Location.NULL)
	    return false;

	return (streamId == Location.toStreamId(loc))
	    && (lineNo == Location.toLineNumber(loc));
    }


    /**
     ** Is a given line # in a given stream (id) before (exclusive)
     ** the line that contains a given location?  <p>
     **
     ** If loc is Location.NULL, then no is returned. <p>
     **/
    static boolean beforeLine(int loc, int lineNo, int streamId) {
	if (loc==Location.NULL)
	    return false;

	if (Location.toStreamId(loc)!=streamId)
	    return false;

	return (Location.toLineNumber(loc) < lineNo);
    }


    /**
     ** Is a given line # in a given stream (id) between the lines that
     ** contain the two given locations (inclusive)? <p>
     **
     ** If either loc is Location.NULL, then no is returned. <p>
     **
     ** Precondition: the two locations must be from the same stream.
     **/
    static boolean inRange(int startLoc, int endLoc, int lineNo,
			   int streamId) {
	if (startLoc==Location.NULL || endLoc==Location.NULL)
	    return false;

	// Check startLoc...endLoc in streamId:
	if (Location.toStreamId(startLoc)!=streamId)
	    return false;
	Assert.notFalse(Location.toStreamId(endLoc)==streamId);

	return (Location.toLineNumber(startLoc) <= lineNo) &&
	       (lineNo <= Location.toLineNumber(endLoc));
    }


    /***************************************************
     *                                                 *
     * Check nonwarn status:			       *
     *                                                 *
     ***************************************************/

    /** Returns how the check tag should be interpreted.  tag should be
      one of the CHK... constants defined in TagConstants. The result
      is one of CHK_AS_ASSUME/CHK_AS_ASSERT/CHK_AS_SKIP. */

  public static int getChkStatus(int tag, int locUse, int locPragmaDecl)
    {
      Assert.notFalse( TagConstants.FIRSTESCCHECKTAG <= tag
		       && tag <= TagConstants.LASTESCCHECKTAG );

      // If uncommented, display the ranges of each check:
      // displayWarningRange(tag, locUse, locPragmaDecl);

      
      // Use the globalStatus if the flag is set
      if (useGlobalStatus)
	  return globalStatus;

      // Check for startLine nowarn:
      if (beforeLine(locUse, startLine, noWarnStreamId))
	  return TagConstants.CHK_AS_ASSUME;


      // check nowarns
      for( int i=0; i<nowarns.size(); i++ ) {
	LexicalPragma lp = nowarns.elementAt(i);
	if( lp instanceof NowarnPragma ) {

	  NowarnPragma np = (NowarnPragma)lp;
	  int nowarnStreamId = Location.toStreamId(np.loc);
	  int nowarnLineNo = Location.toLineNumber(np.loc);

	  if( onLine(locUse, nowarnLineNo, nowarnStreamId) ||
	      onLine(locPragmaDecl, nowarnLineNo, nowarnStreamId) )
	    {
	      // on same line
	      if( np.checks == null || np.checks.size() == 0 ) {
		// applies to all checks
		return TagConstants.CHK_AS_ASSUME;
	      }

	      // search thru listed checks
	      String chkStr = TagConstants.toString(tag);

	      for( int j=0; j<np.checks.size(); j++ )
		if( chkStr.equals( np.checks.elementAt(j).toString() ) ) {
		  // no warn on this tag
		  return TagConstants.CHK_AS_ASSUME;
		}
	    }
	}
      }

      // no line-specific nowarn
      // check general table

      return chkStatus[ tag - TagConstants.FIRSTESCCHECKTAG ];
    }


    /**
     ** Routine to display ranges of checks for debugging use:
     **/
    private static void displayWarningRange(int tag, int locUse,
					   int locPragmaDecl) {
	if (!Info.on)
	    return;

	Info.out("[Will check a possible error of type "
		+ TagConstants.toString(tag) + ":");
	
	ErrorSet.caution(locUse, "Use location:");
	
	if (locPragmaDecl!=Location.NULL) {
	  ErrorSet.caution(locPragmaDecl, "Declaration location:");
	}
	
	Info.out("]");
    }
}

