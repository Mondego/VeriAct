/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

import javafe.util.Assert;
import javafe.util.Location;
import java.util.Hashtable;

import javafe.ast.*;
import escjava.tc.Types;
import escjava.ast.TagConstants;


/**
 ** This class provides methods for unique-ifying names, as described
 ** in ESCJ 23b, "Unique names in ESC/Java".
 **
 ** Methods <code>suffixToString</code> and <code>suffixToLoc</code> call
 ** private method <code>parseSuffix</code> whose out-parameters are stored
 ** in some static fields of this class, these methods are not thread safe.
 ** (To make them thread safe, these numbers could be returned in a
 ** newly allocated array.)
 **/

public final class UniqName {
  
  /**********************************************************************
   * Sets the "<em>default suffix file</em> to be that of location
   * <code>loc</code>, or to none if <code>loc</code> is the null location.
   * The default suffix file is used in the converting of a location to
   * a suffix and back.
   **********************************************************************/

  private static int idDefaultSuffixFile = -1;
  
  public static void setDefaultSuffixFile(int loc) {
      resetUnique();        // Make sure changed names don't cause problems

      if (loc == Location.NULL)
	  idDefaultSuffixFile = -1;
      else
	  idDefaultSuffixFile = Location.toStreamId(loc);
  }
  
  /**********************************************************************
   * Convert a location into a printable string suitable for use as a
   * suffix in unique-ifying a name declared at <code>loc</code>.

   * <p>Precondition: <code>loc</code> should be a valid non-null location.
   * <p>Postcondition:  <code>\result != null</code>
   **********************************************************************/

  //@ requires loc != Location.NULL;
  //@ ensures \result != null;
  public static String locToSuffix(int loc) {
    Assert.notFalse(loc != Location.NULL);

    int streamID = Location.toStreamId(loc);
    if (Location.isWholeFileLoc(loc))
      return streamID + "..";

    String suffix = Location.toLineNumber(loc) + "." + Location.toColumn(loc);
    if (streamID == idDefaultSuffixFile && !escjava.Main.guardedVC)
      return suffix;
    else
      return streamID + "." + suffix;
  }

  /** Returns the location corresponding to <code>suffix</code>.
    * Requires <code>suffix</code> to be a valid suffix.
    **/
  public static int suffixToLoc(String suffix) {
    return suffixToLoc(suffix, false);
  }

  /** Returns the location corresponding to <code>suffix</code>, if any,
    * and the null location if <code>suffix</code> is not a valid suffix.
    * Requires <code>recoverable</code> or that <code>suffix</code> is
    * a valid suffix.
    **/
  public static int suffixToLoc(String suffix, boolean recoverable) {
    if (parseSuffix(suffix, recoverable)) {
      return Location.make( psout0, psout1, psout2 );
    } else {
      return Location.NULL;
    }
  }

  /**********************************************************************
   * Convert a location suffix string into a printable string that
   * describes the location.
   *
   * <p>Precondition: <code>suffix</code> should be a string previously
   * returned by <code>locToSuffix</code>, and the default suffix
   * file should be the same as it was when the corresponding
   * <code>locToSuffix</code> was called.
   *
   * Not thread safe.
   **********************************************************************/

  public static String suffixToString(String suffix) {
    parseSuffix(suffix, false);
    String s = Location.streamIdToFile(psout0).getHumanName();
    if (psout1 == 0)
      return s + "(offset ?)";
    else
      return s + "(" + psout1 + "," + psout2 + ")";
  }

  private static int psout0;
  private static int psout1;
  private static int psout2;

  /** Parses <code>suffix</code>, which is expected to have one of the forms
    * <ul>
    * <li>  Number "." Number "." Number
    * <li>  Number "." Number
    * <li>  Number ".."
    * </ul>
    * The first form indicates a stream ID, a line number (1-based), and
    * and column number (0-based).
    * The second form indicates a line number and a column number, where
    * the implicit stream ID number is <code>idDefaultSuffixFile</code>.
    * The third form indicates a stream number, where the location that
    * was converted into this suffix string was a whole-file location
    * (this occurs if the location refers to a place in a .class file). <p>
    *
    * This method uses <code>psout0</code>, <code>psout1</code>, and
    * <code>psout2</code> as out parameters.  The will contain the values
    * of the stream ID, line number, and column number of the location parsed
    * from the suffix, or the stream ID and two 0's if the suffix has
    * the third form. <p>
    *
    * If <code>recoverable</code> is <code>true</code>, then this method
    * can be called even if <code>suffix</code> is not known to be a valid
    * suffix.  If it isn't, <code>false</code> is returned.  If
    * <code>suffix</code> is a valid suffix, then <code>true</code> is
    * returned. <p>
    *
    * Not thread safe (since global variables are used to contain out
    * parameters).
    **/
  
  private static boolean parseSuffix(String suffix, boolean recoverable) {
    // These numbers will be shifted as dots are discovered
    int a0 = 0;
    int a1 = idDefaultSuffixFile;
    int a2 = 0;
    
    int cDots = 0;
    for (int k = 0; k < suffix.length(); k++) {
      char ch = suffix.charAt(k);
      switch (ch) {
	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
	case '8':
	case '9':
	  a2 = a2 * 10 + (ch - '0');
	  break;
	case '.':
	  cDots++;
	  a0 = a1;  a1 = a2;  a2 = 0;
	  break;
	default:
	  if (recoverable) {
	    return false;
	  }
	  //@ unreachable
	  Assert.fail("unexpected character '" + ch +
		      "' in purported location suffix '" + suffix + "'");
	  break;
      }
    }

    if (! (cDots == 2 || (cDots == 1 && idDefaultSuffixFile != -1))) {
      if (recoverable) {
	return false;
      }
      //@ unreachable
      Assert.fail("wrong number of dots in purported location suffix '" +
		  suffix + "'");
    }
    
    // set out parameters
    psout0 = a0;
    psout1 = a1;
    psout2 = a2;

    return true;
  }


    /**
     ** Returns the unique-ification of the type <code>t</code>. <p>
     **
     ** Handles case 7 of ESCJ 23b. <p>
     **/
    //@ ensures \result != null;
    public static String type(/*@ non_null */ Type t) {
	/*
	 * Special case primitive type TYPE to avoid confusion with any
	 * reference type named "TYPE":
	 */
	if (t instanceof PrimitiveType
		&& ((PrimitiveType)t).tag==TagConstants.TYPECODE)
	    return "T_.TYPE";

	return "T_" + Types.printName(t);
    }


    /***************************************************
     *                                                 *
     * Producing names for variables:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Use this location *only* in declarations of special variables
     ** (see case 4 of ESCJ 23b for the complete list of these).
     **/
    public static final int specialVariable =
		Location.createFakeLoc("<special variable>");

    /**
     ** Use this location *only* in declarations of temporary variables
     ** (see case 6 of ESCJ 23b for rules on the names allowed for these).
     **/
    public static final int temporaryVariable =
		Location.createFakeLoc("<temporary variable>");

    /**
     ** Use this location *only* in declarations of automatically
     ** generated bound variables.  (See case 3 of ESCJ 23b for rules on
     ** the names allowed for these).
     **/
    private static final int autoBoundVariable =
		Location.createFakeLoc("<bound variable>");


    /**
     ** Returns a new bound variable for use in a quantificiation, where
     ** we do not wish to or cannot associate the variable with an
     ** existing VariableAccess. <p>
     **
     ** See newBoundThis() for an important exceptional case, though.<p>
     **
     ** The resulting name will be of the form "<prefix>".<p>
     **
     ** (This partially handles case 3 of ESCJ 23b.) 
     **/
    public static LocalVarDecl newBoundVariable(char prefix) {
	return newBoundVariable(String.valueOf(prefix));
    }

    /**
     ** Returns a new bound variable for use in quantifying over "this" in
     ** an invariant. <p>
     **
     ** The resulting name will be of the form "brokenObj".
     **
     ** (This partially handles case 3 of ESCJ 23b.) 
     **/
    public static LocalVarDecl newBoundThis() {
	return newBoundVariable("brokenObj");
    }

    /**
     ** Private routine to create a new bound variable for use in a
     ** quantificiation, where we do not wish to or cannot associate the
     ** variable with an existing VariableAccess. <p>
     **
     ** This handles case 3 of ESCJ 23b.
     **/
    public static LocalVarDecl newBoundVariable(String name) {
	Identifier id = Identifier.intern(name);
	return LocalVarDecl.make(Modifiers.NONE,  // Java modifiers
	 		         null,            // pragma modifiers
			         id,
			         Types.anyType,
			         autoBoundVariable,
			         null, Location.NULL); // initializer
    }

    public static LocalVarDecl newBoundVariable(String name, Type t) {
	Identifier id = Identifier.intern(name);
	return LocalVarDecl.make(Modifiers.NONE,  // Java modifiers
	 		         null,            // pragma modifiers
			         id,
			         t,
			         autoBoundVariable,
			         null, Location.NULL); // initializer
    }


    /**
     ** Returns a new intermediate-state variable
     ** associated with an existing VariableAccess. <p>
     **
     ** The resulting name will be of the form
     ** <last segment of name>:<location>, where <location> is the
     ** location of the variable reference in the given VariableAccess
     ** if available, and the location of the variable declaration
     ** otherwise. <p>
     **
     ** This handles case 15 of ESCJ 23b.
     **/
    //@ ensures \result != null;
    //@ ensures \result.id == v.id;
    public static LocalVarDecl newIntermediateStateVar(/*@ non_null */ VariableAccess v,
						       /*@ non_null */ String suffix) {
	int loc = v.loc;
	if (loc==Location.NULL)
	    loc = v.decl.locId;

	Identifier id;
	if (suffix.length() == 0) {
	    id = v.id;
	} else {
	    id = Identifier.intern(v.id.toString() + suffix);
	}
	return LocalVarDecl.make( Modifiers.NONE,
				  null,
				  id,
				  v.decl.type,
				  loc,
				  null, Location.NULL );
    }

    //@ ensures \result != null;
    //@ ensures \result.id == v.id;
    public static LocalVarDecl newIntermediateStateVar(/*@ non_null */ GenericVarDecl vd,
						       /*@ non_null */ String suffix) {
	int loc = vd.locId;

	Identifier id;
	if (suffix.length() == 0) {
	    id = vd.id;
	} else {
	    id = Identifier.intern(vd.id.toString() + suffix);
	}
	return LocalVarDecl.make( Modifiers.NONE,
				  null,
				  id,
				  vd.type,
				  loc,
				  null, Location.NULL );
    }


    /**
     ** Returns the unique-ification of the variable <code>v</code>. <p>
     **
     ** Handles cases 3, 4, 6, 10, 11, 12, 13 and 15 of ESCJ 23b. <p>
     **/
    public static String variable(GenericVarDecl v) {
	String s = v.id.toString();
	int loc = v.locId;

	if (loc==autoBoundVariable) {
	    // Case 3:
	    return verifyUnique(v,s);
	} else if (loc==specialVariable) {
	    // Case 4, (part of) 13:
	    return verifyUnique(v,s);
	} else if (loc==temporaryVariable) {
	    // Case 6:
	    return verifyUnique(v,s);
	} else if (loc != Location.NULL) {
	    // Cases 10, 11, 12, (part of) 13, 15:
	    // All of these produce '<last segment of name>:<location>':
	    return verifyUnique(v, s+":"+locToSuffix(loc));
	} else {
	    // This shouldn't happen...
	    // Assert.fail("GenericVarDecl with NULL location: "+ v.id);

	    return verifyUnique(v,s+":unknown");
	}
    }


    /***************************************************
     *                                                 *
     * Ensuring unique names for variables:	       *
     *                                                 *
     ***************************************************/

    private static Hashtable mapObjStr = new Hashtable();
    private static Hashtable mapStrObj = new Hashtable();

    /**
     ** Reset the <n>-uniqueness-ensuring mechanism.
     **/
    public static void resetUnique() {
	mapObjStr = new Hashtable();
	mapStrObj = new Hashtable();
    }


    private static String verifyUnique(GenericVarDecl o,String s) {
	String s2 = (String)mapObjStr.get(o);
	if( s2 != null ) {
	    // Mapping already initialized, use it
	    return s2;
	} else {
	    // Mapping not initialized, so initialize it
	    // Prefer s, if not already used

	    String s3 = s.intern();
	    int i=0;
	    
	    for(;;) {
		if( mapStrObj.get(s3) == null ) {
		    // s does not clash with existing name
		    mapObjStr.put( o, s3 );
		    mapStrObj.put( s3, o );
		    return s3;
		}
		// Increment i, and try new postfix
		i++;
		s3 = (s+"<"+i+">").intern();
	    }
	}
    }
	
}
