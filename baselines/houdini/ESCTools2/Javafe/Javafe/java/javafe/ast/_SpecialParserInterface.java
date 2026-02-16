/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.ast;
//@ model import javafe.tc.TagConstants;

/** <CODE>_SpecialParserInterface</CODE> is not a class that should be
used by general clients of the <CODE>javafe.ast</CODE> package.  It
holds a number of routines and constants used by the
<CODE>javafe.parser</CODE> package that we want to hide from other,
more general clients of <CODE>javafe.ast</CODE>.

<CODE>_SpecialParserInterface</CODE> serves two purposes:

<UL>

<LI> It defines two constants, <CODE>DUMMYLOC</CODE> and
<CODE>IDENT</CODE> common to both packages, breaking what would
otherwise be a mutual recursion between the two packages.  This lets
<CODE>javafe.ast</CODE> be completely independent of
<CODE>javafe.parser</CODE>.

<P> <LI> It defines a "friend" interface to the
<CODE>Identifier</CODE> class, allowing the scanner to have a
specialized interface to <CODE>Identifier</CODE> without exposing that
interface to more general clients.

</UL>

<P> The "friend" interface to <CODE>Identifier</CODE> exposed by
<CODE>_SpecialParserInterface</CODE> includes a extra integer field of
<CODE>Identifier</CODE> objects used to hold token types.  For most
instances of <CODE>Identifier</CODE>, this hidden field holds the
value <CODE>IDENT</CODE>, the token code for identifiers.  However,
for an <CODE>Identifier</CODE> object associated with a keyword, the
hidden field holds the token type for the keyword.  During its
initialization, the scanner class interns all keywords and uses the
<CODE>_SpecialParserInterface.setTokenType</CODE> to write the
appropriate token-type values into the hidden fields of the resulting
<CODE>Identifier</CODE> objects.  Note that no synchronization is done
by <CODE>_SpecialParserInterface</CODE> when reading and writing these
fields.

*/


public abstract class _SpecialParserInterface {

    /**
     * Return the hidden "token type" field of <CODE>id</CODE>.
     *
     * The token code will be one that does not require a
     * non-null auxVal (cf. Token.auxVal).
     */
    //@ ensures \result != TagConstants.BOOLEANLIT;
    //@ ensures \result != TagConstants.INTLIT;
    //@ ensures \result != TagConstants.LONGLIT;
    //@ ensures \result != TagConstants.FLOATLIT;
    //@ ensures \result != TagConstants.DOUBLELIT;
    //@ ensures \result != TagConstants.STRINGLIT;
    //@ ensures \result != TagConstants.CHARLIT;
    //@ ensures \result != TagConstants.LEXICALPRAGMA;
    //@ ensures \result != TagConstants.MODIFIERPRAGMA;
    //@ ensures \result != TagConstants.STMTPRAGMA;
    //@ ensures \result != TagConstants.TYPEDECLELEMPRAGMA;
    //@ ensures \result != TagConstants.TYPEMODIFIERPRAGMA;
    public static int getTokenType(/*@ non_null @*/ Identifier id) {
	return id.tokenType;
    }


    /**
     * Set the hidden "token type" field of <CODE>id</CODE>.
     *
     * The token code must be one that does not require a
     * non-null auxVal (cf. Token.auxVal).
     */
    //@ requires tokenType != TagConstants.BOOLEANLIT;
    //@ requires tokenType != TagConstants.INTLIT;
    //@ requires tokenType != TagConstants.LONGLIT;
    //@ requires tokenType != TagConstants.FLOATLIT;
    //@ requires tokenType != TagConstants.DOUBLELIT;
    //@ requires tokenType != TagConstants.STRINGLIT;
    //@ requires tokenType != TagConstants.CHARLIT;
    //@ requires tokenType != TagConstants.LEXICALPRAGMA;
    //@ requires tokenType != TagConstants.MODIFIERPRAGMA;
    //@ requires tokenType != TagConstants.STMTPRAGMA;
    //@ requires tokenType != TagConstants.TYPEDECLELEMPRAGMA;
    //@ requires tokenType != TagConstants.TYPEMODIFIERPRAGMA;
    public static void setTokenType(/*@ non_null @*/ Identifier id,
				    int tokenType) {
	id.tokenType = tokenType;
    }


  /** Intern a sequence of characters with a pre-computed hashcode.

    <BR> Requires: <CODE>hashcode = hash(text, textlen)</CODE> 

    <P> Ensures: returns the <CODE>Identifier</CODE> instance
    associated with the symbol consisting of the first
    <CODE>textlen</CODE> characters of <CODE>text</CODE>. */

  //@ requires text != null;
  //@ requires 0 <= textlen && textlen <= text.length;
  //@ ensures \result != null;
  public static Identifier intern(char[] text, int textlen, int hashcode) {
    return Identifier.intern(text, textlen, hashcode);
  }

  /** Constant used for hashing.  The hash function used to hash a
    <i>n</i>-character identifier <code>idn</code> is to sum
    <code>HC</code>^<i>(n-(i+1))</i><code>*idn[</code><i>i</i><code>]</code>.
    Note that this is the same hash function used by Java 1.1 for
    hashing <code>String</code> objects.  */
  public static final int HC = Identifier.HC;

  /** Return the hash code used by <CODE>Identifier</CODE> for a given
    sequence of characters. */

  //@ requires text != null;
  //@ requires 0 <= textlen && textlen < text.length;
  public static int hash(char[] text, int textlen)
  { return Identifier.hash(text, textlen); }
}
