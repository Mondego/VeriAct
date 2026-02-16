/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

import javafe.ast.Identifier;
import javafe.ast.LexicalPragma;
import javafe.ast.ModifierPragma;
import javafe.ast.TypeModifierPragma;
import javafe.ast.StmtPragma;
import javafe.ast.TypeDeclElemPragma;
import javafe.ast.PrettyPrint;

import javafe.util.Assert;
import javafe.util.Location;

//@ model import javafe.tc.TagConstants;

/**
 * The <code>Token</code> class defines a set of fields that describe
 * lexical tokens.
 *
 * <p> Allocating an individual <code>Token</code> object for every
 * token consumes a lot of memory.  Instead, our front end tries to
 * reuse <code>Token</code> instances as much as possible.  Thus, for
 * example, <code>Lex</code> is a subclass of <code>Token</code>;
 * <code>Lex</code> returns information about the current token not by
 * allocating a new <code>Token</code> object but rather by filling in
 * the <code>Token</code> fields of itself.  To facilitate the reuse
 * of <code>Token</code> instances, <code>Token</code> has a
 * <code>copyInto</code> method.
 */

public class ParserToken
{
    /***************************************************
     *                                                 *
     * Instance fields:				       *
     *                                                 *
     **************************************************/

    /**
     * Integer code giving the kind of token.
     *
     * <p> To clear a token, set this field to Token.CLEAR and set
     * identifierVal and auxVal to null, as is done in the
     * clear() method. </p>
     *
     * @see javafe.parser.ParserTagConstants
     */
    public int ttype = CLEAR;

    //* The token code to use to clear a token; EOF for now.
    public static final int CLEAR = ParserTagConstants.EOF;

    /** Clear the current token. */
    //@ ensures ttype == CLEAR;
    //@ ensures identifierVal == null;
    //@ ensures auxVal == null;
    public void clear() {
        ttype = CLEAR;
        identifierVal = null;
        auxVal = null;
    }

    /** The location of the first character of the token. */
    //@ invariant ttype != CLEAR ==> startingLoc != Location.NULL;
    public int startingLoc;

    /**
     * The location of the last character of the token.  (This value
     * isn't "off-by-one" right now.)
     */
    //@ invariant ttype != CLEAR ==> endingLoc != Location.NULL;
    public int endingLoc;


    /**
     * Identifier represented by the token.  Must be non-null if
     * <code>ttype</code> is <TT>TagConstants.IDENT</TT>.
     */
    /*@ invariant (ttype==TagConstants.IDENT) ==> (identifierVal != null); */
    public Identifier identifierVal;

    /**
     * Auxillary information about the token.  In the case of literal
     * tokens, this field holds the value of token.  In particular,
     * if <code>ttype</code> is one of the codes on the left of the
     * following table, then <code>auxVal</code> must be an instance
     * of the class on the right:
     *
     * <center><code><table>
     *    <tr><td> TagConstants.INTLIT </td>    <td> Integer </td></tr>
     *    <tr><td> TagConstants.LONGLIT </td>   <td> Long </td></tr>
     *    <tr><td> TagConstants.FLOATLIT </td>  <td> Float </td></tr>
     *    <tr><td> TagConstants.DOUBLELIT </td> <td> Double </td></tr>
     *    <tr><td> TagConstants.STRINGLIT </td> <td> String </td></tr>
     *    <tr><td> TagConstants.CHARLIT </td>   <td> Integer </td></tr>
     *    <tr><td> TagConstants.LEXICALPRAGMA </td>
     *                                <td> LexicalPragma </td></tr>
     *    <tr><td> TagConstants.MODIFIERPRAGMA </td>
     *                                <td> ModifierPragma </td></tr>
     *    <tr><td> TagConstants.STMTPRAGMA </td> <td> StmtPragma </td></tr>
     *    <tr><td> TagConstants.TYPEDECLELEMPRAGMA </td>
     *                                <td> TypeDeclElemPragma</td></tr>
     * </table> </code> </center><p>
     *
     * For the various pragmas, <code>auxVal</code> may be
     * <code>null</code>, but for the literals it may <em>not</em>
     * be.
     */
    /*@ invariant (
     !(ttype==TagConstants.BOOLEANLIT) &&
     (ttype==TagConstants.INTLIT ==> auxVal instanceof Integer) &&
     (ttype==TagConstants.LONGLIT ==> auxVal instanceof Long) &&
     (ttype==TagConstants.FLOATLIT ==> auxVal instanceof Float) &&
     (ttype==TagConstants.DOUBLELIT ==> auxVal instanceof Double) &&
     (ttype==TagConstants.STRINGLIT ==> auxVal instanceof String) &&
     (ttype==TagConstants.CHARLIT ==> auxVal instanceof Integer) &&
     (ttype==TagConstants.LEXICALPRAGMA ==> auxVal instanceof
     LexicalPragma) &&
     (ttype==TagConstants.MODIFIERPRAGMA ==> auxVal instanceof
     ModifierPragma) &&
     (ttype==TagConstants.STMTPRAGMA ==> auxVal instanceof StmtPragma) &&
     (ttype==TagConstants.TYPEDECLELEMPRAGMA ==> auxVal instanceof
     TypeDeclElemPragma) &&
     (ttype==TagConstants.TYPEMODIFIERPRAGMA ==> auxVal instanceof
     TypeModifierPragma)
     ); */
    public Object auxVal;


    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /*
     * (This used to violate invariants ... but it should not any more.)
     *
     * NOTE: This is not a helper; we use invalid tokens in TokenQueue and
     *       Lex.savedState.
     */
    public ParserToken() {}


    /**
     * Copy all the fields of <code>this</code> into
     * <code>dst</code>.  For convenience, returns <code>dst</code>.
     */
    /*@ ensures 
	dst.ttype == ttype &&
	dst.startingLoc == startingLoc &&
	dst.endingLoc == endingLoc &&
	dst.identifierVal == identifierVal &&
	dst.auxVal == auxVal &&
	\result == dst;
    */
    public final /*@ non_null */ ParserToken copyInto(/*@ non_null @*/ ParserToken dst) {
	dst.ttype = ttype;
	dst.startingLoc = startingLoc;
	dst.endingLoc = endingLoc;
	dst.identifierVal = identifierVal;
	dst.auxVal = auxVal;
	return dst;
    }


    /***************************************************
     *                                                 *
     * Operations:				       *
     *                                                 *
     **************************************************/

    /**
     * Return a representation of <code>this</code> suitable for debug
     * output.
     */
    public String ztoString() {
        String result = Location.toFileName(startingLoc);
        if (! Location.isWholeFileLoc(startingLoc))
            result += ":" + Location.toOffset(startingLoc);
        result += ": ";
        if (ttype == ParserTagConstants.IDENT)
            result += "IDENT (" + identifierVal.toString() + ")";
        else if (ttype == ParserTagConstants.CHARLIT || ttype == ParserTagConstants.INTLIT
                 || ttype == ParserTagConstants.LONGLIT
                 || ttype == ParserTagConstants.FLOATLIT
                 || ttype == ParserTagConstants.DOUBLELIT
                 || ttype == ParserTagConstants.STRINGLIT)
            result += (PrettyPrint.inst.toString(ttype) + " ("
                       + PrettyPrint.toCanonicalString(ttype, auxVal) + ")");
        else if (ttype == ParserTagConstants.LEXICALPRAGMA
                 || ttype == ParserTagConstants.MODIFIERPRAGMA
                 || ttype == ParserTagConstants.TYPEDECLELEMPRAGMA
                 || ttype == ParserTagConstants.STMTPRAGMA)
            result += PrettyPrint.inst.toString(ttype) + " (" + auxVal.toString() + ")";
        else result += PrettyPrint.inst.toString(ttype);
        return result;
    }


    /** Check the invariants of <code>this</code>. */

    public void zzz() {
        Assert.notFalse(ttype != ParserTagConstants.IDENT
                        || identifierVal != null);
        Assert.notFalse(ttype != ParserTagConstants.INTLIT
                        || auxVal instanceof Integer);
        Assert.notFalse(ttype != ParserTagConstants.LONGLIT
                        || auxVal instanceof Long);
        Assert.notFalse(ttype != ParserTagConstants.FLOATLIT
                        || auxVal instanceof Float);
        Assert.notFalse(ttype != ParserTagConstants.DOUBLELIT
                        || auxVal instanceof Double);
        Assert.notFalse(ttype != ParserTagConstants.STRINGLIT
                        || auxVal instanceof String);
        Assert.notFalse(ttype != ParserTagConstants.CHARLIT
                        || auxVal instanceof Integer);
        Assert.notFalse(ttype != ParserTagConstants.LEXICALPRAGMA
                        || auxVal == null || auxVal instanceof LexicalPragma);
        Assert.notFalse(ttype != ParserTagConstants.MODIFIERPRAGMA
                        || auxVal == null || auxVal instanceof ModifierPragma);
        Assert.notFalse(ttype != ParserTagConstants.STMTPRAGMA
                        || auxVal == null || auxVal instanceof StmtPragma);
        Assert.notFalse(ttype != ParserTagConstants.TYPEDECLELEMPRAGMA
                        || auxVal == null || auxVal instanceof TypeDeclElemPragma);
    }
}
