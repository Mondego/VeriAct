/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

import javafe.util.CorrelatedReader;

/**
 * <code>PragmaParser</code> objects are called by <code>Lex</code>
 * objects to parse pragmas out of pragma-containing comments.  They
 * are also called to see if a comment contains pragmas in the first
 * place.  See <code>Lex</code> for more details.
 *
 * <p> Pragmas are described using <code>Token</code> objects.  The
 * <code>ttype</code> field of a pragma token must be one of
 * <code>TagConstants.LEXICALPRAGMA</code>,
 * <code>TagConstants.MODIFIERPRAGMA</code>,
 * <code>TagConstants.STMTPRAGMA</code>, or
 * <code>TagConstants.TYPEDECLELEMPRAGMA</code>; the
 * <code>auxVal</code> field must be filled have a type according to
 * the table in <code>Token</code>.
 *
 * @todo These methods need JML specifications.
 *
 * @see javafe.parser.ParserToken
 * @see javafe.parser.Lex
 */

public interface PragmaParser
{
    /**
     * Decide whether a comment contains pragmas.  When it encounters a
     * comment, a <code>Lex</code> object passes the first character of
     * the comment to the <code>checkTag</code> method of its
     * <code>PragmaParser</code>.  If this call returns false, the
     * comment is assumed to contain no pragmas and thus is discarded.
     * If this call returns true, the comment may contain pragmas, so it
     * is converted into a <code>CorrelatedReader</code> and passed to
     * <code>restart</code>.  (The <code>tag</code> argument is either a
     * <code>char</code> or <code>-1</code>; <code>-1</code> indicates
     * the empty comment.) 
     */
    //@ requires -1 <= tag && tag <= 127;
    boolean checkTag(int tag);

    /**
     * @todo Need to write documentation for this method.
     */
    javafe.ast.FieldDecl isPragmaDecl(/*@ non_null @*/ ParserToken l);

    /**
     * Restart a pragma parser on a new input stream.  If
     * <code>this</code> already opened on another
     * <code>CorrelatedReader</code>, closes the old reader.

     * <p> <code>eolComment</code> is true to indicate that the
     * correlated reader stream is reading from a Java comment that
     * begins with "//" as opposed to a Java comment that begins with
     * "/*". </p>
     */
    void restart(/*@ non_null @*/ CorrelatedReader in, boolean eolComment);

    /**
     * Parse the next pragma.  If none are left, returns
     * <code>false</code>; otherwise, returns <code>true</code> and
     * updates fields of <code>destination</code> to describe the
     * pragma.  When <code>false</code> is returned, the pragma parser
     * is expected to close the underlying <code>CorrelatedReader</code>
     * and in other ways clean up resources.
     *
     * <p> This method requires that the <code>PragmaParser</code> is
     * "open;" that is, <code>restart</code> has been called and, since
     * the last call to <code>restart</code>, <code>getNextPragma</code>
     * has not returned false and <code>close</code> has not been
     * called. 
     */
    boolean getNextPragma(/*@ non_null @*/ ParserToken destination);

    /**
     * Stop parsing the current reader.  Sometimes a <code>Lex</code>
     * object will be stopped before its associated
     * <code>PragmaParser</code> has finished reading all pragmas out of
     * a comment (for example, when <code>Lex.restart</code> is called).
     * In this case, the <code>Lex</code> object calls
     * <code>PragmaParser.close</code>.  This method should close the
     * underlying <code>CorrelatedReader</code> and in other ways clean
     * up resources. 
     */
    void close();
}
