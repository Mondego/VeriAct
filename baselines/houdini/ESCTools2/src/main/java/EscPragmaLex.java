/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.parser;

import java.io.IOException;

import javafe.ast._SpecialParserInterface;
import javafe.parser.Lex;
import javafe.parser.PragmaParser;
import javafe.util.ErrorSet;
import escjava.ast.TagConstants;

/**
 * This lexer overrides {@link Lex#scanJavaExtensions(int)} to
 * support "informal predicates".  An informal predicate is an
 * embedded comment. <br>
 * <pre>
 * //@ requires (* this is an informal predicate *);
 * //@ requires true &&(*soIsThis*)
 * //@ ensures (((((* Here's one inside some parens *)))));
 * </pre>
 * Informal predicates can even be more crazy with embedded comments.
 * See the file ESCTools/Escjava/test/escjava/test/test70/C.java for
 * examples.
 *
 * <p> Informal predicates are parsed as boolean literal expressions
 * with an AST decoration of "informalPredicate".  See
 * {@link EscPragmaParser#parsePrimaryExpression(Lex)} and
 * {@link EscPragmaParser#informalPredicateDecoration} for details.
 */

public final class EscPragmaLex extends Lex
{
  // Documented in parent class.
  public EscPragmaLex(PragmaParser pragmaParser, boolean isJava) {
    super(pragmaParser, isJava);
  }

  // Documented in parent class.
  //@ also
  //@ assignable endingLoc, identifierVal, text, textlen;
  protected int scanJavaExtensions(int nextchr) {
    try { 
      if (nextchr == '\\') {
        int h = 0;
        do {
          try {
            text[textlen] = (char)nextchr;
            textlen++;
          } catch (ArrayIndexOutOfBoundsException e) { 
            this.append(nextchr);
          }
          h = _SpecialParserInterface.HC * h + nextchr;
          nextchr = m_in.read();
        } while (Character.isJavaIdentifierPart((char)nextchr));
        m_nextchr = nextchr;
        auxVal = null;
        identifierVal = _SpecialParserInterface.intern(text, textlen, h);
        if (keywords != null) {
          Object val = keywords.get(identifierVal);
          if (val != null) {
            ttype = ((Integer)val).intValue();
          } else {
            int esckeytag = TagConstants.fromIdentifier(identifierVal);
            if (TagConstants.isKeywordTag(esckeytag)) {
              ttype = _SpecialParserInterface.getTokenType(identifierVal);
            } else {
              ErrorSet.fatal(startingLoc, "Unrecognized special keyword");
            }
          }
        } else {
          ttype = TagConstants.NULL;
        }
        return ttype;
      } else if (nextchr == '(') {
        // Try to lex an informal predicate
        m_in.mark();
        nextchr = m_in.read();
        if (nextchr == '*') {
          // now look for the closing "*)"
          while (true) {
            nextchr = m_in.read();
            try {
              text[textlen] = (char)nextchr;
              textlen++;
            } catch (ArrayIndexOutOfBoundsException e) { 
              this.append(nextchr);
            }
            if (nextchr == '*') {
              // read all the '*'s there are
              do {
                nextchr = m_in.read();
                try {
                  text[textlen] = (char)nextchr;
                  textlen++;
                } catch (ArrayIndexOutOfBoundsException e) { 
                  this.append(nextchr);
                }
              } while (nextchr == '*');
              if (nextchr == ')') {
                // we've found the end of the informal predicate
                ttype = TagConstants.INFORMALPRED_TOKEN;
                m_nextchr = nextchr;
                auxVal = new String(text, 0, textlen-2);
                endingLoc = m_in.getLocation();
                m_nextchr = m_in.read();
                return ttype;
              }
            }
            if (nextchr == -1) {
              // error in input
              ErrorSet.fatal(startingLoc, 
                             "Unterminated informal predicate");
            }
          }
        } else {
          // it wasn't an informal predicate after all
          m_in.reset();
        }
      }

      ttype = TagConstants.NULL;
      return ttype;
    } catch (IOException e) {
      ErrorSet.fatal(m_in.getLocation(), e.toString());
      return TagConstants.NULL; // Dummy
    }
  }
} // end of class EscPragmaLex

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */

