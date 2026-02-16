/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

import javafe.ast.*;
import javafe.util.StackVector;
import javafe.Tool;

/**
 * Parses Java types.
 * Extended by {@link javafe.parser.ParseExpr}.
 *
 * @see javafe.ast.ASTNode
 * @see javafe.parser.ParseUtil
 * @see javafe.parser.ParseExpr
 */

public class ParseType extends ParseUtil
{
    public ParseType() {
	//@ set seqIdentifier.elementType = \type(Identifier);
	//@ set seqIdentifier.owner = this;

	//@ set nameIdLocs.owner = this;
	//@ set nameDotLocs.owner = this;
    }

    //* Internal working storage for parseName function.
    //@ invariant seqIdentifier.elementType == \type(Identifier);
    //@ invariant seqIdentifier.owner == this;
    protected final /*@ non_null @*/ StackVector seqIdentifier
            = new StackVector();

    /**
     * Parse an {@link Identifier}.
     */
    //@ requires l.m_in != null;
    //@ ensures \result != null;
    public Identifier parseIdentifier(/*@ non_null @*/ Lex l) {
	if (l.ttype != ParserTagConstants.IDENT) {
            if (l.ttype == ParserTagConstants.UNKNOWN_KEYWORD)
                fail(l.startingLoc, 
                     "Expected an identifier; Unrecognized keyword");
            else 
                if (ParserTagConstants.toString(l.ttype).equals("assert")) {
                    // If the next identifier is "assert", we need to
                    // handle it in a special fashion.  Tool.options
                    // can be null if we are not running in a "real"
                    // tool, as is the case during testing with
                    // javafe.parser.test.TestParse.
                    if ((Tool.options == null) || (Tool.options.assertIsKeyword)) {
                        // if "assert" is a keyword, then it cannot be
                        // used as an identifier.
                        fail(l.startingLoc,
                             "Expected an identifier, but found the \"assert\" keyword used as an identifier.");
                    } else {
                        // if "assert" is NOT a keyword, then it is ok
                        // to use it as an identifier.
                        Identifier r = l.identifierVal;
                        l.getNextToken();
                        return r;
                    }
                }
            // Otherwise, we have seen something that is not an
            // identifier and is not the twofaced "assert", so fail.
            fail(l.startingLoc, "Expected an identifier, found " + 
                 ParserTagConstants.toString(l.ttype) );
	}
	Identifier r = l.identifierVal;
	l.getNextToken();
	return r;
    }

    /**
     * The following private ivars are used in parseName.  They cannot
     * be defined inside parseName because they would be reallocated
     * at each invocation.  They cannot be defined as static variables
     * inside parseName because then the code would not be
     * thread-safe.
     */
    //@ private invariant nameIdLocs.length >= 10;
    //@ private invariant nameIdLocs.owner == this;
    private /*@ non_null @*/ int[] nameIdLocs  = new int[10];
    //@ private invariant nameDotLocs.length == nameIdLocs.length;
    //@ private invariant nameDotLocs.owner == this;
    private /*@ non_null @*/ int[] nameDotLocs = new int[nameIdLocs.length];

    /**
     * Parse a {@link Name}.
     * <pre>
     * Name:
     * Identifier ( . Identifier ) *
     * </pre>
     */
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    public Name parseName(Lex l) {
        Identifier id = null;
        final Identifier assertIdentifier = Identifier.intern("assert");
        final boolean parsingPreAssertJava = 
            Tool.options != null && !Tool.options.assertIsKeyword;

        // If we are parsing pre-1.4 Java code and we see an assert
        // keyword, then we must convert it into a Name.
        if (parsingPreAssertJava && l.ttype == ParserTagConstants.ASSERT) {
            id = assertIdentifier;
        } else {
            // Otherwise, if we saw something that is not an identifier,
            // then it is an error.
            if (l.ttype != ParserTagConstants.IDENT) {
                if (l.ttype == ParserTagConstants.UNKNOWN_KEYWORD)
                    fail(l.startingLoc,
                         "Expected a Name, got '"+PrettyPrint.inst.toString(l.ttype)+"'" +
                         "; Unrecognized keyword");
                else
                    fail(l.startingLoc,
                         "Expected a Name, got '"+PrettyPrint.inst.toString(l.ttype)+"'");
            }
            id = l.identifierVal;
        }
        int loc = l.startingLoc;
        l.getNextToken();
        if (l.ttype != ParserTagConstants.FIELD || l.lookahead(1) != ParserTagConstants.IDENT ||
            (l.lookahead (1) == ParserTagConstants.ASSERT && !parsingPreAssertJava))
            return SimpleName.make(id, loc);

        // Deal with less common, multiple-name case...
        seqIdentifier.push();
        seqIdentifier.addElement(id);
        int stkPtr = 0;
        nameIdLocs[stkPtr] = loc;
        /*@ loop_invariant stkPtr < nameIdLocs.length &&
          @                stkPtr == (seqIdentifier.elementCount -
          @                seqIdentifier.currentStackBottom) - 1;
          @*/
        while (l.ttype == ParserTagConstants.FIELD &&
               (l.lookahead(1) == ParserTagConstants.IDENT ||
                (parsingPreAssertJava && l.lookahead(1) == ParserTagConstants.ASSERT))) {
            // need to use lookahead for "import Name . *" case
            nameDotLocs[stkPtr++] = l.startingLoc;
            l.getNextToken();  // swallow the FIELD

            if (l.ttype != ParserTagConstants.IDENT ||
                (!parsingPreAssertJava && l.ttype == ParserTagConstants.ASSERT)) { 
                if (l.ttype == ParserTagConstants.UNKNOWN_KEYWORD)
                    fail(l.startingLoc, ("Expected an identifier, got '"
                                         +PrettyPrint.inst.toString(l.ttype)+"'" +
                                         "; Unrecognized keyword"));
                else
                    fail(l.startingLoc, ("Expected an identifier, got '"
                                         +PrettyPrint.inst.toString(l.ttype)+"'"));
            }
            if (l.ttype == ParserTagConstants.ASSERT)
                seqIdentifier.addElement (assertIdentifier);
            else
                seqIdentifier.addElement (l.identifierVal);

            // Check for resizing
            if (stkPtr == nameIdLocs.length) {
                // Extend it
                int nuid[] = new int[2*nameIdLocs.length];
                //@ set nuid.owner = this;
                System.arraycopy(nameIdLocs, 0, nuid, 0, nameIdLocs.length);
                int nudot[] = new int[2*nameIdLocs.length];
                //@ set nudot.owner = this;
                System.arraycopy(nameDotLocs, 0, nudot, 0, nameIdLocs.length-1);
                nameDotLocs = nudot;
                nameIdLocs = nuid;
            }
            nameIdLocs[stkPtr] = l.startingLoc;
            l.getNextToken();
        }
        //@ assume stkPtr > 0;   // loop always runs at least once

        // Id locations in nameIdLocs[0 .. stkPtr inclusive]
        // Dot locations in nameIdLocs[0 .. stkPtr-1 inclusive]

        // Copy arrays
        int ids[] = new int[stkPtr+1];
        System.arraycopy(nameIdLocs, 0, ids, 0, stkPtr+1);
        int dots[] = new int[stkPtr];
        System.arraycopy(nameDotLocs, 0, dots, 0, stkPtr);
        return CompoundName.make(IdentifierVec.popFromStackVector(seqIdentifier),
                                 ids, dots);
    }

    /**
     * Parse a {@link TypeName}.
     * <pre>
     * TypeName:
     * Name [TypeModifierPragmas]
     * </pre>
     */
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    //@ ensures \result.syntax;
    public TypeName parseTypeName(Lex l) {
        Name name = parseName(l);
        TypeModifierPragmaVec modifiers = parseTypeModifierPragmas(l);
        return TypeName.make( modifiers, name );
    }

    /**
     * Parse square bracket pairs.  Wraps argument <code>type</code> in
     * {@link ArrayType} objects accordingly.
     *
     * <pre>
     * BracketPairs:
     * (LSQBRACKET RSQBRACKET)*
     * </pre>
     *
     * <p> Warning: when this method sees "{'[' ']'}* {'[' not-']'}",
     * it returns with "l" pointing to the '[' just before the
     * not-']'.
     */
    //@ requires l != null && type != null && l.m_in != null;
    //@ requires type.syntax;
    //@ ensures \result != null;
    //@ ensures \result.syntax;
    public Type parseBracketPairs(Lex l, Type type) {
        // most of this code is to preserve the warning in comment
        // above.  also, it is now recursive to put the annotations on
        // the current dimensions.
        if (l.ttype == ParserTagConstants.LSQBRACKET) {
            int loc=l.startingLoc;
            int i = 1;
            boolean done = false;
            while (!done) {
                switch (l.lookahead(i)) {
                    case ParserTagConstants.TYPEMODIFIERPRAGMA:
                        i++;
                        break;
                    case ParserTagConstants.RSQBRACKET:
                        done=true;
                        break;
                    default:
                        return type;
                }
            }
            l.getNextToken();
            TypeModifierPragmaVec modifiers	= parseTypeModifierPragmas(l);
            expect( l, ParserTagConstants.RSQBRACKET );
            type = ArrayType.make( modifiers,
                                   parseBracketPairs(l,type), loc);
        }
//      old impl:
//      while(l.ttype == TagConstants.LSQBRACKET 
//            && l.lookahead(1) == TagConstants.RSQBRACKET ) {
//        type = ArrayType.make( type, l.startingLoc );
//        l.getNextToken();
//        expect( l, TagConstants.RSQBRACKET );
//      }

        return type;
    }
    
    //@ requires l != null && l.m_in != null;
    public TypeModifierPragmaVec parseTypeModifierPragmas(/*@ non_null @*/ Lex l) {
        if (l.ttype != ParserTagConstants.TYPEMODIFIERPRAGMA) return null;
        TypeModifierPragmaVec seq = TypeModifierPragmaVec.make();
        do {
            seq.addElement((TypeModifierPragma)l.auxVal);
            l.getNextToken();
        } while (l.ttype == ParserTagConstants.TYPEMODIFIERPRAGMA);
        return seq;
    }
    
    /**
     * @return is a tag a {@link PrimitiveType} keyword?
     */
    public boolean isPrimitiveKeywordTag(int tag) {
        switch (tag) {
            case ParserTagConstants.BOOLEAN:
            case ParserTagConstants.BYTE:
            case ParserTagConstants.SHORT:
            case ParserTagConstants.INT:
            case ParserTagConstants.LONG:
            case ParserTagConstants.CHAR:
            case ParserTagConstants.FLOAT:
            case ParserTagConstants.DOUBLE:
            case ParserTagConstants.VOID:
                return true;

            default:
                return false;
        }
    }

    /**
     * Parses a {@link PrimitiveType}.  Returns <code>null</code> on
     * failure.
     *
     * <pre>   
     * PrimitiveType: one of
     * boolean byte short int long char float double void
     * </PRE>
     */
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null ==> \result.syntax;
    public PrimitiveType parsePrimitiveType(Lex l) {

        int tag;

        switch( l.ttype ) {
            case ParserTagConstants.BOOLEAN: tag = ParserTagConstants.BOOLEANTYPE; break;
            case ParserTagConstants.BYTE:    tag = ParserTagConstants.BYTETYPE;    break;
            case ParserTagConstants.SHORT:   tag = ParserTagConstants.SHORTTYPE;   break;
            case ParserTagConstants.INT:     tag = ParserTagConstants.INTTYPE;     break;
            case ParserTagConstants.LONG:    tag = ParserTagConstants.LONGTYPE;    break;
            case ParserTagConstants.CHAR:    tag = ParserTagConstants.CHARTYPE;    break;
            case ParserTagConstants.FLOAT:   tag = ParserTagConstants.FLOATTYPE;   break;
            case ParserTagConstants.DOUBLE:  tag = ParserTagConstants.DOUBLETYPE;  break;
            case ParserTagConstants.VOID:    tag = ParserTagConstants.VOIDTYPE;    break;
            default: return null;	// Fail!
        }
        // get here => tag is defined
    
        int loc = l.startingLoc;
        l.getNextToken();
        return JavafePrimitiveType.make( tag, loc );
    }
  
    /**
     * Parse a type, either a primitive type, a type name, but not an
     * array type.
     *
     * <pre>
     * PrimitiveTypeOrTypeName:
     * PrimitiveType
     * TypeName
     * </pre>
     */
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    //@ ensures \result.syntax;
    public Type parsePrimitiveTypeOrTypeName(Lex l) {
        Type type = parsePrimitiveType(l);
        if( type != null ) 
            return type;
        else
            return parseTypeName(l);
    }
  
    /**
     * Parse a {@link Type}, either a primitive type, a type name, or
     * an array type.
     *
     * <pre>
     * Type:
     * PrimitiveTypeOrTypeName BracketPairs
     * </pre>
     */
    //@ requires l != null && l.m_in != null;
    //@ ensures \result != null;
    //@ ensures \result.syntax;
    public Type parseType(Lex l) {
        Type type = parsePrimitiveTypeOrTypeName(l);
    
        // Allow for brackets on end 
        return parseBracketPairs(l, type);
    }
}
