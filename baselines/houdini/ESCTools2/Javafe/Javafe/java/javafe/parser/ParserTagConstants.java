/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser;

import javafe.util.Assert;

//@ model import javafe.tc.TagConstants;


/**
 * <code>TagConstants</tt> is a class defining the constants used to
 * identify different kinds of tokens.
 */

public class ParserTagConstants extends javafe.ast.ASTTagConstants
{
    public static final int EOF = javafe.ast.ASTTagConstants.LAST_TAG + 1;

    // Value tokens
    public static final int MAX_INT_PLUS_ONE = EOF + 1;
    public static final int MAX_LONG_PLUS_ONE = MAX_INT_PLUS_ONE + 1;

    // Pragma tokens
    public static final int LEXICALPRAGMA = MAX_LONG_PLUS_ONE + 1;
    public static final int MODIFIERPRAGMA = LEXICALPRAGMA + 1;
    public static final int POSTMODIFIERPRAGMA = MODIFIERPRAGMA + 1;
    public static final int STMTPRAGMA = POSTMODIFIERPRAGMA + 1;
    public static final int TYPEDECLELEMPRAGMA = STMTPRAGMA + 1;
    public static final int TYPEMODIFIERPRAGMA = TYPEDECLELEMPRAGMA + 1;

    //alx: dw implicit peer tag, it cannot stand with the other universe 
    //   modifiers, because it isn't a keyword
    public static final int IMPL_PEER = TYPEMODIFIERPRAGMA + 1;
    //alx-end

    public static final int UNKNOWN_KEYWORD = TYPEMODIFIERPRAGMA + 1;

    // Some punctuation tokens (rest come from OperatorTags)
    public static final int COMMA = UNKNOWN_KEYWORD + 1;
    public static final int SEMICOLON = COMMA + 1;
    public static final int LBRACE = SEMICOLON + 1;
    public static final int RBRACE = LBRACE + 1;
    public static final int LPAREN = RBRACE + 1;
    public static final int RPAREN = LPAREN + 1;
    public static final int LSQBRACKET = RPAREN + 1;
    public static final int RSQBRACKET = LSQBRACKET + 1;

    public static final int QUESTIONMARK = RSQBRACKET + 1;
    public static final int COLON = QUESTIONMARK + 1;

    public static final int FIELD = COLON + 1;

    public static final int C_COMMENT = FIELD + 1;
    public static final int EOL_COMMENT = C_COMMENT + 1;

    // Keywords
    public static final int FIRST_KEYWORD = EOL_COMMENT + 1;
    public static final int ABSTRACT = FIRST_KEYWORD;
    public static final int ASSERT   = ABSTRACT + 1;
    public static final int BOOLEAN  = ASSERT + 1;
    public static final int BREAK    = BOOLEAN + 1;
    public static final int BYTE     = BREAK + 1;
    public static final int CASE     = BYTE + 1;
    public static final int CATCH    = CASE + 1;
    public static final int CHAR     = CATCH + 1;
    public static final int CLASS    = CHAR + 1;
    public static final int CONST    = CLASS + 1;
    public static final int CONTINUE = CONST + 1;
    public static final int DEFAULT  = CONTINUE + 1;
    public static final int DO       = DEFAULT + 1;
    public static final int DOUBLE   = DO + 1;
    public static final int ELSE     = DOUBLE + 1;
    public static final int EXTENDS  = ELSE + 1;
    public static final int FALSE    = EXTENDS + 1;
    public static final int FINAL    = FALSE + 1;
    public static final int FINALLY  = FINAL + 1;
    public static final int FLOAT    = FINALLY + 1;
    public static final int FOR      = FLOAT + 1;
    public static final int GOTO     = FOR + 1;
    public static final int IF       = GOTO + 1;
    public static final int IMPLEMENTS = IF + 1;
    public static final int IMPORT   = IMPLEMENTS + 1;
    public static final int INSTANCEOF = IMPORT + 1;
    public static final int INT      = INSTANCEOF + 1;
    public static final int INTERFACE= INT + 1;
    public static final int LONG     = INTERFACE + 1;
    public static final int NATIVE   = LONG + 1;
    public static final int NEW      = NATIVE + 1;
    public static final int NULL     = NEW + 1;
    public static final int PACKAGE  = NULL + 1;
    public static final int PRIVATE  = PACKAGE + 1;
    public static final int PROTECTED= PRIVATE + 1;
    public static final int PUBLIC   = PROTECTED + 1;
    public static final int RETURN   = PUBLIC + 1;
    public static final int SHORT    = RETURN + 1;
    public static final int STATIC   = SHORT + 1;
    public static final int STRICT   = STATIC + 1;
    public static final int SUPER    = STRICT + 1;
    public static final int SWITCH   = SUPER + 1;
    public static final int SYNCHRONIZED = SWITCH + 1;
    public static final int THIS     = SYNCHRONIZED + 1;
    public static final int THROW    = THIS + 1;
    public static final int THROWS   = THROW + 1;
    public static final int TRANSIENT= THROWS + 1;
    public static final int TRUE     = TRANSIENT + 1;
    public static final int TRY      = TRUE + 1;
    public static final int VOID     = TRY + 1;
    public static final int VOLATILE = VOID + 1;
    //alx: dw definition of the universe type modifiers as keywords
    public static final int REP      = VOLATILE + 1;
    public static final int PEER     = REP + 1;
    public static final int READONLY = PEER + 1;
    public static final int WHILE    = READONLY + 1;
    //alx-end
    public static final int LAST_KEYWORD = WHILE;

    public static final int LAST_TAG = LAST_KEYWORD;

    /**
     * @return text representation of <code>code</code> (e.g., "=" for
     * <tt>ASSIGN</tt>).  Requires <code>code</code> is one of the
     * token constants defined in <code>Tokens</code> (including ones
     * inherited from <code>OperatorTags</code>).
     */

    //@ ensures \result != null;
    public static /*@non_null*/ String toString(int code) {
        if (FIRST_KEYWORD <= code && code <= LAST_KEYWORD)
            return getString(code - FIRST_KEYWORD);
        for(int i = 1 + LAST_KEYWORD - FIRST_KEYWORD; i < noTokens; i++)
            if (getCode(i) == code) return getString(i);

        if (code <= javafe.ast.ASTTagConstants.LAST_TAG)
            return javafe.ast.ASTTagConstants.toString(code);

        //alx: dw special case for IMPL_PEER, it isn't a keyword
        if (code==ParserTagConstants.IMPL_PEER)
        	return "[peer]";
        //alx-end

        return "Tag unknown to javafe.parser.TagConstants <" + code 
            + " (+" + (code - javafe.ast.ASTTagConstants.LAST_TAG) + ") >";
    }

    /**
     * Alphabetical list of Java punctuation strings.  In addition to
     * being used in <code>Tokens</code>, this variable is used by
     * <code>Lex</code> to implement <code>addJavaPunctuation</code>.
     */

    //@ invariant \nonnullelements(punctuationStrings);
    static final /*@ non_null @*/ String punctuationStrings[] = {
        "!", "!=", "%", "%=", "&", "&&", "&=", "(", ")", "*", "*=",
        "+", "++", "+=", ",", "-", "--", "-=", ".", "/", "/=", ":", ";",
        "<", "<<", "<<=", "<=", "=", "==", ">", ">=", ">>", ">>=", ">>>", ">>>=",
        "?", "[", "]", "^", "^=", "{", "|", "|=", "||", "}", "~", "/*", "//"
    };

    /**
     * List of codes for Java punctuation.  Order of this list agrees
     * with with order of punctuationStrings.  In addition to being
     * used in <code>Tokens</code>, this variable is used by
     * <code>Lex</code> to implement
     * <code>addJavaPunctuation</code>.
     */

    /*@ invariant punctuationCodes.length == punctuationStrings.length
     @	&& (\forall int i; 0 <= i && i < punctuationCodes.length
     @           ==>  punctuationCodes[i] != TagConstants.IDENT &&
     @                punctuationCodes[i] != TagConstants.BOOLEANLIT &&
     @                punctuationCodes[i] != TagConstants.INTLIT &&
     @                punctuationCodes[i] != TagConstants.LONGLIT &&
     @                punctuationCodes[i] != TagConstants.FLOATLIT &&
     @                punctuationCodes[i] != TagConstants.DOUBLELIT &&
     @                punctuationCodes[i] != TagConstants.STRINGLIT &&
     @                punctuationCodes[i] != TagConstants.CHARLIT &&
     @                punctuationCodes[i] != TagConstants.LEXICALPRAGMA &&
     @                punctuationCodes[i] != TagConstants.MODIFIERPRAGMA &&
     @                punctuationCodes[i] != TagConstants.STMTPRAGMA &&
     @                punctuationCodes[i] != TagConstants.TYPEDECLELEMPRAGMA &&
     @                punctuationCodes[i] != TagConstants.TYPEMODIFIERPRAGMA); */
    //@ invariant punctuationCodes.owner instanceof TagConstants;
    //@ spec_public
    static final int punctuationCodes[] = {
        NOT, NE, MOD, ASGREM, BITAND, AND, ASGBITAND, LPAREN, RPAREN, STAR, ASGMUL,
        ADD, INC, ASGADD, COMMA, SUB, DEC, ASGSUB, FIELD, DIV, ASGDIV, COLON,
        SEMICOLON,
        LT, LSHIFT, ASGLSHIFT, LE, ASSIGN, EQ, GT, GE, RSHIFT, ASGRSHIFT, URSHIFT,
        ASGURSHIFT,
        QUESTIONMARK, LSQBRACKET, RSQBRACKET, BITXOR, ASGBITXOR, LBRACE, BITOR,
        ASGBITOR, OR, RBRACE, BITNOT, C_COMMENT, EOL_COMMENT
    };

    /**
     * Alphabetical list of Java keywords.  The keyword codes are also
     * alphabetical, which means that if X is code of keyword K, then
     * keywordStrings[X - FIRST_KEYWORD] should equal K.
     */

    //@ invariant keywordStrings.length == 1 + LAST_KEYWORD - FIRST_KEYWORD;
    //@ invariant \nonnullelements(keywordStrings);
    //@ spec_public
    private static final /*@ non_null */ String keywordStrings[] = {
        "abstract", "assert", "boolean", "break", "byte", "case", "catch", "char",
        "class", "const", "continue", "default", "do", "double", "else",
        "extends", "false", "final", "finally", "float", "for", "goto", "if",
        "implements", "import", "instanceof", "int", "interface", "long", 
        "native", "new", "null", 
        "package", "private", "protected", "public", "return",
        "short", "static", "strictfp", "super", "switch", "synchronized", "this",
        "throw", "throws", "transient", "true", "try", "void", "volatile", 
        //alx: dw added rep, peer & readonly
        "rep", "peer", "readonly", 
        //alx-end
        "while"
    };

    //@ invariant \nonnullelements(otherStrings);
    //@ spec_public
    private static final /*@ non_null */ String otherStrings[] = {
        "IDENT",
        "CHARLIT", "INTLIT", "2147483648",
        "LONGLIT", "9223372036854775808L", "FLOATLIT", "DOUBLELIT", "STRINGLIT",
        "LEXICALPRAGMA", "MODIFIERPRAGMA", "POSTMODIFIERPRAGMA",
        "STMTPRAGMA", "TYPEDECLELEMPRAGMA", 
        "TYPEMODIFIERPRAGMA", "EOF"
    };

    //@ invariant otherCodes.length == otherStrings.length;
    //@ spec_public
    private static final int otherCodes[] = {
        IDENT,
        CHARLIT, INTLIT, MAX_INT_PLUS_ONE,
        LONGLIT, MAX_LONG_PLUS_ONE, FLOATLIT, DOUBLELIT, STRINGLIT,
        LEXICALPRAGMA, MODIFIERPRAGMA, POSTMODIFIERPRAGMA,
        STMTPRAGMA, TYPEDECLELEMPRAGMA,
        TYPEMODIFIERPRAGMA, EOF
    };

    //@ invariant noTokens == keywordStrings.length + punctuationStrings.length + otherStrings.length;
    //@ spec_public
    private static final int noTokens =
        keywordStrings.length + punctuationStrings.length + otherStrings.length;

    //@ requires 0 <= index && index < noTokens;
    private static int getCode(int index) {
        Assert.precondition(0 <= index && index < noTokens);
        if (index < 1 + LAST_KEYWORD - FIRST_KEYWORD) return FIRST_KEYWORD + index;
        index -= 1 + LAST_KEYWORD - FIRST_KEYWORD;
        if (index < punctuationCodes.length) return punctuationCodes[index];
        index -= punctuationCodes.length;
        if (index < otherCodes.length) return otherCodes[index];
        index -= otherCodes.length;
        Assert.notFalse(false, "Bad invariant");
        return -1; // Dummy
    }

    //@ requires 0 <= index && index < noTokens;
    //@ ensures \result != null;
    private static String getString(int index) {
        Assert.precondition(0 <= index && index < noTokens);
        if (index < keywordStrings.length)
            return keywordStrings[index];
        index -= keywordStrings.length;
        if (index < punctuationStrings.length)
            return punctuationStrings[index];
        index -= punctuationStrings.length;
        if (index < otherStrings.length)
            return otherStrings[index];
        index -= otherStrings.length;
        Assert.notFalse(false, "Bad invariant");
        return null; // Dummy
    }

    /** Perform module-level checks. */
    public static void zzzz() {
        Assert.notFalse(noTokens == (keywordStrings.length
                                     + punctuationStrings.length
                                     + otherStrings.length));
        Assert.notFalse(noTokens == (keywordStrings.length
                                     + punctuationCodes.length
                                     + otherCodes.length));
        // Check for duplicates
        for(int i = 0; i < noTokens; i++)
            for(int j = i+1; j < noTokens; j++)
                if (getCode(i) == getCode(j))
                    System.out.println("Duplicate (" + getCode(i) + ") at " +
                                       getString(i) + " " + getString(j));
    }
}
