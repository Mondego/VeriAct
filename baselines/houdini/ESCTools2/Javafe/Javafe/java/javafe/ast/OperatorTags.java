/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.ast;

import javafe.util.Assert;

/**
 * <code>OperatorTags</code> is a class defining a partially-opaque
 * type for tags used in the AST.  See <code>TagConstants</code> for
 * more information.
 */

public class OperatorTags extends GeneratedTags
{
    // Binary operator tags
    public static final int FIRST_TAG = javafe.ast.GeneratedTags.LAST_TAG + 1;
    public static final int OR = FIRST_TAG;
    public static final int AND = OR + 1;
    public static final int BITOR = AND + 1;
    public static final int BITXOR = BITOR + 1;
    public static final int BITAND = BITXOR + 1;
    public static final int NE = BITAND + 1;
    public static final int EQ = NE + 1;
    public static final int GE = EQ + 1;
    public static final int GT = GE + 1;
    public static final int LE = GT + 1;
    public static final int LT = LE + 1;
    public static final int LSHIFT = LT + 1;
    public static final int RSHIFT = LSHIFT + 1;
    public static final int URSHIFT = RSHIFT + 1;
    public static final int ADD = URSHIFT + 1;
    public static final int SUB = ADD + 1;
    public static final int DIV = SUB + 1;
    public static final int MOD = DIV + 1;
    public static final int STAR = MOD + 1;

    // Assignment-operator tags
    public static final int ASSIGN = STAR + 1;
    public static final int ASGMUL = ASSIGN + 1;
    public static final int ASGDIV = ASGMUL + 1;
    public static final int ASGREM = ASGDIV + 1;
    public static final int ASGADD = ASGREM + 1;
    public static final int ASGSUB = ASGADD + 1;
    public static final int ASGLSHIFT = ASGSUB + 1;
    public static final int ASGRSHIFT = ASGLSHIFT + 1;
    public static final int ASGURSHIFT = ASGRSHIFT + 1;
    public static final int ASGBITAND = ASGURSHIFT + 1;
    public static final int ASGBITOR = ASGBITAND + 1;
    public static final int ASGBITXOR = ASGBITOR + 1;

    // Unary operator tags
    public static final int UNARYADD = ASGBITXOR + 1;
    public static final int UNARYSUB = UNARYADD + 1;
    public static final int NOT = UNARYSUB + 1;
    public static final int BITNOT = NOT + 1;
    public static final int INC = BITNOT + 1;
    public static final int DEC = INC + 1;

    // Postfix unary operators
    public static final int POSTFIXINC = DEC + 1;
    public static final int POSTFIXDEC = POSTFIXINC + 1;

    public static final int LAST_TAG = POSTFIXDEC;

    /**
     * @return the text representation of <code>code</code>.  For
     * example, return the string "=" for the tag {@link #ASSIGN}.
     */
    //@ ensures \result != null;
    public static /*@non_null*/String toString(int opTag) {
        if (FIRST_TAG <= opTag && opTag <= LAST_TAG)
            return opStrings[opTag - FIRST_TAG];

        if (opTag < FIRST_TAG)
            return GeneratedTags.toString(opTag);

        return "Unknown tag <" + opTag + ">";
    }

//    private static final /*@ non_null @*/ int[] opTags = {
//        OR, AND, BITOR, BITXOR, BITAND, NE, EQ, GE, GT, LE, LT,
//        LSHIFT, RSHIFT, URSHIFT, ADD, SUB, DIV, MOD, STAR,
//        ASSIGN, ASGMUL, ASGDIV, ASGREM, ASGADD, ASGSUB,
//        ASGLSHIFT, ASGRSHIFT, ASGURSHIFT, ASGBITAND, ASGBITOR, ASGBITXOR,
//        UNARYADD, UNARYSUB, NOT, BITNOT, INC, DEC, POSTFIXINC, POSTFIXDEC
//    };

    //@ private invariant \nonnullelements(opStrings);
//    //@ private invariant opTags.length == opStrings.length;
    private static final /*@ non_null @*/ String[] opStrings = {
        "||", "&&", "|", "^", "&", "!=", "==", ">=", ">", "<=", "<",
        "<<", ">>", ">>>", "+", "-", "/", "%", "*",
        "=", "*=", "/=", "%=", "+=", "-=",
        "<<=", ">>=", ">>>=", "&=", "|=", "^=",
        "+", "-", "!", "~", "++", "--", "++", "--"
    };

}
