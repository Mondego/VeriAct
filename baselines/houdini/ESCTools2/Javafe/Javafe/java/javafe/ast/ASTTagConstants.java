/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.ast;

/**
 * <code>TagConstants</code> is a class defining a partially-opaque
 * type for tags used in the AST.  "Partially opaque" means that the
 * representation of this type is known -- it's an <code>int</code> --
 * but code should not depend on the detailed mapping of operator tags
 * to integers.
 *
 * @see OperatorTags
 * @see GeneratedTags
 */

public class ASTTagConstants extends OperatorTags
{
    /**
     * Used to indicate that no tag applies, for example, when looking
     * up the tag associated with a string.
     */
    public static final int NULL = -1;

    public static final int IDENT = javafe.ast.OperatorTags.LAST_TAG + 1;

    public static final int ERRORTYPE = IDENT + 1;
    public static final int BOOLEANTYPE = ERRORTYPE + 1;
    public static final int CHARTYPE = BOOLEANTYPE + 1;
    public static final int VOIDTYPE = CHARTYPE + 1;
    public static final int NULLTYPE = VOIDTYPE + 1;
    // The following must be in the order of implicit promotion
    public static final int BYTETYPE = NULLTYPE + 1;
    public static final int SHORTTYPE = BYTETYPE + 1;
    public static final int INTTYPE = SHORTTYPE + 1;
    public static final int LONGTYPE = INTTYPE + 1;
    public static final int FLOATTYPE = LONGTYPE + 1;
    public static final int DOUBLETYPE = FLOATTYPE + 1;

    public static final int BOOLEANLIT = DOUBLETYPE +1;
    public static final int INTLIT = BOOLEANLIT + 1;
    public static final int LONGLIT = INTLIT + 1;
    public static final int CHARLIT = LONGLIT + 1;
    public static final int FLOATLIT = CHARLIT + 1;
    public static final int DOUBLELIT = FLOATLIT + 1;
    public static final int STRINGLIT = DOUBLELIT + 1;
    public static final int NULLLIT = STRINGLIT + 1;
    public static final int BYTELIT = NULLLIT + 1;
    public static final int SHORTLIT = BYTELIT + 1;

    public static final int LAST_TAG = SHORTLIT;

    //@ private invariant \nonnullelements(tags);
    private static final /*@ non_null @*/ String[] tags = { 
        "IDENT", "ERRORTYPE",
        "BOOLEANTYPE", "CHARTYPE", "VOIDTYPE", "NULLTYPE",
        "BYTETYPE", "SHORTTYPE", "INTTYPE", "LONGTYPE",
        "FLOATTYPE", "DOUBLETYPE",
        "BOOLEANLIT", "INTLIT", "LONGLIT", "CHARLIT",
        "FLOATLIT", "DOUBLELIT", "STRINGLIT", "NULLLIT",
        "BYTELIT", "SHORTLIT" };

    //@ requires tag <= LAST_TAG;
    //@ ensures \result != null;
    public static /*@non_null*/String toString(int tag) {
        if (IDENT <= tag && tag <= LAST_TAG)
            return tags[tag - IDENT];

        return OperatorTags.toString(tag);
    }
}
