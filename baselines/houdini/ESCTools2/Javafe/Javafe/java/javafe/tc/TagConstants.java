/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

public class TagConstants extends javafe.parser.ParserTagConstants
{
    public static final int TYPESIG = javafe.parser.ParserTagConstants.LAST_TAG + 1;

    public static final int LAST_TAG = TYPESIG;

    //@ ensures \result != null;
    public static /*@non_null*/String toString(int tag) {
	switch (tag) {
	    case TYPESIG:
		return "TYPESIG";

	    default:
		return javafe.parser.ParserTagConstants.toString(tag);
	}
    }
} // end of class TagConstants

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
