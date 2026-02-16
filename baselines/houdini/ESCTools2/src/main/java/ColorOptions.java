package escjava;

import java.awt.Color;

public class ColorOptions {
    static public final /*@ nullable*/ Color notParsed = null;
    static public final /*@ non_null @*/ Color typecheckOK = Color.GREEN;
    static public final /*@ non_null @*/ Color typecheckCaution = Color.YELLOW;
    static public final /*@ non_null @*/ Color typecheckError = Color.RED;
    static public final /*@ non_null @*/ Color staticcheckOK = Color.GREEN;
    static public final /*@ non_null @*/ Color staticcheckError = Color.RED;
    static public final /*@ non_null @*/ Color staticcheckTimeout = Color.BLUE;
    static public final /*@ non_null @*/ Color childError = new Color(255,128,0);
}
