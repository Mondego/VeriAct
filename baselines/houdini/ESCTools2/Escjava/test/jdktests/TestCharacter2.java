public class TestCharacter2 extends LocalTestCase { 

    public static void testWhiteSpaceConsistent() {
        Character.isWhitespace('Z');
        //@ assert false;  // TEST FOR CONSISTENCY - 1
    }

    public static void testWhiteSpace() {
        if (Character.isWhitespace('a')) {
            // the spec is not strong enough to prove it is not white spc but should be satisfiable
            //@ unreachable;  // TEST FOR CONSISTENCY - 2
        }
    }

    public static void testGetType() {
        char[] spcs = { '\u0020', '\u00A0', '\u2000' };
        for (int i = 0; i < spcs.length; i++) {
            char ch = spcs[i];
            //@ assert (Character.getType(ch) == Character.SPACE_SEPARATOR);
        }
    }
}
