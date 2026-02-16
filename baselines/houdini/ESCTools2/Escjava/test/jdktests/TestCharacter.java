public class TestCharacter extends LocalTestCase {

  public void testConstructor() {
    Character cc = new Character('a');
    assertTrue( cc.charValue() == 'a');
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testEquals() {
    Character ca = new Character('a');
    Character cb = new Character('b');
    Character cc = new Character('a');
    assertTrue( ca.equals(ca) );
    assertTrue(!ca.equals(cb) );
    assertTrue( ca.equals(cc) );
    assertTrue(!ca.equals(null) );
    Object o = new Object();
    assertTrue(!ca.equals(o) );

    assertTrue( ca.compareTo(cc) == 0);
    assertTrue( ca.compareTo(ca) == 0);

//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testToString() {
    Character cc = new Character('a');
    String s = cc.toString();
    assertTrue( s.length() == 1);
    assertTrue( s.charAt(0) == 'a');
    String ss = Character.toString('a');
    assertTrue( s.equals(ss) );
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testDigit() {
    assertTrue( Character.isDigit('5') );
    char c = '7';
    assertTrue( Character.isDigit(c) );
    assertTrueNP( Character.forDigit(Character.digit(c,10),10) == c);
    //@ assert ( Character.forDigit(Character.digitVal(c),10) == c);
    assertTrueNP ( Character.digit('3',10) == 3);
    //@ assert ( Character.digitVal('3') == 3);
    assertTrue( Character.forDigit(4,10) == '4');
//@ assert false; // TEST FOR CONSISTENCY
  }

  public void testClass() {
    assertTrue( Character.MIN_RADIX == 2);
    assertTrue( Character.MAX_RADIX >= 16);
// FIXME    //@ assert ( Character.TYPE == \type(char));
    // FIXME assertTrue( Character.TYPE == char.class);
//@ assert false; // TEST FOR CONSISTENCY
  }
/* FIXME
   - need stuff on character classes, that all the constants are distinct
   - need stuff on the Subsets
   - no definition of compareTo
   - need more to relate digit, digitVal, forDigit
   isLowerCase
   isUpperCase
   isTitleCase
   isDigit
   isDefined
   isLetter
   isJavaIdentifierStart
   isJavaIdentifierPart
   isUnicodeIdentifierStart
   isUnicodeIdentifierPart
   isIdentifierIgnorable
   toLowerCase
   toUpperCase
   toTitleCase
   digitVal
   digit
   forDigit
   getNumericValue

   isSpaceChar
   isWhitespace
   isISOControl
   getDirectionality
   isMirrored
   getType
   compareTo
*/
}
