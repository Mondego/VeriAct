public class TestString2 extends LocalTestCase {

  //@ requires s.length() > 10;
  //@ ensures \result == "ble ble";
  public static String testNotMatch(/*@ non_null */ String s) {
      //      int i = s.lastIndexOf(".") + 1;
      String sub = s.substring(3);
      return sub;
  } 

  public static void testSubstringConsistency() {
        String s = "fooooo";
        String sub = s.substring(3);
        //@ assert false;
    }


    /*@  public normal_behavior
      @   requires 0 <= beginIndex;
      @   requires beginIndex <= s.length();
      @   ensures \result.equals(s.substring(beginIndex, s.length()));
      @   ensures  \fresh(\result);
      @ also
      @  public exceptional_behavior
      @   requires beginIndex < 0 || beginIndex > s.length();
      @   signals_only StringIndexOutOfBoundsException;
      @*/
    public /*@ pure @*/ /*@ non_null @*/ String testSubstringSimulation(/*@ non_null*/String s, int beginIndex)   throws StringIndexOutOfBoundsException {
        return s.substring(beginIndex, s.length());
    }

    //*@ ensures \fresh(\result);
    public static String testSubstringIdentity(/*@ non_null */ String s) {
        String a = s.substring(0, s.length());
        //@ assert a.equals(s);
        return a;
    }
}
