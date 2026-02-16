// This may be repeating some of the tests in the nijmegen/JCF
// subdirectory
// Suggested by Joe who reference Martijn Warnier - 1/7/04
// and the FMCO paper on Java Program Verification Challenges

class Static {
    static boolean result1, result2, result3, result4;
    
    /*@ normal_behavior
      @   requires !\is_initialized(Static) &&
      @            !\is_initialized(c1) &&
      @            !\is_initialized(c2);
      @ assignable Static.*, c1.*, c2.*;
      @   ensures  result1  &&
      @            !result2 &&
      @            result3  &&
      @            result4;
      @*/
    static void m() {
        result1 = c1.b1;
        result2 = c2.b2;
        result3 = c1.d1;
        result4 = c2.d2;
    }
}

class c1 {
    static boolean b1 = c2.d2;
    static boolean d1 = true;
}

class c2 {
    static boolean d2 = true;
    static boolean b2 = c1.d1;
}


