
class setfieldsuper {
  int instancevar;
}

class setfield extends setfieldsuper {
  static int classvar; // Used for testing by fieldref class
  int[] arrayvar;

  public static void m1(setfield o) {
    //@ assume o != null;
    o.classvar = 10;
    o.instancevar = 10;
    // o.instancevar += 5;
    //@ assert 20 == o.instancevar + classvar;
  }

  public static void m2(setfield o) {
    //@ assume o != null;
    //@ assume o.arrayvar != null;
    //@ assume o.arrayvar.length == 10;
    o.arrayvar[9] = 10;
    //@ assert 10 == o.arrayvar[9];
  }
}
