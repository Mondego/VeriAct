package problems;

class Predicates
{
  public static /*@ pure @*/ boolean I(int i) {
    return true;
  }

  public static /*@ pure @*/ boolean O(Object o) {
    return true;
  }

  public static /*@ pure @*/ boolean S(String s) {
    return true;
  }
  
  public static /*@ pure @*/ boolean J(int i, int j) {
    return true;
  }
}

