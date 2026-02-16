package problems;

import problems.Predicates;

class JmlAssertionsAndAssumptionsKey extends Predicates
{
  static int i;
  static Object o;
  static String s;

  //@ modifies i, s, o;
  static void m() {
    i++;
    s = new String("foo");
    o = new Integer(1);
  }

  //@ modifies i;
  static void n(int j, Object p, String t) {
    //@ assume p != null;
    //@ assert p != null;
    if (p.equals(t))
      return;
    //@ assume i == 1;
    //@ assume j == 1;
    i |= 0x100;
    j |= 0xEFF;
    //@ assert (i & j) == 1;
    t += "bar";
  }

  //@ modifies JmlAssertionsAndAssumptionsKey.s;
  static Object o(int i, Object o, String s) {
    //@ assume o instanceof Integer;
    //@ assume ((Integer)o).intValue() == 1;
    //@ assume s != null;

    //@ assert ((Integer)o).intValue() == 1;
    //@ assert s != null;
    if (s.equals(o))
      return new Integer(i);
    //@ assume i == 257;
    i ^= 0xFF;
    //@ assert i == 510;
    JmlAssertionsAndAssumptionsKey.s += "piggie";
    return s;
  }

  static Object p(int i, Object o, String s) {
    //@ assume o instanceof Integer;
    //@ assume s != null;

    //@ assert o instanceof Integer;
    //@ assert s != null;
    switch (i) {
      case -255:
        s = "duck";
        return o;
      case -1:
        o = s;
        return new Short((short)i++);
      case 0:
        s = (String)o;
        return new Byte((byte)i--);
      case 1:
        i *= -2;
        return (String)o + s;
      case 255:
        i = o.hashCode();
        return s;
      case 257:
        s = ((Integer)o).toString();
        return s;
      default:
        return null;
    }
  }
  
  public static void main(String[] args) {
    //@ assume i == 0;
    // assume s == null;
    // assume o == null;
    // assert i == 0;
    // assert s == null;
    // assert o == null;
    m();
    //@ assume i == 1;
    //@ assert i == 1;
    //@ assert s.equals("foo");
    //@ assert ((Integer)o).intValue() == 1;
    n(i, o, s);
    //@ assert i == 256 + 1;
    //@ assert s.equals("foo");
    //@ assert new Integer(1).equals(o);
    Object p = p(-1 & i, o, (String)o(i, o, s));
    //@ assert p.equals("1");
    Object q = p(-1, o, (String)p);
    //@ assert i == 257;
    //@ assert s.equals("foopiggie");
    //@ assert ((Integer)o).intValue() == 1;
    //@ assert ((Short)q).shortValue() == -1;
  }
}
