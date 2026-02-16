package problems;

import problems.Predicates;

class JmlAssertions extends Predicates
{
  static int i;
  static Object o;
  static String s;

  static void m() {
    i++;
    s = new String("foo");
    o = new Integer(1);
  }

  static void n(int j, Object p, String t) {
    //@ assert O(p);
    if (p.equals(t))
      return;
    i |= 0x100;
    j |= 0xEFF;
    //@ assert J(i, j);
    t += "bar";
  }

  static Object o(int i, Object o, String s) {
    //@ assert O(o);
    //@ assert S(s);
    if (s.equals(o))
      return new Integer(i);
    i ^= 0xFF;
    //@ assert I(i);
    JmlAssertions.s += "piggie";
    return s;
  }

  static Object p(int i, Object o, String s) {
    //@ assert O(o);
    //@ assert S(s);
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
    //@ assert I(i);
    //@ assert S(s);
    //@ assert O(o);
    m();
    //@ assert I(i);
    //@ assert S(s);
    //@ assert O(o);
    n(i, o, s);
    //@ assert I(i);
    //@ assert S(s);
    //@ assert O(o);
    Object p = p(-1 & i, o, (String)o(i, o, s));
    //@ assert O(p);
    Object q = p(-1, o, (String)p);
    //@ assert I(i);
    //@ assert S(s);
    //@ assert O(o);
    //@ assert O(q);
  }
}
