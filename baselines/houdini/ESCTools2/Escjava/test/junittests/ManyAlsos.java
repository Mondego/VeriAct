// This test checks for aggresive use of 'also' for complex branching behavior.

public class ManyAlsos {
  boolean _b; //@ in objectState;
  int _i; //@ in objectState;
  Object _o; //@ in objectState;

  /*@ public normal_behavior
    @   requires b;
    @   modifies _b, _i, _o;
    @   ensures _b == b;
    @ also
    @ public normal_behavior
    @   requires !b;
    @   modifies this.*;
    @   ensures _i == -1;
    @ also
    @ public behavior
    @   requires i == 2;
    @   modifies objectState;
    @   diverges _b && (_o instanceof String);
    @ also
    @ public exceptional_behavior
    @   requires i == -2;
    @   modifies \not_specified;
    @   signals (Exception) (_o instanceof String);
    @ also
    @ public normal_behavior
    @   requires i != -2 && i != 2;
    @   modifies \not_specified;
    @   ensures \result == (b ? i == 3 : o != null);
    @*/
  public boolean m(boolean b, int i, Object o) throws Exception {
    if (b)
      _b = b;
    else
      _i = -1;
    switch (i) {
      case -1: _o = null; break;
      case 0: if (b) _i = 1; else _i = 2; break;
      case 1: _o = new Object(); break;
      default:
        _o = new String();
    }
    if (i == 2)
      while(true) {
        _b = true;
      }
    if (i == -2)
      throw new Exception();
    return (b ? i == 3 : o != null);
  }
}
