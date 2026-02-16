/* Copyright Hewlett-Packard, 2002 */

class Try {
  void m1() {
    int x,y;
    T2 t;
    try {
      x=0;
      //@ assume \typeof(t) == \type(T2) && t != null;
      throw t;
    } catch(T3 t3) {
      x=3;
    } catch(T1 t1) {
      x=1;
    } catch(T2 t2) {
      x=2;
    }
    //@ assert x==1;
  }
}

class T1 extends Throwable {}
class T2 extends T1 {}
class T3 extends T2 {}
