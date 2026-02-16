class ObjStack3 {
  /*@ non_null */ Object [] a;
  //@ invariant a.length == 10;
  //@ invariant \elemtype(\typeof(a)) == \type(Object);
  int n;  //@ invariant 0 <= n & n <= 10;
  //@ invariant (\forall int i; n <= i & i < 10 ==> a[i] == null);
  //@ invariant a.owner == this;

  ObjStack3() {
    n = 0;
    a = new Object[10];
    //@ set a.owner = this;
  }

  //@ requires n < 10;
  void Push(Object o) {
    a[n++] = o;
  }

  //@ requires n > 0;
  Object Pop() {
    Object o = a[--n];
    a[n] = null;
    return o;
  }

}

class Foo {
  /*@ non_null */ Object [] b;
  //@ invariant b.owner == this;

  void m() {
    //@ assert (\forall ObjStack3 x, y; x != y ==> x.a != y.a);
    //@ assert (\forall Foo x, y; x != y ==> x.b != y.b);
    /*@ assert (\forall ObjStack3 x; (\forall Foo y;
                   (Object)x != y ==> x.a != y.b));
    */
  }

  Foo() {
    b = new Object[10];
    //@ set b.owner = this;
  }



}

