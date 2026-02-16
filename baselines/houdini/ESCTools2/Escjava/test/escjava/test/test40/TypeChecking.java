class TypeChecking {

    /*@ non_null @*/ int x;	// error: non-reference type

}


class Top {
    void get(Object x) {}
}

class Bottom extends Top {
    void get(/*@ non_null @*/ Object x) {}		// error: overrides!
}


class Returns {

  /*@ non_null @*/ Returns() { }  // error: cannot be used with constructor

  /*@ non_null @*/ String name() { return "hello"; }

  abstract /*@ non_null @*/ String m();

  /*@ non_null */ static Object p() { return new Object(); }

  /*@ non_null */ int q0() { return 3; }  // error:  non-reference result type
  /*@ non_null */ void q1() { }  // error:  non-reference result type

}

class Ret {
  public /*@ non_null @*/ Object h() { return new Object(); }
}

interface IReturn {
  public /*@ non_null @*/ Object h();
}

interface IReturn2 extends IReturn {
  public /*@ non_null @*/ Object h();  // error: cannot be used on override
}

interface IReturn3 {
  public /*@ non_null @*/ Object h();
}

class EReturns0 extends Ret implements IReturn, IReturn3 {
  public /*@ non_null @*/ Object h() {  // error: cannot be used on override
    return new Object();
  }
}

class EReturns1 extends Ret {
  public /*@ non_null @*/ Object h() {  // error: cannot be used on override
    return new Object();
  }
}

class EReturns2 implements IReturn3 {
  public /*@ non_null @*/ Object h() {  // error: cannot be used on override
    return new Object();
  }
}

class EReturns3 extends Ret implements IReturn {
  public Object h() {
    return new Object();
  }
}

class EReturns4 extends EReturns3 implements IReturn3 {
  public Object h() {
    return new Object();
  }
}

class EReturns5 extends EReturns3 implements IReturn3, IReturn {
  public /*@ non_null */ Object h() {  // error: cannot be used on override
    return new Object();
  }
}
