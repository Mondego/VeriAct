// Tests the use of non_null_by_default class annotation in the
// presense of non_null and nullable annotations and nested and inner
// classes, with and without inheritance.

class /*@ non_null_by_default @*/ NonNullDefault {
  /* non_null */ Object o;    // reminder of default
  /*@ nullable @*/ Object nu; // override class-scope default
  /*@ non_null @*/ Object nn; // redundant

  NonNullDefault() {
    // should give a warning on o and nn
  }

  NonNullDefault(Object p) {
    o = p;
    // should give a warning about nn
  }

  NonNullDefault(Object p, 
                 /*@ non_null *@/ /* redundant */ Object q, 
                 /*@ nullable @*/ /* override class-scope default */ Object r) {
    o = p;
    nu = r;
    nn = q;
    // no warnings
  }
  
  /*@ nullable @*/ Object nu_m1() {
    return o;
  }

  /*@ nullable @*/ Object nu_m2() {
    return nu;
  }

  /*@ nullable @*/ Object nu_m3() {
    return nn;
  }

  /*@ non_null @*/ Object nn_m1() {
    return o;
  }

  /*@ non_null @*/ Object nn_m2() {
    return nu;
  }

  /*@ non_null @*/ Object nn_m3() {
    return nn;
  }

  /* non_null reminder of default */ Object m1() {
    return o;
  }

  /* non_null reminder of default */ Object m2() {
    return nu;
  }

  /* non_null reminder of default */ Object m3() {
    return nn;
  }
}

/*@ non_null_by_default @*/ interface I {
  final static /* non_null default */ Object o = new Object();
  final static /*@ non_null @*/ /* redundant */ Object p = new Object();
  final static /*@ nullable @*/ /* overriding interface-scope default */ Object q = null;
  
  /* non_null default */ Object m1();

  /*@ non_null @*/ /* redundant */ Object m2();

  /*@ nullable @*/ /* overriding interface-scope default */ Object n(/* non_null default */ Object p);

  /*@ nullable @*/ /* overriding interface-scope default */ Object n(/*@ nullable @*/ /* overriding interface-scope default */ Object p);
}

/* non_null_by_default reminder of default */ class C extends NonNullDefault {
}

/*@ nullable_by_default @*/ class D extends C {
  // not overriding class-scope annotation on C or NonNullDefault
  // since class-scoped annotations are not inherited
}

class E_right implements I {
  /* non_null default */ Object m1() {
    return new Object();
  }

  /*@ non_null @*/ /* redundant */ Object m2() {
    return new Object();
  }

  /*@ nullable @*/ /* overriding interface-scope default */ Object n(/* non_null default */ Object p) {
    return null;
  }
}

class E_typeerror implements I {
  /* non_null default */ Object m1() {
    return new Object();
  }

  /*@ nullable @*/ /* not consistent with I's definition of m2 */ Object m2() {
    return null;
  }

  /*@ nullable @*/ /* overriding interface-scope default */ Object n(Object p) {
    return null;
  }
}

class E_typeerror implements I {
  /* non_null default */ Object m1() {
    return new Object();
  }

  /*@ nullable @*/ /* not consistent with I's definition of m2 */ Object m2() {
    return null;
  }

  /*@ nullable @*/ /* overriding interface-scope default */ Object n(Object p) {
    return null;
  }
}

