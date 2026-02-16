// This test file checks whether or not certain Null and NonNull checks are
// emitted.
// Note:  This test file is different from most, in that it is run with the -pgc
// switch.  The reason is that there is no easy to otherwise test whether a
// given check has been laid down.

class CheckSuper {
  int f;
  /*@ non_null */ Check x;

  /*@ non_null */ static Check g;
  static Check h;
}

class Check extends CheckSuper {
  /*@ non_null */ Check y;
  Check w;
  /*@ non_null */ static Check gg = new CheckSub();

  void m(/*@ non_null */ Check p, Check q) {
    int k = this.f;
    k = f;
    k = super.f;
    k = ((CheckSub)this).f;          // this performs a Cast check
    k = (this).f;

    k = x.f;
    k = this.x.f;
    k = y.f;
    k = ((CheckSub)this).z.f;        // this performs a Cast check
    
    k = h.f;                         // this performs a Cast check
    k = g.f;
    k = CheckSuper.g.f;
    k = this.g.f;
    Check ee = null;
    k = ee.g.f;
    k = ((CheckSuper)null).g.f;      // this performs a Cast check
    k = gg.f;
    
    k = p.f;
    /*@ non_null */ Check r = p;
    k = r.f;

    k = q.f;                         // this performs a Null check
    /*@ non_null */ Check s = q;     // this performs a NonNull check
    k = s.f;

    Check t = p;
    k = t.f;                         // this performs a Null check
  }

  Check(/*@ non_null */ Check p, Check q) {
    int k = x.f;
    k = y.f;                         // this performs a Null check
    k = ((CheckSub)this).z.f;        // this performs a Cast check

    k = p.f;
    k = q.f;                         // this performs a Null check

    k = h.f;                         // this performs a Cast check
    k = g.f;
    k = this.g.f;
    k = super.g.f;
    k = p.g.f;
    k = gg.f;
    k = this.gg.f;
    k = p.gg.f;

    /*@ non_null */ Check r = p;
    k = r.f;
  }

}

class CheckSub extends Check {
  /*@ non_null */ Check z;

  CheckSub() {
    super(null, null);               // this performs a NonNull check on the first parameter
    Check p = x;
    m(p, p);                         // this performs a NonNull check on the first parameter
    m(x, null);
    super.m(p, p);                   // this performs a NonNull check on the first parameter
    super.m(x, null);
    Check r = this;
    r.m(p, p);                       // this performs a Null check on "r" and a NonNull check on the first parameter
    r.m(x, null);                    // this performs a Null check on "r"
    /*@ non_null */ Check s = x;
    s.m(p, p);                       // this performs a NonNull check on the first parameter
    ((Check)(s)).m(x, null);
  }

  void m(Check p, Check q) {  // this is a method override, so the parameters inherit any "non_null" modifier pragma
    int k = p.f;
    k = q.f;                         // this performs a Null check
  }
}
