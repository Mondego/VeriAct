/* Copyright Hewlett-Packard, 2002 */

class Sync {

  Sync c,d;
  void m1() {
    //@ assume this.c != null && this.d != null;
    //@ assume \max(\lockset) < this.c;
    //@ assume this.c < this.d;
    synchronized(this.c) {
      synchronized(this.d) {
      }
    }
  }
  
  void m2() {
    //@ assume \lockset[this];
    synchronized(this) {
      synchronized(this) {
      }
    }
  }

  void m3() {
    //@ assume \max(\lockset) < this;
    synchronized(this) {
      synchronized(this) {
      }
    }
  }

  synchronized void m4() {
    synchronized(this) {
      synchronized(this) {
      }
    }
  }

  static synchronized void m5() {
    //@ assert \lockset[Sync.class];
  }

  synchronized void m6() {
    /*@ assert \lockset[Sync.class]; */  // no way, Jose
  }
}

class SyncAxiom {
  Object mu, nu;
  //@ invariant nu != null;
  SyncAxiom() {
    mu = new Object();
    nu = new Object();
  }
  
  //@ axiom (\forall SyncAxiom s; s < s.nu);
  //@ axiom (\forall SyncAxiom s; s.mu != null ==> s < s.mu && s.mu < s.nu);
  
  /*@ monitored */ int x;
  /*@ monitored_by mu; */ int y;
  /*@ monitored_by nu; */ int z;
  /*@ monitored_by mu, nu; */ int w;
  /*@ monitored_by this; */ int v;
  /*@ monitored_by this, mu; */ int u;

  synchronized int m0() { // fails (Deadlock)
    synchronized (mu) { // fails (Null)
      synchronized (nu) {
	return x;
      }
    }
  }

  //@ requires \max(\lockset) < this || \lockset[this];
  //@ requires mu != null;
  synchronized int m1() {
    synchronized (mu) { // fails (Deadlock)
      synchronized (nu) {
	return x;
      }
    }
  }

  //@ requires \max(\lockset) < this || \lockset[this];
  //@ requires mu != null && \max(\lockset) < mu;
  synchronized int m2() {
    synchronized (mu) {
      synchronized (nu) {
	x = y = z = w = v = u = 2;
	return x + y + z + w + v + u;
      }
    }
  }

  //@ requires \max(\lockset) < this || \lockset[this];
  synchronized int m3() {
    x = v = 3;
    return x + u + v;
  }

  //@ requires \max(\lockset) < this;
  int m4() {
    if (mu == null) {
      synchronized (this) {
	x = v = u = 4;
	return x + v + u;
      }
    } else {
      synchronized (mu) {
	y = 4;
	int i = y + u;
	return i + x;  // fails on reading x
      }
    }
  }

  //@ requires \max(\lockset) < this;
  void m5() {
    if (mu == null) {
      synchronized (this) {
	u = 5;
      }
      synchronized (nu) {
	int i = w;
	w = i + 1;
      }
    } else {
      synchronized (mu) {
	int i = w;
	w = i + 1; // fails
      }
    }
  }

  //@ requires \max(\lockset) < this;
  int m6() {
    synchronized (this) {
      if (mu == null) {
	u = 5;
	synchronized (nu) {
	  int i = w;
	  w = i + 1;
	}
      } else {
	synchronized (mu) {
	  int i = w;
	  w = i + 1; // fails
	}
      }
    }
  }

  int m7() {
    return u;  // fails
  }

  void m8() {
    u = 8;  // fails
  }

  int m9() {
    if (mu == null) {
      return w;  // fails
    }
  }

  //@ requires \max(\lockset) < this;
  void m10() {
    if (mu != null) {
      synchronized (mu) {
	v = 4;  // fails
      }
    }
  }

  //@ requires \max(\lockset) < this;
  int m11() {
    if (mu != null) {
      synchronized (mu) {
	return v;  // fails
      }
    }
  }

  void m12() {
    //@ assert \lockset[\max(\lockset)];
    //@ assert null <= \max(\lockset);
    //@ assert \lockset[null];
  }

  //@ requires mu0 < mu1 && mu1 < mu2;
  //@ requires nu0 <= nu1 && nu1 <= nu2;
  //@ requires pu0 <= pu1 && pu1 < pu2;
  //@ requires qu0 < qu1 && qu1 <= qu2;
  void m13Transitivity(Object mu0, Object mu1, Object mu2,
		       Object nu0, Object nu1, Object nu2,
		       Object pu0, Object pu1, Object pu2,
		       Object qu0, Object qu1, Object qu2) {
    //@ assert mu0 <= mu2;
    //@ assert mu0 < mu2;
    //@ assert nu0 <= nu2;
    //@ assert pu0 <= pu2;
    //@ assert pu0 < pu2;
    //@ assert qu0 < qu2;
    //@ assert qu0 <= qu2;
  }

  void m14(Object mu, int k) {
    switch (k) {
      case 0:
	//@ assert null <= mu;
	break;
      case 1:
	//@ assume \lockset[mu];
	//@ assert null <= mu;
	break;
      case 2:
	//@ assume \lockset[mu];
	//@ assert mu <= \max(\lockset);
	break;
    }
  }
}
