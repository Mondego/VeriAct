/** This test file produces various sychronization warnings to make sure that
 ** the caret positions in the error messages are placed correctly.<p>
 **
 ** This file does not check any non-synchronization errors.  These are
 ** found in file Hotspot.java.
 **/

class SynchHotspot {
  //@ monitored
  int x;

  Object mu0, mu1;
  double[] d /*@ monitored_by mu0, this.mu1; */;

  SynchHotspot() {
    // These don't give any warnings in constructors
    x = 0;
    d = null;
  }
  
  int mRace0() {
    return x;
  }
  
  int mRace1() {
    return this.x;
  }
  
  void mRace2() {
    x = 5;
  }
  
  void mRace3() {
    this.x = 5;
  }
  
  void mRace4() {
    this.x += 5;
  }
  
  void mRace5() {
    this.x++;
  }
  
  void mRace6() {
    ++this.x;
  }
  
  double[] mRace7() {
    return d;
  }
  
  void mRace8() {
    d = new double[100];
  }

  double mRace9() {
    return d[1];
  }

  //@ requires d != null;
  void mRace10() {
    this.d[0] = 0.1e2;
  }

  //@ requires \max(\lockset) < this || \lockset[this];
  synchronized void mRace11() {
    x++;  // okay
    d = null;  // not okay
  }

  //@ requires mu0 != null && \max(\lockset) < mu0;
  void mRace12() {
    double[] a;
    synchronized (mu0) {
      a = d;  // okay
    }
    double y;
    if (a != null && a.length != 0) {
      y = a[0];  // okay, even though this gets the element d[0]  (we don't
                 // have an annotation to prevent this)
    }
    synchronized (mu0) {
      d = null;  // race condition (mu1 is not held)
    }
  }

  //@ requires mu0 != null && mu1 != null;
  //@ requires \max(\lockset) < mu0 && mu0 < mu1;
  void mRace13() {
    double[] a;
    synchronized (mu0) {
      a = d;  // okay
    }
    synchronized (mu1) {
      a = d;  // okay
    }
    synchronized (mu0) {
      synchronized (mu1) {
	a = d;  // okay
	d = a;  // okay
      }
    }
    synchronized (mu1) {
      d = a;  // race condition (mu0 not held)
    }
  }

  //@ requires mu1 != null && \max(\lockset) < mu1;
  void mRace14() {
    double[] a = d;  // okay
    d = a;  // race condition (mu0 not held)
  }

  synchronized int mRace15(SynchHotspot h) { //@ nowarn Deadlock;
    int y = x;  // okay
    if (h == null) {
      return 0;
    } else {
      return y + h.x;  // race condition (h not held)
    }
  }

  void mDeadlock0() {
    synchronized (mu0) {  // trying to lock null
    }
  }

  //@ requires \max(\lockset) < this || \lockset[this];
  synchronized void mDeadlock1() {
    synchronized (this) { // okay
      synchronized (this) { // okay
	if (mu0 != null) {
	  synchronized (mu0) {  // possible deadlock
	  }
	}
      }
    }
  }

  void mDeadlock2() {
    mDeadlock1();  // does not respect locking order
  }

  void mDeadlock3() {
    this.mDeadlock1();  // does not respect locking order
  }

  void mDeadlock4(SynchHotspot h) {
    if (h != null) {
      h.mDeadlock1();  // does not respect locking order
    }
  }

  //@ requires \max(\lockset) < SynchHotspot.class || \lockset[SynchHotspot.class];
  static synchronized void mDeadlock5(SynchHotspot h) {
    if (h == null) {
      mDeadlock5(h);
    } else {
      h.mDeadlock1();  // does not respect locking order
    }
  }

  static void mDeadlock6() {
    mDeadlock5(null);  // does not respect locking order
  }
}
