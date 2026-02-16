/* Copyright Hewlett-Packard, 2002 */

/** This test file produces various warnings to make sure that
 ** the caret positions in the error messages are placed correctly.<p>
 **
 ** This file does not check any synchronization errors.  These are
 ** found in file SynchHotspot.java.
 **/

class HotspotSuper {
  static int gg;  //@ invariant 0 <= gg;

  //@ requires gg != 4;
  HotspotSuper() {
  }

  //@ requires 0 < z;
  HotspotSuper(int z) {
    gg = z;
  }
}

public class Hotspot extends HotspotSuper {
  int ff;
  float flfl;
  double dodo;
  Object nn /*@ non_null */;
  int kk = 6;  /*@ invariant kk == kk;  invariant 0 <= kk; */

  int uuCount /*@ readable_if uuArray != null; */;
  int[] uuArray;
	    
  // The following constructor contains no errors
  Hotspot(int x) /*@ requires 10 <= x && gg != 4;
		     ensures kk == 6; ensures ff == 0; */ {
    nn = this;
  }

  int mNullField0(Hotspot h) {
    return h.ff;
  }

  void mNullField1(Hotspot h) {
    h.ff = 2;
  }

  void mNullField2(Hotspot h) {
    h.ff += 2;
  }

  void mNullField3(Hotspot h) {
    h.ff++;
  }

  void mNullField4(Hotspot h) {
    ++h.ff;
  }

  int mNullField5(int a[]) {
    return a.length;
  }

  int mNullArray0(int a[]) {
    return a[0];
  }
  
  void mNullArray1(int a[]) {
    a[0] = 8;
  }
  
  void mNullArray2(int a[]) {
    a[0] += 8;
  }
  
  void mNullArray3(int a[]) {
    a[0]++;
  }
  
  void mNullArray4(int a[]) {
    ++a[0];
  }

  void mNullCall(Hotspot h) {
    h.mNullCall(h);
  }
  
  void mNullThrow0(Hotspot h) throws Throwable {
    Throwable e = null;
    throw e;
  }

  void mNullThrow1(Hotspot h) throws Throwable {
    Throwable e = null;
    Label: throw e;
  }

  void NullSynchronized(Object mu) {
    synchronized (mu) {
    }
  }
  
  int mBoundsBelow(int a[], int n) {
    if (a != null) {
      return a[n];
    } else {
      return 0;
    }
  }
  
  int mBoundsAbove(int a[], int n) {
    if (a != null && 0 <= n) {
      return a[n];
    } else {
      return 0;
    }
  }

  int mZeroDiv0(int x, int y, float f, double d) {
    f = f / f;  // okay
    d = d / d;  // okay
    return x / y;
  }

  void mZeroDiv1(int x, int y, float f, double d) {
    f /= f;  // okay
    d /= d;  // okay
    x /= y;
  }

  void mZeroDiv2(int y, float f, double d) {
    flfl /= f;  // okay
    dodo /= d;  // okay
    ff /= y;
  }

  void mZeroDiv3(int[] ai, float[] af, double[] ad) {
    if (af != null && af.length != 0) {
      af[0] /= af[0];  // okay
    }
    if (ad != null && ad.length != 0) {
      ad[0] /= ad[0];  // okay
    }
    if (ai != null && ai.length != 0) {
      ai[0] /= ai[0];
    }
  }
  
  int mZeroMod0(int x, int y, float f, double d) {
    f = f % f;  // okay
    d = d % d;  // okay
    return x % y;
  }

  void mZeroMod1(int x, int y, float f, double d) {
    f %= f;
    d %= d;
    x %= y;
  }

  void mZeroMod2(int y, float f, double d) {
    flfl %= f;  // okay
    dodo %= d;  // okay
    ff %= y;
  }

  void mZeroMod3(int[] ai, float[] af, double[] ad) {
    if (af != null && af.length != 0) {
      af[0] %= af[0];  // okay
    }
    if (ad != null && ad.length != 0) {
      ad[0] %= ad[0];  // okay
    }
    if (ai != null && ai.length != 0) {
      ai[0] %= ai[0];
    }
  }
  
  Hotspot mCast(Object o) {
    return (Hotspot)o;
  }

  void mNonNullVar0(Object o) {
    /*@ non_null */ Object x;
    x = o;
  }
  
  void mNonNullVar1(Object o) {
    /*@ non_null */ Object x = o;
  }
  
  void mNonNullField(Object o) {
    nn = o;
  }
  
  void mNonNullParam(Object o, /*@ non_null */ Object p) {
    /*@ non_null */ Object x = p;
    mNonNullParam(o, x);  // this is okay
    mNonNullParam(o, (5 < 2 ? p : o));  // this is not
  }

  void mArrayStore(Object[] a) {
    if (a != null && a.length == 12) {
      a[7] = this;
    }
  }

  void mAssert(int x) {
    //@ assert x < 2;
  }
  
  void mUnreachable(int x) {
    if (x != 2) {
      switch (x) {
	case 0:  break;
	case 1:
	case 2:
	  //@ unreachable;
	  break;
	default:
	  break;
      }
    }
  }

  //@ requires x != 0;
  void mPreRequires0(int x) {
    mPreRequires0(  x -1 );
  }
  
  //@ requires x != 0;
  void mPreRequires1(int x) {
    this.mPreRequires1(x-1);
  }
  
  //@ requires h != null && x != 0;
  void mPreRequires2(Hotspot h, int x) {
    h.mPreRequires2(this, x-1);
  }
  
  //@ requires x != 0;
  static void mPreRequires3(int x) {
    mPreRequires3(x-1);
  }

  void mPreRequires4(int x) {
    Hotspot h = new Hotspot(x);
  }

  Hotspot(Hotspot hnn) {
    super(-5);  // fails to meet HotspotSuper(int) precondition
    if (hnn != null) {
      nn = hnn;
    } else {
      nn = this;
    }
  }
  
  Hotspot(int x, int y) {
    this(x+y);  // fails to meet Hotspot(int) precondition
  }

  Hotspot(double d) {  // fails to meet HotspotSuper() precondition
  }

  void mPreInvariant0() {
    gg = -2;
    mPreInvariant0();
  }
  
  void mPreInvariant1() {
    gg = -2;
    this.mPreInvariant1();
  }
  
  void mPreInvariant2(Hotspot h) {
    gg = -2;
    if (h != null) {
      h.mPreInvariant2(this);
    }
  }

  void mPreInvariant3() {
    gg = -8;
    Hotspot h = new Hotspot(12, 14);
  }

  Hotspot(float f, double d) {
    super(- (gg= -3)); // meets HotspotSuper(int) precondition, but violates
                       // an invariant
  }

  Hotspot(float f0, float f1, float f2) {
    this((float)(gg= -3), f1);  // meets Hotspot(float, double) precondition, but
                                // violates an invariant
  }

  /*@ requires x < 0; ensures \result == 12; */
  int mPostEnsures0(int x) {
    return x;
  }
  
  /*@ ensures ff == 12; */
  void mPostEnsures1(int x) {
    ff = x;
  }

  /*@ exsures (SomeException se) se == null; */
  void mPostExsures0() throws Throwable {
    throw new SomeException();
  }
  
  //@ ensures kk == 0;
  Hotspot(int[] a) {
    super(12);
    nn = this;
  }

  // Conspicuously missing from this test file are methods that violate
  // modifies clauses, but our checker doesn't currently support modifies
  // checking.
  
  void mPostInvariant0(int x) {
    gg = x;
  }

  void mPostInvariant1(int x) {
    kk = x;
  }

  Hotspot(double[][] aad) {
    this((int[])null);
    if (aad != null && 15 < aad.length && aad[10] != null && aad[10].length != 0) {
      kk = (int)aad[10][0];  // may fail to establish invariant
    }
  }
  
  Hotspot(double[][] aad, int x) {
    this((int[])null);
    gg = x;  // may fail to establish invariant
  }

  void mPostException0(RuntimeException e) {
    if (e != null) {
      throw e;
    }
  }

  void mPostException1(RuntimeException e) {
    if (e != null) {
      Label: throw e;
    }
  }

  void mUninit0() {
    int x = 13 /*@ uninitialized */;
    int y = 2;
    if (x < y) {
      y++;
    }
  }
  
  void mUninit1(int z, int w) {
    int x = 0 /*@ uninitialized */;
    int y = 0 /*@ uninitialized */;
    if (z == 10) {
      y = 68;
    } else {
      y = 72;
    }
    if (y < 70) {  // okay
      x = 20;
    }
    if (z == 10 && x < w) {  // okay
      y++;
    }
    x++;  // possible read of meaningless value
  }

  int mUndef0(int[] a) {
    /*@ readable_if a != null; */ int c;
    c = 12;    // always okay to write to c
    return c;  // but can read c only if a is non-null
  }

  int mUnwrit0(int[] a) {
    /*@ writable_if a != null; */ int c;

    if (a==null)
	c = 3;    // error: not ok to write to c if a==null
    else
	c = 12;    // ok here
    return c;  // but can read c always
  }
  
  int mUndef1(int[] a, int[] b) {
    /*@ readable_if a != null; */ int c = 12;
    if (a != null) {
      return c;
    }
    a = new int[3];
    int z = c;   // okay, since a is now non-null
    c = a[2];
    a = b;
    return z + c;
  }
  
  int mUndef2() {
    uuCount = 12;    // always okay to write to uuCount
    return uuCount;  // but can read uuCount only if uuArray is non-null
  }

  void mUndef3() {
    int[] a = new int[10] //@ readable_if gg < 10;
      ;
    if (gg == 7) {
      a[0] += 1;  // okay
    }
    a[1] *= 20;  // not okay
  }

  int mUndef4() {
    boolean b;
    int c /*@ readable_if b; */;
    b = false;
    c = 0;
    return c;
  }

  /*****************
  void mUndef5() {
    //@ readable_if 0 <= c;
    int c = 6;
    c++;  // okay
    c -= 8;  // okay
    c++;  // not okay
  }
  *****************/

  void mLoopInvInitWhile() {
    int x = 0;
    //@ loop_invariant x != 0;
    while (true) {
    }
  }

  void mLoopInvIterWhile() {
    int x = 1;
    //@ loop_invariant x != 0;
    while (true) {
      x = 0;
    }
  }

  void mLoopInvInitDo0() {
    int x = 0;
    //@ loop_invariant x != 0;
    do {
    } while (true);
  }

  void mLoopInvInitDo1() {
    int x = 0;
    //@ loop_invariant x != 0;
    Label: do {
    } while (true);
  }

  void mLoopInvIterDo0() {
    int x = 1;
    //@ loop_invariant x != 0;
    do {
      x = 0;
    } while (true);
  }

  void mLoopInvIterDo1() {
    int x = 1;
    //@ loop_invariant x != 0;
    Label: do {
      x = 0;
    } while (true);
  }

  void mLoopInvInitFor() {
    //@ loop_invariant x != 0;
    for (int x = 0; x < 10; x++) {
    }
  }

  void mLoopInvIterFor() {
    //@ loop_invariant x == 0;
    for (int x = 0;; x++) {
    }
  }
}

class HotspotInvariant0 {
  int hx;  //@ invariant hx == 0;
  // the default constructor fails to establish the following invariant
  int hs;  //@ invariant hs == 3;
}

class HotspotInvariant1 {
  int hx = 2;  //@ invariant hx == 2;
  // the default constructor fails to establish the following invariant
  int hs = 2;  //@ invariant hs == 3;
}

class SomeException extends Throwable {
}
