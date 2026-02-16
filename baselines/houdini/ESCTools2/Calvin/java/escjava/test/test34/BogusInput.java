/* Copyright Hewlett-Packard, 2002 */

// This class contains many errors that the type checker should catch
// The tests are designed to pass the parser, but many fail the type checker.

/*@ uninitialized */
/*@ non_null */
/*@ readable_if x == 5 */
/*@ writable_if x == 5 */
/*@ monitored_by null, true */
/*@ monitored */
class BogusInput {
  //@ axiom this != null
  
  /*@ readable_if this != null */  // this is allowed
  /*@ writable_if this != null */  // this is allowed
  int x;
  //@ axiom x == 5

  //@ readable_if x < 10
  //@ writable_if x < 10
  static int g;
  /*@ axiom g == 12 */  // this is allowed

  //@ axiom \lockset == \lockset
  //@ invariant \lockset == \lockset
  //@ axiom \max(\lockset) == null;
  //@ invariant \max(\lockset) == null

  //@ axiom \result == \result;
  //@ invariant \result == \result

  //@ axiom \old(x) == x && (\forall BogusInput t; t.x == t.x);
  //@ invariant \old(x) == x && (\forall BogusInput t; t.x == t.x);
  
  //@ requires \result == \result
  //@ modifies a[ (\result == \result ? 0 : 1 )]
  //@ ensures \result == \result
  BogusInput(int[] a) {
  }

  //@ requires \result == 0
  //@ modifies a[\result]
  //@ ensures \result == 0
  int m(int[] a) {
  }

  //@ requires \result == 0
  //@ modifies a[\result]
  //@ ensures \result == 0
  //@ also_modifies a[\result]
  //@ also_ensures \result == 0;
  int notAMethod;
  
  /*@ uninitialized */ int uninitA;
  /*@ uninitialized */ int uninitB = 5;
  /*@ uninitialized */ static int uninitC;
  /*@ uninitialized */ static int uninitD = 5;
  /*@ uninitialized */ int uninitE, uninitF;
  /*@ uninitialized */ int uninitM() { return 0; }
  void uninit(/*@ uninitialized */ int p) {
    /*@ uninitialized */ int a;
    /*@ uninitialized */ int b = 0;
    /*@ uninitialized */ int c = 0, d;
    /*@ uninitialized */ int e = 0, f = 0;
    
    // The following are from ESCJ 17:
    /*@ uninitialized */ Object x = null, y = new Object();
    /*@ uninitialized */ Object w, z = null;
    /*@ uninitialized */ Object v;
  }

  /*@ non_null */ int nn0;
  int nn1 /*@ non_null */;
  /*@ non_null */ Object nn2;
  Object nn3 /*@ non_null */;
  /*@ non_null */ Cloneable nn4;
  /*@ non_null */ int[] nn5, nn6;
  /*@ non_null */ int nn7[], nn8;
  /*@ non_null */ static int nn9;
  /*@ non_null */ static BogusInput nn10;
  /*@ non_null */ static BogusInput nn11 = null;  // ironically, the type checker allows this one
  /*@ non_null */ static BogusInput nn12 = new BogusInput(null);
  /*@ non_null */ static BogusInput nn13, n14 = new BogusInput(null), n15;
  /*@ non_null */ /*@ non_null */ /*@ non_null non_null */ Object n16;
  /*@ non_null */ Object nonNullM() { return null; }
  void nonNull(/*@ non_null */ int p, int q, /*@ non_null */ Object r) {
    /*@ non_null */ int a;
    /*@ non_null */ Object b;
    /*@ non_null */ Object c = null;
    /*@ non_null */ Object d = null, e = new Object(), f;
    Object g = null, h = new Object(), i /*@ non_null */ ;
    /*@ non_null */ int[] j;
    int[] k /*@ non_null */;
    /*@ non_null */ int l[];
    int m[] /*@ non_null */;
    /*@ uninitialized non_null */ int[] n = null;
  }

  /*@ readable_if true; */ int defif0;
  /*@ readable_if defif0 == defif1; */ int defif1;
  /*@ readable_if true; */ static int defif2;
  /*@ readable_if defif0 == defif3; */ static int defif3;
  /*@ readable_if defif4 || defif5; */ boolean defif4, defif5;  // this is allowed
  /*@ readable_if true; */ int definedifM() { return 0; }
  void definedIf0(/*@ readable_if true; */ int x) {
    /*@ readable_if true; */ int a;
    /*@ readable_if a == 2; */ int b;
    /*@ readable_if 0 <= c; */ int c;
    /*@ readable_if a == 2; */ int d = 5;
    /*@ readable_if this == null || defif0 == defif2; */ int e;
  }
  static void definedIf1() {
    /*@ readable_if this == null || defif0 == defif2; */ int e;
  }

  void definedIf2() {
    /*@ readable_if defif0 == 3; */ boolean defif0;  // this is fine
    /*@ readable_if a; */ boolean a;  // this is not fine
    /*@ readable_if a; */ boolean x, y;  // this is fine
    /*@ readable_if c; */ boolean b, c, d;  // this is not fine, for any of the variables
  }
  
  Object[] mus;
  /*@ monitored_by mus[0], this, undeclaredVariable, this.mus[1]; monitored_by mb0 */ int mb0;
  /*@ monitored_by mus, this, this.mus */ static int mb1;
  /*@ monitored_by mus */ void monitoredBy0(/*@ monitored_by this */ int p) {
    /*@ monitored_by this */ int x;
  }
  /*@ monitored_by mb1 */ static void monitoredBy1(/*@ monitored_by mb1 */ int p) {
    /*@ monitored_by mb1 */ int x;
    /*@ monitored_by this */ int y;
  }

  /*@ monitored */ int mo0;
  /*@ monitored monitored */ /*@ monitored */ /*@ monitored_by this */ int mo1 /*@ monitored */;
  /*@ monitored */ static int mo2;
  /*@ monitored */ void monitored0(/*@ monitored */ int p) {
    /*@ monitored */ int x;
  }
  /*@ monitored */ static void monitored1(/*@ monitored */ int p) {
    /*@ monitored */ int x;
  }

  static void pp() {
    this.pp();  // "this" is not allowed in this context
  }

  int[] iarr;
  //@ requires mus[*] == null;
  //@ modifies mus[*];
  //@ modifies iarr[iarr[*]];
  //@ ensures mus[*] == null;
  void wildRef() {
  }

  static int xx;
  //@ axiom new Object(blah);
  //@ axiom new int[5];
  //@ axiom (xx = 0) == 0;
  //@ axiom (xx *= 0) == 0;
  //@ axiom (xx /= 0) == 0;
  //@ axiom (xx %= 0) == 0;
  //@ axiom (xx += 0) == 0;
  //@ axiom (xx -= 0) == 0;
  //@ axiom (xx <<= 0) == 0;
  //@ axiom (xx >>= 0) == 0;
  //@ axiom (xx >>>= 0) == 0;
  //@ axiom (xx &= 0) == 0;
  //@ axiom (xx |= 0) == 0;
  //@ axiom (xx ^= 0) == 0;
  //@ axiom (++xx) == 0;
  //@ axiom (xx++) == 0;
  //@ axiom (--xx) == 0;
  //@ axiom (xx--) == 0;

  void mPredicates() {
    // These are okay
    /*@ assert (\forall int i; i < i+1); */
    /*@ assert (\exists int i; i*i < 0); */
    /*@ assert (\lblneg Hello 1 < 2); */
    /*@ assert (\lblpos Hello 1 < 2); */
    
    // Argument to cast cannot be a predicate
    /*@ assert (boolean)(\forall int i; i < i+1); */
    /*@ assert (boolean)(\exists int i; i*i < 0); */
    /*@ assert (boolean)(\lblneg Hello 1 < 2); */
    /*@ assert (boolean)(\lblpos Hello 1 < 2); */
    
    // Arguments to ? : cannot be predicates
    /*@ assert (\forall int i; i < i+1) ? (\forall int i; i < i+1) : (\forall int i; i < i+1); */
    /*@ assert (\exists int i; i*i < 0) ? (\exists int i; i*i < 0) : (\exists int i; i*i < 0); */
    /*@ assert (\lblneg Hello 1 < 2) ? (\lblneg Hello 1 < 2) : (\lblneg Hello 1 < 2); */
    /*@ assert (\lblpos Hello 1 < 2) ? (\lblpos Hello 1 < 2) : (\lblpos Hello 1 < 2); */

    // These are okay
    int xyz = 0;
    /*@ assert (\forall int i; i < 0 ? i < i*i : i <= i*i); */
    /*@ assert (\exists int i; i < 0 ? false : i == i*i); */
    /*@ assert (\lblneg MyLabel xyz == 0); */
    /*@ assert (\lblpos MyLabel xyz == 0); */

    // Name clashes
    /*@ assume (\forall int i; (\forall int i; i == i)); */
    /*@ assume (\forall int i; (\forall Object j; (\exists double i; i == i))); */
    /*@ assume (\forall int i, i; (\forall Object j; (\exists double i; i == i))); */
    /*@ assume (\forall int i, j; (\forall Object j; (\exists double i; i == i))); */
    /*@ assume (\forall char xyz; xyz == xyz); */
  }
}
