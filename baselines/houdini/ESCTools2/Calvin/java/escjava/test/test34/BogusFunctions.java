/* Copyright Hewlett-Packard, 2002 */

// These tests test the argument types and return types of special ESC/Java functions

class BogusFunctions {
  //@ ghost public \TYPE tt;
  //@ ghost public int ii;
  //@ ghost public int[][] ga;
  Cloneable[] aa;
  
  void mTypes() {
    //@ assert \type(BogusFunctions) <: \type(Object);
    //@ assert tt <: (\TYPE)tt;
    //@ assert \elemtype(tt) <: tt;
    //@ set tt = tt <: (\TYPE)tt ? \type(BogusFunctions[]) : \typeof(this);
    //@ assert \elemtype(\typeof(this)) == \elemtype(\elemtype(\elemtype(\type(Object))));

    /*@ set ii = \type(Cloneable); */  // illegal
    /*@ assert (\type(Object) <: \typeof(this)) == 12; */ // illegal
    /*@ assert \type(java.lang.Object[]); */  // illegal
    /*@ assert \elemtype(\type(java.lang.Object[])); */  // illegal
    /*@ assert \elemtype(5 < 5) == \elemtype((\TYPE)this); */  // illegal
  }

  //@ ghost public \TYPE[][] mENNghost0;
  //@ ghost public \TYPE[] mENNghost1;
  void mElemsnonnull() {
    Object[] a = new Object[0];
    //@ assert \nonnullelements(a);
    //@ assert \nonnullelements(ga);
    //@ assert \nonnullelements(aa);
    //@ assert \nonnullelements(mENNghost0);

    int[] b = null;
    /*@ assert \nonnullelements(b); */  // illegal
    /*@ assert \nonnullelements(12); */  // illegal
    /*@ assert \nonnullelements(12 < 13); */  // illegal
    /*@ assert \nonnullelements(a) == 12; */  // illegal
    //@ assert \nonnullelements(mENNghost1);
  }

  //@ ensures \fresh(this);
  //@ ensures \fresh(null);
  /*@ ensures \fresh(tt); */  // illegal
  /*@ ensures \fresh(ii); */  // illegal
  /*@ ensures \fresh(); */  // illegal
  /*@ ensures \fresh(this, null); */  // illegal
  void mFresh() {
  }

  //@ modifies ii, tt
  //@ ensures \old(ii) == ii;
  //@ ensures tt == \old(tt);
  //@ ensures (\forall BogusFunctions bf; bf.aa != null) ==> \old((\forall BogusFunctions bf; bf.aa != null));
  /*@ ensures \old(ii) == \old(tt); */  // illegal
  /*@ ensures \old() == \old(ii, ii); */  // illegal
  void mPRE() {
  }

  void mMutexOps() {
    int[] a = new int[1];
    //@ assume \lockset[this];
    //@ assume \max(\lockset) <= \max(\lockset);
    //@ assume \max(\lockset) < \max(\lockset);
    //@ assume \max(\lockset) == \max(\lockset);
    //@ assume \max(\lockset) != \max(\lockset);
    //@ assume (\max(\lockset) < \max(\lockset)) == (3 < 4);

    /*@ assume (\max(\lockset) < 4) == (3 < \max(\lockset)); */  // illegal
    /*@ assume \max() == \max(this); */  // illegal
    /*@ assume \max(\lockset) > \max(\lockset); */  // illegal
    /*@ assume \max(\lockset) >= \max(\lockset); */  // illegal
    /*@ assume \lockset[12]; */  // illegal
    /*@ assume a[this] == 12; */  // illegal
    /*@ assume \lockset[this] == 12; */  // illegal
    /*@ assume \max(12, 13) == 13; */  // illegal
    /*@ assume (\max(\lockset) < \max(\lockset)) == 3; */  // illegal
  }

  // All of the following are allowed:
  //@ requires null < this;
  //@ requires null < null;
  //@ requires this < null;
  //@ requires o < this && this < null && null < o;
  //@ requires this < o && null < this && o < null;
  //@ requires null <= this;
  //@ requires null <= null;
  //@ requires this <= null;
  //@ requires o <= this && this <= null && null <= o;
  //@ requires this <= o && null <= this && o <= null;
  //@ requires \lockset[null];
  void nullBelow(Object o) {
  }
}
