
class BogusGhost {
  //@ ghost public int a = 5
  //@ ghost int b
  //@ ghost protected int c
  //@ ghost public int[][] d[][];
  //@ ghost public static int e;
  //@ ghost static public int f;
  //@ ghost final public int g;
  //@ ghost static final public int h;
  //@ ghost public \TYPE i;
  /*@ ghost public \TYPE [] [ ] j [] [][ ]; */
  //@ ghost public BogusGhost k;
  //@ ghost public int l;
  //@ ghost static public int l;
  int m;
  //@ ghost public int m;
  int n;
  //@ ghost public static int n;
  static int o;
  //@ ghost public int o;
  static int p;
  //@ ghost public static int p;
  //@ ghost public int ghost;
  
  //@ invariant k != null;

  //@ ghost public int x;
}

interface BogusGhostInterfaceA {
  //@ ghost public int x;
}

interface BogusGhostInterfaceB {
  //@ ghost public int x;
}

interface BogusGhostInterfaceC extends BogusGhostInterfaceA, BogusGhostInterfaceB {
  //@ ghost public int x;
}

class BogusGhostSub extends BogusGhost {
  //@ ghost public int x;
}

class BogusGhostSubSubA extends BogusGhostSub implements BogusGhostInterfaceA, BogusGhostInterfaceB {
  //@ ghost public int x;
}

class BogusGhostSubSubB extends BogusGhostSub implements BogusGhostInterfaceA, BogusGhostInterfaceB {
  //@ ghost public int x;
}

class BogusGhostX extends BogusGhostSub implements BogusGhostInterfaceA, BogusGhostInterfaceC {
  //@ ghost public int x;
}

class BogusGhostXX extends BogusGhostX {
  //@ ghost public int x;
}

class BogusGhostXXX extends BogusGhostXX implements BogusGhostInterfaceB {
  int x;
}

class GhostSet {
  //@ ghost public static int x;
  //@ ghost public int y;
  //@ ghost public \TYPE t;
  //@ ghost public double[] d;
  
  void m0() {
    // The following two "set" pragmas are legal
    //@ set x = 0
    //@ set y = 0
  }

  static void m1() {
    /*@ set x = 0; */  // legal
    /*@ set y = 0; */   // not legal
  }

  //@ modifies t
  //@ ensures \old(t == \type(int)) ==> t == \type(double);
  void m2() {
    //@ set t = \type(GhostSet);
    //@ set t = \type(int[]);
    //@ set t = \type(\TYPE);
    //@ set t = \type(\TYPE[]);
    //@ set t = \type(double);
  }

  int z;
  //@ ghost public int[] w;
  
  int m3() {
    /*@ set z = 5; */            // not legal
    /*@ set w = new int[10]; */  // not legal
    /*@ set w[z] = 3; */         // not legal
  }

  //@ ghost public boolean bo;
  void m4() {
    /*@ set bo = (\forall int i; i < i+1); */
    /*@ set bo = (\exists int i; i*i < 0); */
    /*@ set bo = (\lblneg Hello 1 < 2); */
    /*@ set bo = (\lblpos Hello 1 < 2); */
  }
}
