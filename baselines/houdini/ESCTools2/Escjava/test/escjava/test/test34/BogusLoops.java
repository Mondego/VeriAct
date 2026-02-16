class BogusLoops {
  int loopinvs(int x) {
    int y = 3;
    int[] a = new int[10];

    //@ loop_invariant x < x+2
    //@ loop_invariant y == y
    while (x++ < y) {
      //@ assert 0 < x
      //@ unreachable
      //@ assume 0 < y
      // Loop invariants cannot be placed here:
      //@ loop_invariant a != null
      //@ loop_invariant \old(x) <= x;
      a[0] = y;
      // A loop invariant cannot be placed here:
      //@ loop_invariant a[0] == 0;
    }

    //@ loop_invariant x < x+2;
    //@ loop_invariant y == y;
    do {
      //@ assert 0 < x
      //@ unreachable
      //@ assume 0 < y
      // Loop invariants cannot be placed here:
      //@ loop_invariant a != null;
      //@ loop_invariant \old(x) <= x;
      a[0] = y;
      // A loop invariant cannot be placed here:
      //@ loop_invariant a[0] == 1;
    } while (x++ < y);
    
    //@ loop_invariant x < x+2;
    //@ loop_invariant y == y;
    for (int i = loopinvs(10); i < 20; i++) {
      //@ assert 0 < x
      //@ unreachable
      //@ assume 0 < y
      // Loop invariants cannot be placed here:
      //@ loop_invariant a != null;
      //@ loop_invariant \old(x) <= x;
      a[0] = y;
      // A loop invariant cannot be placed here:
      //@ loop_invariant a[0] == 2;
    }

    while (x++ < y) {
    }
    //@ loop_invariant a[0] == 3;

    //@ loop_invariant 2 < y;
    while (x++ < y) {
      a[1] = x;
      //@ loop_invariant 3 < y;
      //@ loop_invariant 4 < y;
      do {
	//@ loop_invariant 5 < y;
	for (x++, y++; y < 20; x++, y++) {
	  {}
	  //@ loop_invariant a[0] == 4;
	}
      } while (x++ < y);
    }
    
    while (false)
      ;
    //@ loop_invariant a[0] == 5;

    while (x++ < y) {
      ;
      //@ loop_invariant a[0] == 6;
    }

    
    while (x++ < y) {
      MyLabel: { //@ loop_invariant a[0] == 7;
      }
      //@ loop_invariant a[0] == 8;
    }

    //@ loop_invariant a[0] == 9;
    while (x++ < y) {
      int z = 0;
    }
    
    //@ loop_invariant a[0] == 10;
    while (x++ < y) {
      int z;
    }
    
    return 0;
  }

  void goodLoops(int x) {
    int y = 0;
    int[] a = new int[20];
    //@ loop_invariant \lockset == \lockset;
    //@ loop_invariant \fresh(a) ==> \old(x+y) == x+y;
    while (x++ < y) {
    }
  }
}
