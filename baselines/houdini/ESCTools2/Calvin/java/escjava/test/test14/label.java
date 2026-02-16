/* Copyright Hewlett-Packard, 2002 */


public class label {
  
  void r0_label_test() {
    int x;
    MyLabel: {
      x = 0;
      if (x == 0)
	break MyLabel;
      x = 1;
    }
    //@ assert x == 0;
  }
  
  void r1_label_test() {
    int x;
    MyLabel: x = 0;
    //@ assert x == 0;
  }
  
  void r2_label_test() {
    int x = 0;
    MyLabel: break MyLabel;
    //@ assert x == 0;
  }
  
  void r3_label_test() {
    int x;
    MyLabel: {
      x = 0;
      if (x == 2)
	break MyLabel;
      if (x == 0)
	break MyLabel;
      x = 1;
    }
    //@ assert x == 0;
  }

  void r4_label_test() {
    MyLabel: ;
  }

  void r5_label_test() {
    { MyLabel: ; }
    { MyLabel: ; }
  }

  void r6_label_test() {
    int x = 0;
    Outer: {
      Inner: {
	break Outer;
      }
      x = 1;
    }
    //@ assert x == 0;
  }

  void r7_label_test() {
    int x = 0;
    Outer: {
      Inner: {
	break Inner;
      }
      x = 1;
    }
    //@ assert x == 1;
  }

  void r8_label_test() {
    int x = 0;
    MyLoop:
    while (x < 10) {
      x++;
    }
    //@ assert x == 10;
  }

  void r9_label_test() {
    int x = 0;
    MyLoop:
    while (x < 10) {
      if (x == 5)
	break;
      x++;
    }
    //@ assert x == 5;
  }
  
  void r10_label_test() {
    int x = 0;
    MyLoop:
    while (x < 10) {
      if (x == 5)
	break MyLoop;
      x++;
    }
    //@ assert x == 5;
  }
  
  void r11_label_test() {
    //@ assert true;
    OuterLoop:
    while (true) {
      InnerLoop:
      while (true) {
	//@ assert true;
      }
    }
  }
}
