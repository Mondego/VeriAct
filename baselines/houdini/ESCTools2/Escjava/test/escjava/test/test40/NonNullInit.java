public class NonNullInit {
  /*@ non_null */ Object qq = new Object();
  /*@ non_null */ Object nn;

  NonNullInit(int x) {
    nn = this;  // this initializes "nn"
  }

  NonNullInit(double x) {
    if (x < 0.0) {
      nn = new Object();
    }
  }  // "nn" not initialized if "if" not taken

  NonNullInit(Object x) {
    nn = x;  // this produces a NonNull error
  }

  //@ requires x != null;
  NonNullInit(Object x, Object y) {
    nn = x;
  }

  NonNullInit(int[] a) {
    nn = qq;
  }

  NonNullInit(int x, int y) {
  }  // fails to assign to "nn"

  NonNullInit(double x, double y) {
    this((int)x, (int)y);
  }
}
