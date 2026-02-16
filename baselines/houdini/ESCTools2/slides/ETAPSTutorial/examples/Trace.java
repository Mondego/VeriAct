public class Trace {
  //@ ensures \result > 0;
  int m(int i) {
    if (i == 0) return 1;
    if (i == 2) return 0;
    return 4;
  }
}
