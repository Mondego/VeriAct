public class Index {
  void m() {
    int i = 0;
    int j = 8/i;  // Causes a ZeroDiv warning
    Object[] oo = new Object[i-1]; // NegSize warning
    oo = new Object[10];
    i = oo[-1].hashCode(); // IndexNegative warning
    i = oo[20].hashCode(); // IndexTooBig warning
  }
}
