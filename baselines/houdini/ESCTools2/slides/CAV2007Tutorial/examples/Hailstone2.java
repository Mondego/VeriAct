public class Hailstone2 {

    /*@   requires 0 < n;
      @   requires n % 2 != 0;
      @   ensures \result == (3*n+1)/2;
      @ also
      @   requires 0 < n;
      @   requires n % 2 == 0;
      @   ensures \result == n/2;
      @*/
    public static /*@ pure @*/ int h(int n)
    {
        if (n % 2 != 0) {
            return (3*n+1)/2;
        } else {
            return n/2;
        }
    }
}
