
class CausedAnErrorOnce00 {
  static int classvar;
  int instancevar;

  public static int m1(CausedAnErrorOnce00 o) {
    //@ assume o != null;
    return CausedAnErrorOnce00.classvar + o.classvar + o.instancevar;
  }

  public int m2() {
    return instancevar + this.instancevar + super.instancevar;
  }
}
