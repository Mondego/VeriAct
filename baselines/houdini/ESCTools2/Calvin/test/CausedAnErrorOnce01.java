/* Copyright Hewlett-Packard, 2002 */


class CausedAnErrorOnce01 {
  static int classvar;
  int instancevar;

  public static int m1(fieldref o) {
    //@ assume o != null
    return fieldref.classvar + o.classvar + o.instancevar;
  }

  public int m2() {
    return instancevar + this.instancevar + super.instancevar;
  }
}
