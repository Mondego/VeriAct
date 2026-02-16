// Test case for Bug #259
//#FLAGS: -Q

public class StringCat {


  final String xxx = "xxx";

  /*@
  normal_behavior
          ensures false;
  @*/
  public String aaa() {
    return "xxx"+"a";
  }

  /*@
  normal_behavior
          ensures false;
  @*/
  public String bbb() {
    return xxx+"a";
  }

  /*@
  normal_behavior
          ensures \result != null;
  @*/
  public String ccc() {
    return "xxx"+"a";
  }

  /*@
  normal_behavior
	ensures \result != null;
  @*/
  public String ddd() {
    return xxx+"a";
  }
}
