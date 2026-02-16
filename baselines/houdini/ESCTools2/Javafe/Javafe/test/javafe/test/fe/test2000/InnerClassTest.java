/**
 * Test for loop in typechecker when an inner class is a static class
 * and the parent class is not.
 * 
 * @author kiniry
 * @created 21 Aug 2005
 */
public class InnerClassTest {
  public static class StaticInnerClass extends InnerClassTest {
  }
  public class InnerClass extends InnerClassTest {
  }
}
