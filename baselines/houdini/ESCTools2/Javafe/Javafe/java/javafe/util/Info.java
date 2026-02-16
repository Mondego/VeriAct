/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

/**
 * The <code>Info</code> class is responsible for displaying verbose
 * and debugging information to the user.
 *
 * <p> Information is displayed only if the <code>on</code> flag is set
 * (via the <code>-v</code> option, defaults to unset).
 *
 * <p> Currently, information is displayed by printing it to
 * <code>System.out</code>.
 *
 * <p> Future versions of this class may support identifying
 * information with the section it comes from; verbose control could
 * then be allowed on a section-by-section basis.
 */

public class Info
{
  // Prevent javadoc from displaying a public constructor
  private Info() {}

  // Class Variables

  /**
   * Verbose and debugging information is displayed iff this is true.
   * Defaults to false.
   */
  public static boolean on = false;

  // Reporting information

  /**
   * Report verbose or debugging information if <code>on</code> is
   * set.
   * 
   * <p> Precondition: <code>msg</code> is not null.
   *
   * <p> The message is displayed directly, without any indication
   * that it is verbose or debugging information.
   *
   * <p> Clients of this routine may wish to place calls to it
   * within a conditional on <code>on</code>.  For example,
   *
   * <code>
   *    if (Info.on) Info.out("[total count = " + countNodes() + "]");
   * </code><p>
   *
   * <p> This may be especially useful if <code>countNodes()</code>
   * is an expensive operation.
   */
  //@ requires msg != null;
  public static void out(String msg) {
    if (on) {
      System.out.println(msg);
      System.out.flush();
    }
  }
}
