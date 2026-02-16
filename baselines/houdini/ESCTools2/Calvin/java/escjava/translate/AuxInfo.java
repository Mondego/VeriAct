/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

/** This class provides an interface through which one can create a
  * unique ID for given references.  It is used by ESC/Java to associate
  * more information with a given asserted condition.
  **/

import javafe.util.Assert;

public class AuxInfo {
  
  /** Initializes the data structures, clearing any previous information
    * of which IDs have been used.  This method should be called before
    * any other in this class.  The method may also free and null out
    * certain structures, aiming to help the garbage collector.
    **/
  
  public static void reset() {
    // drop all but the first buffer
    AuxInfoLink d = first;
    while (d != null) {
      d = d.clean();
    }
    last = first;
    n = 0;
    usedInLast = 0;
  }

  /** Creates and returns an ID for the reference <code>o</code>.
    * The returned ID has not been returned since the last call to
    * <code>reset</code>, even if this method has been called for
    * the same <code>o</code> since then.
    **/

  //@ ensures 0 <= RES;
  public static int put(/*@ non_null */ Object o) {
    if (usedInLast == AuxInfoLink.LINK_BUFFER_SIZE) {
      last.next = new AuxInfoLink();
      last = last.next;
      usedInLast = 0;
    }
    //@ assert usedInLast < AuxInfoLink.LINK_BUFFER_SIZE;
    int id = n;
    last.a[usedInLast] = o;
    usedInLast++;
    n++;
    return id;
  }

  /** Returns the reference associated with <code>id</code>, as returned
    * by <code>put</code> since the last call to <code>reset</code>
    * It is assumed that <code>id</code> has indeed been returned by
    * <code>put</code> since the last call to <code>reset</code>.
    **/
  
  //@ requires 0 <= id;
  //@ ensures RES != null;
  public static Object get(int id) {
    // first find containing buffer
    AuxInfoLink d = first;
    while (AuxInfoLink.LINK_BUFFER_SIZE <= id) {
      //@ loop_invariant d != null;
      //@ loop_invariant 0 <= id;
      //@ assume d.next != null;
      d = d.next;
      id -= AuxInfoLink.LINK_BUFFER_SIZE;
    }
    Object o = d.a[id];
    // Assert.notNull(o); this assert does fail!
    //@ assume o != null;
    return o;
  }

  private static /*@ non_null */ AuxInfoLink first = new AuxInfoLink();
  private static /*@ non_null */ AuxInfoLink last = first;
  //@ invariant last.next == null;
  private static int n = 0;
  //@ invariant 0 <= n;
  private static int usedInLast = 0;
  //@ invariant 0 <= usedInLast && usedInLast <= AuxInfoLink.LINK_BUFFER_SIZE;
  //@ invariant usedInLast <= n;
}

class AuxInfoLink {
  static final int LINK_BUFFER_SIZE = 1024;
  /*@ non_null */ Object[] a = new Object[LINK_BUFFER_SIZE];
  //@ invariant a.length == LINK_BUFFER_SIZE;
  //@ invariant typeof(a) == type(Object[]);
  AuxInfoLink next = null;

  //@ modifies this.next;
  //@ ensures RES == PRE(this.next);
  //@ ensures this.next == null;
  AuxInfoLink clean() {
    // try to help garbage collector
    for (int i = 0; i < a.length; i++) {
      a[i] = null;
    }
    AuxInfoLink n = next;
    next = null;
    return n;
  }

  //@ ensures next == null;
  AuxInfoLink() {
  }
}
