/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.ast;

/** Various utility methods. */

public class Util {

  /** Returns the size of the AST <code>n</code>.  The size
    * includes the size of all children nodes, even children
    * nodes that are <code>null</code>.
    **/
  //@ ensures 0 < \result;
  public static int size(/*@ non_null */ ASTNode n) {
    int t = 1;  // count this node

    int k = n.childCount();
    for (int i = 0; i < k; i++) {
      Object c = n.childAt(i);

      if (c instanceof ASTNode) {
	t += size((ASTNode)c);
      } else {
	t += 1;
      }
    }

    return t;
  }

  /** Returns the size of the AST <code>n</code> (see above),
    * except returns -1 if the size exceeds <code>limit</code>.
    **/
  //@ requires 0 <= limit;
  //@ ensures \result == -1 || (0 < \result && \result <= limit);
  public static int size(/*@ non_null */ ASTNode n, int limit) {
    if (limit == 0) {
      return -1;  // limit exceeded
    }
    int t = 1;  // count this node
    limit -= 1;

    int k = n.childCount();
    // loop_invariant t + limit == METHOD_PRE(limit);
    // loop_invariant 0 < t && t <= METHOD_PRE(limit);
    //@ loop_invariant 0 <= limit;
    for (int i = 0; i < k; i++) {
      Object c = n.childAt(i);

      int s;
      if (c instanceof ASTNode) {
	s = size((ASTNode)c, limit);
	if (s == -1) {
	  return -1;  // limit exceeded while counting children
	}
      } else {
	if (limit == 0) {
	  return -1;  // limit exceeded
	}
	s = 1;
      }
      //@ assert s <= limit;
      t += s;
      limit -= s;
    }

    return t;
  }
}
