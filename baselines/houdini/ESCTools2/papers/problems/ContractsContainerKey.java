package problems;

/**
 * A contains that holds objects.
 *
 * @author Joe Kiniry
 */

interface ContractsContainerKey
{
  //@ ghost boolean cannot_contain_null;

  //@ initially isEmpty();
  //@ initially elementCount() == 0;
  //@ invariant isEmpty() <==> (elementCount() == 0);
  //@ invariant elementCount() >= 0;

  /**
   * @return any element from this container.
   */
  //@ normal_behavior
  //@   modifies objectState;
  //@   ensures cannot_contain_null <==> (\result != null);
  //@   ensures \old(elementCount() > 0) ==> (elementCount() == \old(elementCount() - 1));
  //@   ensures (\old(elementCount() == 0) && !cannot_contain_null) ==> ((\result == null) && (elementCount() == \old(elementCount())));
  Object getElement();

  /**
   * @param o the object to add to this container.
   */
  //@ normal_behavior
  //@   requires cannot_contain_null <==> (o != null);
  //@   modifies objectState;
  //@   ensures (elementCount() == \old(elementCount() + 1)) || (elementCount() == \old(elementCount()));
  void addElement(Object o);

  /**
   * @return the number of elements in this container.
   */
  /*@ pure @*/ int elementCount();

  /**
   * @return <code>true</code> iff this container is empty.
   */
  /*@ pure @*/ boolean isEmpty();
}
