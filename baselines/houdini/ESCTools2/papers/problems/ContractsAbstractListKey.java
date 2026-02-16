package problems;

/**
 * A list of objects.
 *
 * @author Joe Kiniry
 */

public abstract class ContractsAbstractListKey implements ContractsContainerKey
{
  /**
   * {@inheritDoc}
   * @return the first element in the list.
   */
  public abstract Object getElement();

  /**
   * {@inheritDoc}
   * @param o the element to add to the end of the list.
   */
  //@ normal_behavior
  //@   modifies objectState;
  public abstract void addElement(Object o);

  /**
   * {@inheritDoc}
   */
  public abstract int elementCount();

  /**
   * {@inheritDoc}
   */
  public boolean isEmpty() {
    return (elementCount() == 0);
  }

  /**
   * Sort the objects in this list.  If the elements of the list are
   * not all subtypes of the class {@link java.lang.Comparable}, then
   * the list is unchanged; otherwise, it is presumed that all
   * elements are comparable to all other elements and the list is
   * sorted according to their "comparable-ness".
   */
  //@ normal_behavior
  //@   modifies objectState;
  public abstract void sort();
}
