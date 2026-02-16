package problems;

/**
 * A contains that holds objects.
 *
 * @author Joe Kiniry
 */

interface Container
{
  /**
   * @return any element from this container.
   */
  Object getElement();

  /**
   * @param o the object to add to this container.
   * @question Can <code>null</code> be added to a container?
   */
  void addElement(Object o);

  /**
   * @return the number of elements in this container.
   */
  int elementCount();

  /**
   * @return <code>true</code> iff this container is empty.
   */
  boolean isEmpty();
}
