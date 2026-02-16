package problems;

/**
 * A set of objects.
 *
 * @author Joe Kiniry
 */

class Set /* extends Bag? DefensiveSet? */ implements Container
{
  /**
   * {@inheritDoc}
   */
  public Object getElement() {
    return null;
  }

  /**
   * {@inheritDoc}
   * @question How does this method relate to {@link Bag#addElement(Object)}?
   */
  public void addElement(Object o) {
  }

  /**
   * {@inheritDoc}
   */
  public int elementCount() {
    return 0;
  }

  /**
   * {@inheritDoc}
   */
  public boolean isEmpty() {
    return false;
  }
}
