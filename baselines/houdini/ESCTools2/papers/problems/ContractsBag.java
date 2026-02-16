package problems;

/**
 * A bag of objects.  Bags can contain multiple incidences of the same
 * object.
 *
 * @author Joe Kiniry
 */

class Bag /* extends Set? */ implements Container
{
  /**
   * {@inheritDoc}
   */
  public Object getElement() {
    return null;
  }

  /**
   * {@inheritDoc}
   * @question How does this method relate to {@link Set#addElement(Object)}?
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
