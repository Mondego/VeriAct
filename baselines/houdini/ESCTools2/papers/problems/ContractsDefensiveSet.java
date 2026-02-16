package problems;

/**
 * A set of objects.  This class uses exceptions to indicate
 * what kinds of objects are permissible and thus makes
 * fewer demands of the client.
 *
 * @author Joe Kiniry
 */

import java.util.NoSuchElementException;

class DefensiveSet /* extends Bag? Set? implements Container? */
{
  /**
   * {@inheritDoc}
   * @exception NoSuchElementException when?
   * @question When might this exception be thrown?
   * @question Consider how the precondition of this method is
   * different than, say, {@link Set#getElement()}, because of the
   * postcondition.
   */
  public Object getElement() throws NoSuchElementException {
    return null;
  }

  /**
   * {@inheritDoc}
   * @exception IllegalArgumentException when?
   * @question What kinds of arguments are illegal?
   */
  public void addElement(Object o) throws IllegalArgumentException {
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
