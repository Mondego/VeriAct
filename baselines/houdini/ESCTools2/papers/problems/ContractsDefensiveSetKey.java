package problems;

/**
 * A set of objects.  This class uses exceptions to indicate
 * what kinds of objects are permissible and thus makes
 * fewer demands of the client.
 *
 * @author Joe Kiniry
 */

/**
 * Just as an example, this defensive set cannot contain the element
 * <code>null</code>.  This shows how you can add a
 * <code>non_null</code> annotation to descendents, but once the
 * annotation is applied to a method, it must be applied to all
 * descendents.
 */

class ContractsDefensiveSetKey extends ContractsSetKey implements ContractsContainerKey
{
  public ContractsDefensiveSetKey() {
    super();
    //@ set cannot_contain_null = true;
  }

  /**
   * {@inheritDoc}
   * @exception NoSuchElementException when the set is empty.
   */
  //@ also
  //@ exceptional_behavior
  //@   requires elementCount() == 0;
  //@   modifies objectState;
  //@   signals (java.util.NoSuchElementException) true;
  public Object getElement() throws java.util.NoSuchElementException {
    if (elementCount() == 0)
      throw new java.util.NoSuchElementException();
    else
      return super.getElement();
  }

  /**
   * {@inheritDoc}
   * @exception IllegalArgumentException if the parameter
   * <code>o</code> is <code>null</code>.
   */
  //@ also
  //@ exceptional_behavior
  //@   requires o == null;
  //@   modifies \nothing;
  //@   signals (IllegalArgumentException) true;
  public void addElement(Object o) throws IllegalArgumentException {
    if (o == null)
      throw new IllegalArgumentException();
    else
      super.addElement(o);
  }
}
