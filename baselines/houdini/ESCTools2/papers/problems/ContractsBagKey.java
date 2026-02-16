package problems;

/**
 * A bag of objects.  Bags can contain multiple incidences of the same
 * object.
 *
 * @author Joe Kiniry
 */

class ContractsBagKey implements ContractsContainerKey
{
  protected /*@ non_null @*/ java.util.Vector my_elements; //@ in objectState;
  //@ maps my_elements.* \into objectState;

  public ContractsBagKey() {
    my_elements = new java.util.Vector();
    //@ set cannot_contain_null = false;
  }

  /**
   * {@inheritDoc}
   */
  public Object getElement() {
    if (elementCount() > 0)
      return my_elements.remove(0);
    else
      return null;
  }

  /**
   * {@inheritDoc}
   */
  public void addElement(Object o) {
    my_elements.add(o);
  }

  /**
   * {@inheritDoc}
   */
  public int elementCount() {
    return my_elements.size();
  }

  /**
   * {@inheritDoc}
   */
  public boolean isEmpty() {
    return (elementCount() == 0);
  }
}
