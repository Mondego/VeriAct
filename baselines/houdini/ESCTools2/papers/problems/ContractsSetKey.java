package problems;

/**
 * A set of objects.
 *
 * @author Joe Kiniry
 */

class ContractsSetKey extends ContractsBagKey implements ContractsContainerKey
{
  public ContractsSetKey() {
    super();
    //@ set cannot_contain_null = false;
  }

  /**
   * {@inheritDoc}
   * @param o the element to add to this set.
   */
  public void addElement(Object o) {
    if (my_elements.contains(o))
      return;
    else
      super.addElement(o);
  }
}
