package problems;

/**
 * A list of objects implemented as an array.
 *
 * @author Joe Kiniry
 */

class ContractsArrayListKey extends ContractsAbstractListKey implements ContractsContainerKey
{
  public int MAX_SIZE = 1000;
  
  /*@ spec_public @*/ private Object[] my_list; //@ in objectState;
  //@ maps my_list[*] \into objectState;
  //@ invariant \nonnullelements(my_list);
  //@ invariant my_list.length == MAX_SIZE;

  private int elementCount; //@ in objectState;
  //@ private invariant 0 <= elementCount && elementCount < MAX_SIZE;

  public ContractsArrayListKey() {
    my_list = new Object [MAX_SIZE];
    //@ set cannot_contain_null = true;
    //@ set my_list.owner = this;
  }

  /**
   * {@inheritDoc}
   */
  public Object getElement() {
    if (elementCount == 0)
      return null;
    Object result = my_list[0];
    System.arraycopy(my_list, 1, my_list, 0, elementCount-1);
    elementCount--;
    return result;
  }

  /**
   * {@inheritDoc}
   */
  //@ also
  //@ normal_behavior
  //@   requires o != null;
  //@   requires elementCount() < MAX_SIZE-1;
  //@   requires \typeof(o) <: \elemtype(\typeof(my_list));
  //@   ensures my_list[\old(elementCount)] == o;
  //@   ensures elementCount == \old(elementCount + 1);
  public void addElement(Object o) {
    if (elementCount() < MAX_SIZE-1) {
      my_list[elementCount()] = o;
      elementCount++;
    }
  }

  /**
   * {@inheritDoc}
   */
  //@ also
  //@ normal_behavior
  //@   ensures \result == elementCount;
  public int elementCount() {
    return elementCount;
  }

  /**
   * {@inheritDoc}
   */
  public boolean isEmpty() {
    return (elementCount() == 0);
  }

  /**
   * {@inheritDoc}
   */
  //@ also
  //@ normal_behavior
  //@   requires (\forall int i; 0 <= i && i < elementCount(); \typeof(my_list[i]) <: \type(Comparable));
  //@   modifies my_list[*];
  //@   ensures (\forall int i; 0 <= i && i < elementCount(); ((Comparable)my_list[i]).compareTo(((Comparable)my_list[i+1])) < 0);
  public void sort() {
    for (int i = 0; i < elementCount; i++)
      if (!((my_list[i] == null) || (my_list[i] instanceof Comparable)))
        return;
    java.util.Arrays.sort(my_list);
  }
}
