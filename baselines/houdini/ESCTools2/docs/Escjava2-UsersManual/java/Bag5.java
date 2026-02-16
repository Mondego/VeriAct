package eu.etaps.tutorial.bags;

/**
 * A bag of integers.
 *
 * @author The DEC SRC ESC/Java research teams (noemail@hp.com)
 * @author Joe Kiniry (kiniry@acm.org)
 * @version ETAPS-01042007
 */
class Bag5 {
  /** A representation of the elements of this bag of integers. */
  private /*@ spec_public non_null */ int[] my_contents; //@ in objectState;
  //@ maps my_contents[*] \into objectState;

  /** This size of this bag. */
  private /*@ spec_public */ int my_bag_size; //@ in objectState;
  //@ invariant 0 <= my_bag_size && my_bag_size <= my_contents.length;

  //@ public ghost boolean empty; in objectState;
  //@ invariant empty == (my_bag_size == 0);

  /**
   * Build a new bag, copying <code>input</code> as its initial contents.
   * @param the_input the initial contents of the new bag.
   */
  //@ public behavior
  //@   requires the_input != null;
  //@   assignable objectState;
  //@   ensures empty == (the_input.length == 0);
  //@   signals (Exception) false;
  public Bag5(final int[] the_input) {
    my_bag_size = the_input.length;
    my_contents = new int[my_bag_size];
    System.arraycopy(the_input, 0, my_contents, 0, my_bag_size);
    //@ set empty = my_bag_size == 0;
  }

  /** @return if this bag is empty. */
  //@ public behavior
  //@   ensures \result == empty;
  //@   signals (Exception) false;
  public /*@ pure */ boolean isEmpty() {
    return my_bag_size == 0;
  }

  /** @return the minimum value in this bag and remove it from the bag. */
  //@ public behavior
  //@   requires !empty;
  //@   assignable objectState;
  //@   signals (Exception) false;
  public int extractMin() {
    int m = Integer.MAX_VALUE;
    int mindex = 0;
    for (int i = 0; i < my_bag_size; i++) {
      if (my_contents[i] < m) {
        mindex = i;
        m = my_contents[i];
      }
    }
    my_bag_size--;
    //@ set empty = my_bag_size == 0;
    //@ assert empty == (my_bag_size == 0);
    my_contents[mindex] = my_contents[my_bag_size];
    return m;
  }
}
