public class IntervalRemove extends Interval {

    //@ requires l < h;
    //@ assignable low, high;
    //@ ensures low == l && high == h;
    public IntervalRemove(int l, int h) {
        super(l,h);
    }

    /** Remove i, which must be low or high. */
    /*@ requires low+1 < high;
      @ requires i == low || i == high;
      @ assignable low, high;
      @ ensures i == \old(low)
      @     ==> low == \old(low+1)
      @         && high == \old(high);
      @ ensures i == \old(high) 
      @     ==> high == \old(high-1) 
      @         && \not_modified(low);
      @*/
    public void remove(int i) {
        if (i == low) { low++;} else { high--; }
    }
}
