//@ refine "Interval.jml";
/** Non-empty inclusive intervals of integers. */
public class Interval {
    /** Low and high bounds */
    protected /*@ spec_public @*/ int low, high;

    /** Initialize this Interval to be [l,h] */
    public Interval(int l, int h) {
        low = l; high = h;
    }

    /** Is i in this interval? */
    public /*@ pure @*/ boolean contains(int i) {
        return low <= i && i <= high;
    }
}
