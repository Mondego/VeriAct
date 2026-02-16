package decsrc.container;

/**
* Abstract set of integers.  Note that a particular set implementation
* may have restrictions on the range of integers that can be inserted
* into the set.
* <p>
* The contents of these sets can be iterated as follows:
*    <pre>
*    for (int iter = -1; (iter = set.next(iter)) >= 0; ) {
*	  process(set.get(iter));
*    }
*    </pre>
* <p>
* List of operations to add:
* <ul>
* <li> Exclusive-or
* <li> equals
* <li> is_strict_subset_of
* </ul>
*/
public abstract class IntSet {
    /** Return number of elements in this. */
    public abstract int count();

    /** Returns the smallest allowed element in the set. */
    public abstract int min_allowed();

    /** Returns the largest allowed element in the set. */
    public abstract int max_allowed();

    /** Returns true iff "x is_element_of this".
	Requires "min_allowed <= x <= max_allowed". */
    public abstract boolean contains(int x);

    /** post(this) = union(pre(this), {i}).
	Returns true iff "post(this) != pre(this)".
	Requires "min_allowed <= x <= max_allowed". */
    public abstract boolean insert(int x);

    /** post(this) = pre(this) - {i}.
	Returns true iff "post(this) != pre(this)".
	Requires "min_allowed <= x <= max_allowed". */
    public abstract boolean remove(int x);

    /** post(this) = {} */
    public abstract void clear();

    // Iteration

    /** Return the smallest index "x" such that "x > iter" and
	an element exists at index "x".  Return -1 if no such
        index exists.  Requires "iter" is either -1, or a
	value returned by a previous call to "next", and the
	set has not been modified since that call. */
    public abstract int next(int iter);

    /** Return the element at index "x".  Requires "x" was returned
	by a previous call to "next", the set has not been modified
	since that call, and "x >= 0". */
    public abstract int get(int x);

    // Vector operations.  Some of these have default implementations
    // that subclasses may override for efficiency.

    /** post(this) = pre(peer). */
    public void copy_from(/*@non_null*/IntSet intSetPeer) {
	if (this != intSetPeer) {
	    clear();
	    int iter = -1;
	    while ((iter = intSetPeer.next(iter)) >= 0) {
		insert(intSetPeer.get(iter));
	    }
	}
    }

    /** post(this) = intersect(pre(this), pre(peer)).
	Returns true iff post(this) != pre(this). */
    public abstract boolean intersect(/*@non_null*/IntSet intSetPeer);

    /** post(this) = union(pre(this), pre(peer)).
	Returns true iff post(this) != pre(this). */
    public boolean union(/*@non_null*/IntSet intSetPeer) {
	boolean changed = false;
	if (this != intSetPeer) {
	    int iter = -1;
	    while ((iter = intSetPeer.next(iter)) >= 0) {
		changed |= insert(intSetPeer.get(iter));
	    }
	}
	return changed;
    }

    /** post(this) = pre(this) - pre(peer).
	Returns true iff post(this) != pre(this). */
    public abstract boolean subtract(/*@non_null*/IntSet intSetPeer);

    /** Returns true iff "this" is a subset of "peer". */
    public boolean is_subset_of(/*@non_null*/IntSet intSetPeer) {
	if (this != intSetPeer) {
	    int iter = -1;
	    while ((iter = next(iter)) >= 0) {
		if (!intSetPeer.contains(get(iter))) {
		    return false;
		}
	    }
	}
	return true;
    }

    /** Return string representation. */
    public /*@non_null*/String toString() {
	CharVec vec = new CharVec();
	int iter = -1;
	String prefix = "";
	while ((iter = next(iter)) >= 0) {
	    int x = get(iter);
	    vec.appendString(prefix);
	    vec.appendString(Integer.toString(x));
	    prefix = ", ";
	}
	return vec.toString();
    }
}
