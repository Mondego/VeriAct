public class MethodAnnotations2 {

    //@ spec_public
    /*@ non_null */ final private int[] array = new int[20];

    //@ spec_public
    private int size;

    //@ private normal_behavior
    //@ ensures \result == array.length;
    //@ model pure public int capacity();

    //@ private normal_behavior
    //@ ensures \result == size;
    //@ pure
    public int size() { return size; }

    //@ public invariant size() >= 0 && size() <= capacity();

    //@ requires size() < capacity();
    //@ assignable size,array[*];
    //@ ensures size() > \old(size());
    //@ ensures size() == \old(size() + 1);
    //@ signals (Exception) false;
    public void push(int i) {
        array[size++] = i;
        //@ assert size() == \old(size())+1;
    }

    //@ requires size() > 0;
    //@ assignable size,array[*];
    //@ ensures size() < \old(size());
    //@ ensures size() == \old(size() - 1);
    //@ signals (Exception) false;
    public int pop() {
        return array[--size];
    }

}

class Client {

    //@ requires q != null;
    //@ requires q.size() < q.capacity();
    //@ requires q.size() > 0;
    //@ modifies q.size, q.array[*];
    //@ ensures q.size() == \old(q.size()) + 1;
    static void m(MethodAnnotations2 q) {
	q.push(0);
	q.pop();
	//@ assert q.size() == \old(q.size());
	q.pop();
	q.push(0);
	//@ assert q.size() == \old(q.size());
	q.push(0);
    }
}
