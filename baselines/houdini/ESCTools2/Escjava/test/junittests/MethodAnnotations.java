public class MethodAnnotations {

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
    //@ signals (Exception) false;
    public void push(int i) {
        array[size++] = i;
        //@ assert size() == \old(size())+1;
    }

    //@ requires size() > 0;
    //@ assignable size,array[*];
    //@ ensures size() < \old(size());
    //@ signals (Exception) false;
    public int pop() {
        return array[--size];
    }

}
