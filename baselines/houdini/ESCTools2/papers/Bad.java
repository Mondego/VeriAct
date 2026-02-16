public class Bad {
    /*@ ensures \result == o.i; */
    /*@ pure */ 
    static public int valueOfI(Example o);

    //@ ensures valueOfI(o) > 0;
    public void init(Example o);
}
