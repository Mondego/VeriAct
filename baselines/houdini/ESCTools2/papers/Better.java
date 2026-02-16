public class Better {
    /*@ ensures (o != null && o.i > 0) ==> 
                     \result == o.i;
        diverges false;
        signals (Exception e) false;
     */
    //@ pure
    static public int valueOfI(Example o);

    //@ ensures valueOfI(o) > 0;
    public void init(Example o);
}
