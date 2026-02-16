public interface NoRep {
    //@ public model String outputText;
    //@ invariant outputText != null;

    /*@ assignable outputText;
        ensures outputText.equals(
                    \old(outputText) + s); */
    public void print(String s);
}
