package escjava.vcGeneration;

import java.util.Vector;

import java.lang.ArrayIndexOutOfBoundsException;

abstract public class TFunction extends TNode {

    /*
     * This field is not protected because we can remove childs
     * in TProofSimplifier.java
     */
    public /*@ non_null @*/ Vector sons = new Vector();

    /*@
     @ ensures \old(sons.size()) == sons.size()-1;
     @*/
    void addSon(/*@ non_null @*/TNode n) {
        sons.add(n);

        //++
        if (n.parent != null)
            TDisplay.err("node already have a parent");
        //++
        else
            n.parent = this;
    }

    /*@
     @ ensures index >= 0 && index <= sons.size()-1;
     @*/
    public TNode getChildAt(int index) {
        try {
            return (TNode) sons.elementAt(index);
        } catch (ArrayIndexOutOfBoundsException e) {
            TDisplay.err("ArrayIndexOutOfBoundsException"
                    + e.getMessage());
            return null; /* notice this default behaviour */
        }
    }

    protected void typeTree() {

        /* recursive call on the sons */
        for (int i = 0; i <= sons.size() - 1; i++)
            getChildAt(i).typeTree();

    }

}
