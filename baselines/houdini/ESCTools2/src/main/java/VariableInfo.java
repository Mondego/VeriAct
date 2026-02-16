package escjava.vcGeneration;

import java.util.Iterator;
import java.util.Set;
import java.util.HashMap;

import java.util.regex.Pattern;
import java.util.regex.Matcher;

public class VariableInfo {

    /**
     * README :
     *
     * This class is used to contain renaming of variables.
     * For each variable a VariableInfo object is created.
     *
     * If you want to change the way variables are renamed, just change 
     * {@link pvsRename} or {@link sammyRename} or add
     * a function at the end.
     */

    public/*@ non_null @*/String old = null;

    public TypeInfo type = null;

    public String def = null;

    private ProverType prover = null;

    public boolean typeSure = false;

    /**
     * This last field wasn't planned intially. 
     * But it is necessary when some variables have two types
     * like for a class like :
     *
     * class A {

     int i1;

     }
     *
     * Then the variable i1 must have as first type "%Field"
     * and "integer" two.
     */
    private TypeInfo secondType = null;

    /**
     * When we call this constructor, we know the old type, so we give it.
     */
    public VariableInfo(/*@ non_null @*/String old, TypeInfo type) {
        this.old = old;
        this.type = type;
    }

    /**
     * Constructor for specifying the renaming of the variable.
     */
    public VariableInfo(/*@ non_null @*/String old, TypeInfo type, /*@ non_null @*/String def, /*@ non_null @*/ProverType prover) {
        this.old = old;
        this.type = type;
        this.def = def;
        this.prover = prover;
    }

    public void setSecondType(/*@ non_null @*/TypeInfo type) {

        /*
         * do some control
         */
        if (secondType != null) {
            if (secondType != type)
                System.err
                        .println("Inconsistancy, changing type of a field named "
                                + old
                                + " from "
                                + secondType.old
                                + " to "
                                + type.old);
        } else
            secondType = type;

    }

    public TypeInfo getSecondType() {
        return secondType;
    }

    public/*@ non_null @*/String getVariableInfo() {
        return prover.getVariableInfo(this);
    }
    
    public boolean equals(Object o){
    	if(o instanceof VariableInfo) {
    		VariableInfo vi = (VariableInfo) o;
    		return old.equals(vi.old);
    	}
    	return false;
    }
}
