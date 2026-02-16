package escjava;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.Comparator;


/**
 * Represents a command line option. Constructor and new options obtained by calling <code>registerOption</code>, 
 * which also stores the new option in a static collection.
 *
 * @author Mikolas Janota
 */
public class Option {
    //@ private invariant \nonnullelements(names);
    private final /*@ non_null*/String[] names;

    //@ private static invariant options.elementType == \type(Option) && !options.containsNull;
    private static /*@ non_null*/ ArrayList options = new ArrayList();

    private static /*@ non_null*/ Comparator stringComparator = String.CASE_INSENSITIVE_ORDER;

    static {
       //@ set options.elementType = \type(Option);
    }



    /*@ public normal_behavior
      @ requires \nonnullelements(names);
      @   ensures \fresh(\result); */
    /*@ also private behavior
      @   requires \nonnullelements(names);
      @   assignable options.objectState; 
      @   ensures options.contains(\result); 
      @   ensures \fresh(\result); */
    public static /*@ non_null*/Option registerOption(/*@ non_null*/String[] names) {
        Option newOption = new Option(names);
        options.add(newOption);
        return newOption;
    }


    /**
     * Searches for a registed option given one of its names.
     *
     * @param name the name of the option searched for
     * @return the option if foudn, <code>null</code> otherwise
     */
    //@ ensures \result != null ==> \result.isMe(name);
    public static Option findOption(/*@ non_null*/String name) {
        for (Iterator it = options.iterator(); it.hasNext();) {
            Option o = (Option)it.next();
            if (o.isMe(name)) {
                    return o;
            }

        }
        return null;
    }


    // Tests whether the given option has been registered.
    /*@ pure */ public static boolean isRegistered(/*@ non_null*/Option o) {
        return options.contains(o);
    }

    /*@ pure */public /*@ non_null*/String toString() {
        String retv;
        if (names.length > 0) {
            return names[0];
        } else {
            return "<unspecified name>";
        }
    }

    /*@ pure */ public boolean equals(Object o) {
        return this == o;
    }

    /** Typed equivalent of the <code>equals</code> method. */
    /*@ pure */public boolean isSame(Option o) {
        return this == o;
    }



    //@ requires \nonnullelements(names);
    private /*@pure*/ Option(/*@ non_null */String[] names) {
        this.names = (String[]) names.clone();
    }

    /*@ pure */public boolean isMe(/*@ non_null */String name) {
        //@ loop_invariant (\forall int x; 0 <= x & x < i; !(stringComparator.compare(names[x], name) == 0));
        for (int i = 0; i < names.length; i++) {
            if (stringComparator.compare(names[i], name) == 0) {
                return true;
            }
        }

        return false;
    }

}