package escjava;

import java.util.Set;
import java.util.LinkedHashSet;
import java.util.HashMap;
import java.util.Iterator;

/**
 * Checks consistency of option combinations according a set of logical rules.
 *
 * @author Mikolas Janota
 */
public class OptionsConsistency {
    /** set of option pairs that mutually exclude each other */
    private final /*@ non_null */ Option[][] excludes;

    /** set of option pairs that mutually exclude each other */
    private final /*@ non_null */ HashMap induces = new HashMap();;

    //@ private invariant setOptions.elementType == \type(Option) & setOptions.containsNull == false;
    /** set of options that have been set so far */ 
    private  /*@ non_null */LinkedHashSet setOptions;

    /** Describes an option conflict. */
    public class Conflict {
        private /*@ non_null*/Option opt1, opt2;
        /*@ pure */ private Conflict(/*@ non_null*/Option opt1, /*@ non_null*/Option opt2) {
            this.opt1 = opt1;
            this.opt2 = opt2;
        }     
        public /*@ non_null*/Option getOpt1() {
            return opt1;
        }   
        public /*@ non_null*/Option getOpt2() {
            return opt2;
        }   
    }


    /**
     * Initializes a new <code>OptionsConsistency</code> instance.
     *
     * @param excludes sets of options, where each set imposes mutual exclusion on its members
     */
    /*@ public normal_behavior
     @    requires \nonnullelements(excludes);
      @   requires (\forall int i; 0 <= i & i < excludes.length; excludes[i].length >= 2); */
    /*@ pure */ public OptionsConsistency(/*@ non_null */Option[][] excludes) {
        this.excludes = (Option[][]) excludes.clone();
        setOptions = new LinkedHashSet();
        //@ set setOptions.elementType = \type(Option);
        //@ set setOptions.containsNull = false;
    }

    /**
     * Add the induces relation for a given option. Whenever
     * <code>option</code> is set the options in
     * <code>inducedOptions</code> will be automatically set. 
     * @see #set(Option)
     */
    /*@ public normal_behavior
      @ requires \nonnullelements(inducedOptions); */
    /*@ also private normal_behavior
      @ assignable induces.objectState; */
    public void addInduces(/*@ non_null */Option option, /*@ non_null */Option[] inducedOptions) {
        induces.put(option, inducedOptions);
    }

    /**
     * Test whether a given option whould cause a conflict if set.
     * @see #set(Option)
     * @return a conflict description if there is a conflict, <code>null</code> otherwise.
     */
    public Conflict isConflict(/*@non_null*/Option option) {
        Set optionsOn = new LinkedHashSet(); /*@ set optionsOn.elementType = \type(Option); set optionsOn.containsNull = false; */
        // collect options induced by [option]
        collectInduced(option, optionsOn);

        // add the options currently set
        //optionsOn.addAll(this.setOptions);
        for (Iterator it = setOptions.iterator(); it.hasNext(); ) {
            optionsOn.add(it.next());
        }

        // add the tested option
        optionsOn.add(option);

        return excludedConflict(optionsOn);
    }


    /**
     * Set a given option. Induced options are automatically set as well.
     * @see #isOn(Option)
     * @see #addInduces(Option, Option[])
     */
    /*@ private normal_behavior
      @    assignable setOptions.objectState;
      @    ensures \not_modified(setOptions.elementType, setOptions.containsNull);
    */

    public void set(/*@ non_null */Option option) {
        setOptions.add(option);
        Set induced = collectInduced(option);
        for (Iterator it = induced.iterator(); it.hasNext();) {
            setOptions.add(it.next());
        }
    }

    /**
     * Determines whether a given option is set.
     * @see #set(Option)
     */
    /*@ pure */public boolean isOn(Option option) {
        return setOptions.contains(option);
    }


    /**
     * Tests whether the given set of options contains a pair of individually excluded ones.
     * @return a description of the conflict, <code>null</code> otherwise.
     */
    //@ requires optionsOn.elementType == \type(Option) && optionsOn.containsNull == false;
    /*@ pure */private Conflict excludedConflict(/*@ non_null */Set optionsOn) {
        Object[] os = optionsOn.toArray();
        for (int exlSetIndex = 0; exlSetIndex < excludes.length; exlSetIndex++) {
            Option[] exlSet = excludes[exlSetIndex];

        for (int i = 0; i < os.length; i++)
            for (int j = i+1; j < os.length; j++) {
                Option os_i = (Option) os[i];
                Option os_j = (Option) os[j];
                if (isMember(os_i, exlSet) && isMember(os_j, exlSet)) {
                    return new Conflict(os_i, os_j);
                }
            }
        }
        return null;
    }


    //@ ensures \fresh(\result);
    //@ ensures \result.elementType == \type(Option) & \result.containsNull == false;
    private /*@ non_null */Set collectInduced(/*@ non_null */Option option) {
        Set implied = new LinkedHashSet(); /*@ set implied.elementType = \type(Option); set implied.containsNull = false; */
        collectInduced(option, implied);
        return implied;
    }

    //@ requires optionsOn.elementType == \type(Option) & optionsOn.containsNull == false;
    //@ ensures \not_modified(optionsOn.elementType, optionsOn.containsNull);
    //@ assignable optionsOn.objectState;
    private void collectInduced(/*@ non_null */Option option, /*@ non_null */Set optionsOn) {
        Option[] implied = (Option[]) induces.get(option);

        if (implied != null) {
            for (int i = 0; i < implied.length; i++) {
                Option currentImplied = implied[i];
                optionsOn.add(currentImplied);
                collectInduced(currentImplied, optionsOn);
            }
        }
    }


    //@ requires \nonnullelements(options);
    /*@ pure */private boolean isMember(/*@ non_null */Option option, /*@ non_null */Option[] options) {
        return contains(options, option) >= 0;
    }


    //@ requires \nonnullelements(options);
    /*@ pure */private int contains(/*@ non_null */Option[] options, /*@ non_null */Option option) {
        for (int i = 0; i < options.length; i++)  {
            if (options[i].equals(option))
                return i;
        }
        return -1;
    }




}
