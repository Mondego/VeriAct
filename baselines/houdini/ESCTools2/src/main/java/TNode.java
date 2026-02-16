package escjava.vcGeneration;

import java.io.*;
import java.util.Iterator;
import java.util.HashMap;
import java.util.Set;

abstract public class TNode {

    /*@
     @ protected invariant (\forall TNode i,j ; i != j; i.id != j.id);
     @ protected invariant lastType >= 0 &&
     @   (typeProofSet ==> (lastType <= 1 && lastType <= 3)) &&
     @   (isroot ==> (this instanceof TRoot)) && // you probably want this to be <==> [Chalin]
     @   (parent != null || this instanceof TRoot); // ditto
     @*/

    // FIXME explain it in the doc
    static/*@ spec_public @*/protected PrintStream dotPs;

    protected int id;

    static protected int counter = 0;

    protected boolean isroot = false;

    public TypeInfo type = null;

    // internal representation of last type asked
    static/*@ spec_public @*/protected int lastType = 0;

    static/*@ spec_public @*/protected boolean typeProofSet = false;

    public TFunction parent = null;

    String label = null;

    /** map containing the variables names
     * When the tree is created, old name are entered as keys with 
     * a new {@link VariableInfo}(oldName, type) associated as value.
     * When the vc are generated, each time a variable is looked at, we look 
     * at this map. We look into the {@link VariableInfo} object associated.
     * If the renaming hasn't been done into this object, we do it.
     * If present, we use the already existing name.
     * Thus we will only rename one time and use previous value for next acccess.
     *
     * It handles multiple renaming because each {@link VariableInfo} object contains 
     * multiples fields for renaming (take a look at it).
     */
    static /*@ spec_public*/ protected /*@non_null*/ HashMap variablesName = new HashMap();

    /**
     * map containing all the types used in the proof.
     * 
     * After calling {@link typeTree}, each node has his {@link TypeInfo} field 
     * not null. It points on a TypeInfo object which is a value of this HashMap.
     * Each TypeInfo object contains the old type, the pvs corresponding type,
     * and the sammy corresponding type.
     * The old type can be {reference, integer, boolean, type}.
     * The tree is typechecked after call to {@link SetOutputType} and
     * this map is filled with only the type asked.
     * Ie if you call setOutputType("unsortedPvs"), the tree will be typed
     * with the old type and the types for the unsorted logic of pvs.
     * If you make another call to setOutputType, it will add type the tree
     * for the new tree etc...
     */
    static public/* spec_public non_null protected */HashMap typesName = new HashMap();

    /**
     * We add some types that we know will be used to avoid looking
     * at the map typesName when we want to add it.
     */
    static public TypeInfo _Reference = null;

    static public TypeInfo _Type = null;

    static public TypeInfo _DOUBLETYPE = null;

    static public TypeInfo _boolean = null;

    static public TypeInfo _char = null;

    static public TypeInfo _double = null;

    static public TypeInfo _Field = null;

    static public TypeInfo _INTTYPE = null;

    static public TypeInfo _integer = null;

    static public TypeInfo _float = null;

    static public TypeInfo _String = null;

    static public TypeInfo _Time = null;

    static public TypeInfo _Path = null;

    public TNode() {

        // FIXME :)
        //@ assert false;

        counter += 1;
        id = counter;
    }

    static public ProverType prover = null;
    
    // init every both variable and type map.

    static public void init(ProverType prover) {
        TNode.prover = prover;
        
        typesName = new HashMap();
        variablesName = new HashMap();

        prover.init();
    }

    protected void generateDeclarations(/*@ non_null @*/Writer s, ProverType p) {
        p.generateDeclarations(s, variablesName);
    }

    /**
     * This function add variable to the global map name.
     * @param oldName the old name.
     * @param type the type of the variable which has been infered from the old tree.
     *
     * @return the VariableInfo object representing this variables name 
     */
    /*@
     @ ensures variablesName.containsKey(oldName);
     @*/
    static public/*@ non_null @*/VariableInfo addName(/*@ non_null @*/String oldName, /*@ non_null @*/String type, String def) {

        if (!variablesName.containsKey(oldName)) {
            // we will do the renaming when necessary (look at {@link VariableInfo})

            TypeInfo ti = null;

            /*
             * We retrieve the type if necessary.
             */
            if (type != null)
                ti = TNode.addType(type);
            else
                TDisplay.warn("Adding variable named "
                        + oldName + " with no types");

            //@ assert typesName.containsKey(type);
            variablesName.put(oldName, new VariableInfo(oldName, ti, def, prover));
        }
        //++
        else {

            // debugging stuff that checks the new type is the same as the previous on
            if (type != null) {

                VariableInfo vi = (VariableInfo) variablesName.get(oldName);
                if (vi.type != typesName.get(type)) {

                    TDisplay
                            .warn("You're trying to add a variable named "
                                            + oldName
                                            + " whith type "
                                            + type.toString()
                                            + " which has been already stored with the type "
                                            + vi.type.toString()
                                            + " to the proof tree");
                }
            }

        }

        //@ assert variablesName.containsKey(oldName);

        /* we return the node which represents this variable.
         * Note that if this oldName wasn't present in the hashMap
         * the node returned is the one added created with TName node = new TName(oldName);
         */
        return (VariableInfo) variablesName.get(oldName);

        //++
    }

    static public/*@ non_null @*/VariableInfo addName(/*@ non_null @*/String oldName, /*@ non_null @*/String type) {
        return addName(oldName, type, null);
    }

    /**
     * return the {@link VariableInfo} object associated with this name
     */
    // xxx requires variablesName.contains(s);  //prj 15may2006 s not defined in this context
    static public/*@ non_null @*/VariableInfo getVariableInfo(/*@ non_null @*/String name) {
        return (VariableInfo) variablesName.get(name);
    }

    /**
     * return the {@link VariableInfo} object associated of the caller which
     * must be an instance of TName.
     */
    // xxx requires variablesName.contains(s); //prj 15may2006 s not defined in this context
    /*@ non_null @*/VariableInfo getVariableInfo() {

        //@ assert this instanceof TName;

        if (!(this instanceof TName)) {
            TDisplay.err("Warning calling getVariableInfo on a non TName Node");
            return null;
        } else {
            TName n = (TName) this;
            return getVariableInfo(n.name);
        }

    }

    /**
     * This function add type to the global map .
     * @param oldName the old name.
     *
     * @return the node pointing on that variable (can already exist if this add is
     * useless, in this case we return the previous TypeInfo representing this type).
     */
    /*@
     @ ensures typesName.containsKey(oldType);
     @*/
    static public/*@ non_null @*/TypeInfo addType(/*@ non_null @*/String oldType, String def) {

        if (!typesName.containsKey(oldType)) {
            // we will do the renaming when necessary (look at {@link TypeInfo})

            TypeInfo ti = new TypeInfo(oldType, def, prover);
            typesName.put(oldType, ti);

            //@ assert typesName.containsKey(oldType);

            return ti;
        } else {
            //@ assert typesName.containsKey(oldType);

            return (TypeInfo) typesName.get(oldType);
        }

    }

    static public/*@ non_null @*/TypeInfo addType(/*@ non_null @*/String oldType) {
        return TNode.addType(oldType, null);
    }

    /** dot id which is unique because of adding 
     'id' to the name of the node */
    protected/*@ non_null @*/String dotId() {

        /* escjava.vcGeneration.Tabc => r = abc */
        String r = getClass().getName().substring(22);
        r = r + id;

        return r;
    }

    // to avoid printing eachtime 'escjava.dsaParser.' and remove first 'T'
    protected/*@ pure non_null @*/String getShortName() {
        return getClass().getName().substring(22);
    }

    /*@
     @ requires typeProofSet;
     @ ensures type != null;
     @*/
    abstract protected void typeTree();

    /**
     * Add the type if not present to the global map of type
     * and fill the correct field with it.
     */
    protected void setType(String s, boolean sure) {
        type = TNode.addType(s);
    }


    protected void setType(TypeInfo type, boolean sure) {

        if (this instanceof TName) {
            TName m = (TName) this;

            //@ assert m.type == null;

            // retrieve current type;
            if (!variablesName.containsKey(m.name)) {
                TDisplay
                        .err("You're manipulating a TName ("
                                        + m.name
                                        + ") node, yet the name isn't in the global map 'variablesName'");
            }
            // take care no else here

            VariableInfo vi = (VariableInfo) variablesName.get(m.name);

            if (vi.type == null) {// we set it
                vi.type = type;
            } else {
                if (vi.type != type) {// inconsistency
                    if (vi.typeSure) // we don't change it
                        TDisplay.err("Variable named " + m.name + ", has type "
                                        + vi.type.old
                                        + " yet you try to change it to "
                                        + type.old);
                    else {
                        TDisplay.warn("Changing type of " + m.name + " (which was "
                                        + vi.type.old + ") to " + type.old);
                        vi.type = type;
                    }
                }

            }
        } else {
            if (this.type != null) {
                // we compare the existing type
                if (this.type != type) { // The two types are not equals
                    // inconsistancy
                    TDisplay.warn("Typechecking inconsistancy, " + this.toString()
                                    + "has type " + this.type.old
                                    + "but you're trying to set his type to "
                                    + type.old);
                }
            } else
                // type is null que pasa ?
                TDisplay.err( "Node " + this.toString() + " has no type");

        }
    }

    /**
     * return the type of the node according to the type of proof asked
     * or "?" if type isn't known.
     */
    /*@
     @ requires typeProofSet;
     @*/
    public/*@ non_null @*/String getType() {

        if (this instanceof TName) {
            // retrieve the type in the global variablesNames map
            TName m = (TName) this;

            //@ assert m.type == null;

            // retrieve current type;
            if (!variablesName.containsKey(m.name)) {
                TDisplay
                        .err("You're manipulating a TName ("
                                        + m.name
                                        + ") node, yet the name isn't in the global map 'variablesName'");
            }
            // take care no else here

            VariableInfo vi = (VariableInfo) variablesName.get(m.name);

            if (vi.type == null)
                return "?";
            else {
                TypeInfo ti = vi.type;

                String result = "";
                /*
                 * Handle double type here
                 */
                if (ti == _Field) { // variables may have a second type;
                    if (vi.getSecondType() != null) {

                        /*
                         * FIXME : explanations needed here, tricky stuff.
                         */

                        TypeInfo ti2 = vi.getSecondType();
                        VariableInfo vi2 = getVariableInfo(ti2.old);

                        result = vi2.getVariableInfo() + "\\n";

                    }

                }

                result = result + ti.getTypeInfo();

                return result;
            }

        } else {
            if (this.type == null)
                return "?";

            return this.type.getTypeInfo();
        }

    }

    /**
     * Return the typeInfo associated with this node. It can just be
     * the 'type' field for non instance of TName. Or in the case
     * of TName node, we retrieve it in the global map.
     */
    public TypeInfo getTypeInfo() {

        if (!(this instanceof TName))
            return this.type;
        else { //retrieve the type in the map

            TName m = (TName) this;

            //@ assert m.type == null;

            // retrieve current type;
            if (!variablesName.containsKey(m.name)) {
                TDisplay
                        .err("You're manipulating a TName ("
                                        + m.name
                                        + ") node, yet the name isn't in the global map 'variablesName'");
            }
            // take care no else here

            VariableInfo vi = (VariableInfo) variablesName.get(m.name);

            //++
            // 	    if(vi.type != null)
            // 		System.out.println("returning "+vi.type.old+" for "+this.toString());
            //++

            // can be null
            return vi.type;
        }

    }

    public /*@non_null*/ String toString() {
        if (type != null)
            return getShortName() + id + ", " + type.old;
        else
            return getShortName() + id;
    }

    abstract public void accept(/*@ non_null @*/TVisitor v);

    static public void printInfo() {

        // print all variables and their renaming in a 'nice' way
        Set keySet = variablesName.keySet();

        Iterator iter = keySet.iterator();
        String keyTemp = null;
        VariableInfo viTemp = null;

        System.out.println("\n=== List of variables ===");
        System.out
                .println("[oldName, newName : oldType]\n");

        while (iter.hasNext()) {

            try {
                keyTemp = (String) iter.next();
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            viTemp = (VariableInfo) variablesName.get(keyTemp);

            /* output informations in this format : oldName, pvsUnsortedName,
             * pvsName, sammyName, type.
             */
            if (viTemp.type != null) // FIXME
                System.out.println("[" + viTemp.old + ", " + viTemp.getVariableInfo() + " : "
                        + viTemp.type.old + "]");
            else
                System.out.println("[" + viTemp.old + ", " + viTemp.getVariableInfo()
                        + " : ?type? ]");
        }

        keySet = typesName.keySet();
        iter = keySet.iterator();
        TypeInfo tiTemp = null;

        System.out.println("\n=== List of type      ===");
        System.out.println("[old, new]\n");

        while (iter.hasNext()) {

            try {
                keyTemp = (String) iter.next();
            } catch (Exception e) {
                System.err.println(e.getMessage());
            }
            tiTemp = (TypeInfo) typesName.get(keyTemp);

            /* output informations in this format : old, pvsUnsorted,
             * pvs, sammy.
             */

            System.out.println("[" + tiTemp.old + ", " + tiTemp.getTypeInfo() + "]");
        }

        System.out.print("\n");
    }
}
