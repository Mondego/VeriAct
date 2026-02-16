package escjava.vcGeneration;

public class TypeInfo {

    /**
     * README :
     * 
     * This class is used to store old types and corresponding new types.
     * 
     * If you want to change the way types are renamed, just change
     * {@link pvsRename} or {@link sammyRename} or add a function at the end.
     */

    public/*@non_null*/String old = null;

    public String def = null;

    private ProverType prover = null;

    public TypeInfo(/*@non_null*/String old) {
        this.old = old;
    }

    /**
     * Constructor for specifying the renaming of the type.
     */
    public TypeInfo(/*@non_null*/String old, /*@non_null*/String def, /*@non_null@*/ProverType prover) {
        this.old = old;
        this.def = def;
        this.prover = prover;
    }

    public/*@non_null*/String getTypeInfo() {
        return prover.getTypeInfo(this);
    }

}
