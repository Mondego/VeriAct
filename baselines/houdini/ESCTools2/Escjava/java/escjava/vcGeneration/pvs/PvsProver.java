package escjava.vcGeneration.pvs;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.io.*;

import javafe.ast.Expr;

import escjava.translate.InitialState;
import escjava.vcGeneration.*;

public class PvsProver extends ProverType {

    public String labelRename(String label) {
        label = label.replace('.','_');
        return label;
    }
    
    public TVisitor visitor(Writer out) {
        return new TPvsVisitor(out);
    }

    public void getProof(Writer out, String proofName, TNode term) {
        try {
            out.write(proofName + " : CONJECTURE\n"); // let's be modest...
            out.write("ForAll(\n");
            generateDeclarations(out, term);
            out.write(") :\n\n");
            generateTerm(out, term);
        } catch (IOException e) {
            // Intentionally do nothing
        }
    }

    public String getVariableInfo(VariableInfo caller) {

        if (caller.type != TNode._Type) {
            if (caller.def == null)
                pvsRename(caller);

            return caller.def;
        } else {
            /*

             When variable is a type, we first have to check if it's in the type table.
             If yes, we take the name here (it's the case with predefined types like INTYPE, integer, Reference etc...

             Else it's normally a user defined Type

             */

            Set keySet = TNode.typesName.keySet();
            Iterator iter = keySet.iterator();
            TypeInfo tiTemp = null;
            String keyTemp = null;

            while (iter.hasNext()) {

                try {
                    keyTemp = (String) iter.next();
                } catch (Exception e) {
                    System.err.println(e.getMessage());
                }
                tiTemp = (TypeInfo) TNode.typesName.get(keyTemp);

                if (tiTemp.old.equals(caller.old)) {
                    return getTypeInfo(tiTemp);
                }

            }

            TDisplay
            .warn("Considering "
            		+ caller.old
            		+ " as a user defined type, or a not (yet) handled variable.");

            pvsRename(caller);

            return caller.def;
        }

    }

    /* @
       @ ensures pvs != null;
       @ */
    private void pvsRename(VariableInfo caller) {

        // definitions of different regexp used.
        Pattern p1 = Pattern.compile("\\|(.*):(.*)\\.(.*)\\|");
        Matcher m1 = p1.matcher(caller.old);

        // <variables not handled>
        Pattern p2 = Pattern.compile("Unknown tag <.*>");
        Matcher m2 = p2.matcher(caller.old);

        Pattern p3 = Pattern.compile("\\|brokenObj<.*>\\|");
        Matcher m3 = p3.matcher(caller.old);
        // </variables not handled>

        String name = null;
        String line = null;
        String column = null;

        int i = 0;

        //System.out.println("Performing regular expression matching against "+old+" ");

        /*
         This case is done first because some variables not handled are "%Type" ones
         and thus otherwise they will be matched by the next if construct
         */
        if (m2.matches() || m3.matches() || caller.old.equals("brokenObj")) { // variable is not handled yet, ask David Cok about
            // some things
            caller.def = "%NotHandled";
        }

        // <case 1>
        /*
         If variable's a type, we must prefix his renaming by userDef?.
         Why ?
         Because if the user defined a type named 'Reference', otherwise we will use is as it
         in the proof and it will interfer with our predefined types.

         Since '?' isn't a valid character in the java notation and is Ok for pvs,
         we use it to make the difference.
         */
        else if (caller.type == TNode._Type) {
            TDisplay.warn("Considering " + caller.old
                    + " as a user defined type.");

            // renaming done here
            caller.def = "userDef?" + caller.old;
        }
        // </case 1>

        // <case 2>, capturing |y:8.31|
        else if (m1.matches()) {
            //@ assert m1.groupCount() == 3;

            if (m1.groupCount() != 3)
                TDisplay.err("m.groupCount() != 3");

            for (i = 1; i <= m1.groupCount(); i++) {

                if (m1.start(i) == -1 || m1.end(i) == -1)
                    TDisplay.err("Return value of regex matching is -1");
                else {

                    String temp = caller.old.substring(m1.start(i), m1.end(i));
                    //@ assert temp != null;

                    switch (i) {
                    case 1:
                        name = temp;
                        //@ assert name != null;
                        break;
                    case 2:
                        line = temp;
                        //@ assert line != null;
                        break;
                    case 3:
                        column = temp;
                        //@ assert column != null;
                        break;
                    default:
                        TDisplay.err("Switch call incorrect, switch on value " + i);
                        break;
                    }
                } // no error in group

            } // checking all group

            /* renaming done here, if you want the way it's done
             (for pattern like |y:8.31|)
             do it here. */
            caller.def = name + "_" + line + "_" + column;

        } // </case 2>
        else {
            caller.def = caller.old; // FIXME handle everything
        } // regexp didn't match
    }

    public String getTypeInfo(TypeInfo caller) {
        if (caller.def == null)
            pvsRename(caller);
        return caller.def;
    }

    /* @
       @ ensures pvs != null;
       @ */
    private void pvsRename(TypeInfo caller) {

        if (caller.old.equals("null"))
            caller.def = "Reference";
        else {
            // common rules here //fixme, be more specific maybe
            if (caller.old.startsWith("java.")) //check if in the form java.x.y 
                caller.def = caller.old.replace('.', '_');
            else {
                TDisplay.warn("Type not handled  : "
                        + caller.old);
                TDisplay
                        .warn("Considering it as a user defined type... ie ReferenceType");
                caller.def = "ReferenceType";
            }
        }
    }

    public void init() {
        // Predefined types

        TNode._Reference = TNode.addType("%Reference", "Reference");
        TNode._Time = TNode.addType("%Time", "Time");
        TNode._Type = TNode.addType("%Type", "ReferenceType");
        TNode._boolean = TNode.addType("boolean", "Boolean");
        TNode._char = TNode.addType("char", "T_char");
        TNode._DOUBLETYPE = TNode.addType("DOUBLETYPE", "ContinuousNumber"); // fixme, is it JavaNumber or BaseType ?
        TNode._double = TNode.addType("double", "ContinuousNumber"); //fixme
        TNode._Field = TNode.addType("%Field", "Field"); // fixme there's a lot of different fields in the pvs logic, I need to capture that
        TNode._INTTYPE = TNode.addType("INTTYPE", "T_int"); //fixme like DOUBLETYPE
        TNode._integer = TNode.addType("integer", "DiscreteNumber"); //fixme
        TNode._float = TNode.addType("float", "ContinuousNumber");
        TNode._Path = TNode.addType("%Path", "Path"); // used to modelize different ways
        // of terminating a function
        //_String = addType("String" "String"); fixme, does this type appears in original proof ?

        // Predefined variables name
        // variables used by the old proof system and that we still need
        TNode.addName("ecReturn", "%Path", "preDef?ecReturn");
        TNode.addName("ecThrow", "%Path", "preDef?ecThrow");
        TNode.addName("XRES", "%Reference", "preDef?XRes");
    }
    
    public Expr addTypeInfo(InitialState initState, Expr tree) {
        return tree;
    }

    /** We expect this method to be called with a tree of type TBoolImplies:
     * <ul>
     * <li>the hypothesis of the tree is expected to be typing information</li>
     * <li>the conclusion of the tree is expected to be the VC body</li>
     * </ul>
     * Since the Pvs prover is a strongly typed logic, we may eliminate this typing 
     * information <i>before</i> providing any additional simplifications.
     * 
     * Strictly speaking, we should check that the hypothesis is as expected (FIXME).
     */   
    public TNode rewrite(TNode tree) {
        TProofSimplifier psvi = new TProofSimplifier();
        tree.accept(psvi);
        return tree;
    }

	public void generateDeclarations(/*@ non_null */ Writer s, HashMap variablesName) {
        Set keySet = variablesName.keySet();

        Iterator iter = keySet.iterator();
        String keyTemp = null;
        VariableInfo viTemp = null;

        /*
         * Needed to avoid adding a comma after last declaration. As some declaration can be skipped
         * it's easier to put comma before adding variable (thus need for testing firstDeclaration
         * instead of last one)
         */
        boolean firstDeclarationDone = false;

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
            if (viTemp.type != null) {
                try {
                    if (firstDeclarationDone
                            && !viTemp.getVariableInfo().equals("%NotHandled"))
                        s.write(",\n");

                    if (!viTemp.getVariableInfo().equals("%NotHandled")) { // skipping non handled variables
                        s.write(viTemp.getVariableInfo() + " : "
                                + viTemp.type.getTypeInfo());

                        if (!firstDeclarationDone)
                            firstDeclarationDone = true;
                    }
                } catch (IOException e) {
                    // Intentionally do nothing
                }
            } else
                // FIXME test that it nevers happen
            	TDisplay
            	.warn("Type of variable "
            			+ keyTemp
            			+ " is not set when declarating variables for the proof, skipping it...");
        }
    }
}
