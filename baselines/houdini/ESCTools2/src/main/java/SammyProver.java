package escjava.vcGeneration.sammy;

import java.util.HashMap;
import java.io.*;

import javafe.ast.Expr;

import escjava.translate.GC;
import escjava.translate.InitialState;
import escjava.vcGeneration.*;

public class SammyProver extends ProverType {
    
    public String labelRename(String label) {
        return label;
    }
    
    public TVisitor visitor(Writer out) {
        return null; //FIXME
    }
    
    public void getProof(Writer out, String proofName, TNode term) {
        //FIXME
    }
    
    public/*@ non_null @*/String getVariableInfo(VariableInfo caller) {

        if (caller.type != TNode._Type) {
            if (caller.def == null)
                sammyRename(caller);

            return caller.def;
        } else
            return getTypeInfo(caller.type);
    }

    // fixme, does nothing atm
    /*! ensures sammy != null; !*/
    private void sammyRename(VariableInfo caller) {
        caller.def = caller.old;
    }

    public String getTypeInfo(TypeInfo caller) {
        if (caller.def == null)
            sammyRename(caller);

        return caller.def;
    }

    // fixme, does nothing atm
    /*! ensures sammy != null; !*/
    private void sammyRename(TypeInfo caller) {
        caller.def = caller.old;
    }

    public void init() {
        // Predefined types

        TNode._Reference = TNode.addType("%Reference", "?");
        TNode._Time = TNode.addType("%Time", "?");
        TNode._Type = TNode.addType("%Type", "?");
        TNode._boolean = TNode.addType("boolean", "?");
        TNode._char = TNode.addType("char", "?");
        TNode._DOUBLETYPE = TNode.addType("DOUBLETYPE", "?"); // fixme, is it JavaNumber or BaseType ?
        TNode._double = TNode.addType("double", "?"); //fixme
        TNode._Field = TNode.addType("%Field", "?"); // fixme there's a lot of different fields in the pvs logic, I need to capture that
        TNode._INTTYPE = TNode.addType("INTTYPE", "?"); //fixme like DOUBLETYPE
        TNode._integer = TNode.addType("integer", "?"); //fixme
        TNode._float = TNode.addType("float", "?");
        TNode._Path = TNode.addType("%Path", "?"); // used to modelize different ways
        // of terminating a function
        //_String = addType("String", "?"); fixme, does this type appears in original proof ?

        // Predefined variables name
        // variables used by the old proof system and that we still need
        TNode.addName("ecReturn", "%Path", "ecReturn");
        TNode.addName("ecThrow", "%Path", "ecThrow");
        TNode.addName("XRES", "%Reference", "XRES");        
    }
    
    public Expr addTypeInfo(InitialState initState, Expr tree) {
        tree = GC.implies(initState.getInitialState(), tree);
        return tree;
    }

    public TNode rewrite(TNode tree) {
        // Intentionally do nothing!
        return tree;
    }

    public void generateDeclarations(/*@ non_null @*/ Writer s, HashMap variablesName) {
	// TODO Auto-generated method stub
    }
}
