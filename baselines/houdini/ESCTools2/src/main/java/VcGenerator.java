package escjava.vcGeneration;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;
import escjava.ast.*;
import escjava.translate.*;
import escjava.ast.TagConstants;
import escjava.prover.Atom;

import java.io.*;

public class VcGenerator {

    /**
     * README :
     *
     * This class is an interface to the vc generation suite
     * (this is done this way to be able to put all the other classes in a new package.)
     *
     * Usage :
     * You have to supply the root node of the vc tree you want to translate.
     * Then you can call any of the public finction to get what you want.
     * It has been designed in a way that makes sure we won't do the work 
     * two times. I.e., if you call the constructor and never call old2Dot(),
     * we never compute the dot file.
     * Most of the time (ie except for development and debugging), you have
     * to do : 
     * 1/ vcg = new VcGenerator(p,e,..); // p is the prover type and e is the root node of the vc tree
     * 2/ vcg.getProof(vc); // return, as a string, proof script for the given vc
     */

    private/*@ spec_public @*/ProverType prover = null;

    private/*@ spec_public non_null @*/ASTNode oldRootNode = null;

    private/*@ spec_public non_null @*/TNode newRootNode = null;

    private/*@ spec_public @*/StringBuffer oldDot = null;

    private/*@ spec_public @*/boolean computationDone = false;

    /*@
     @ public invariant computationDone ==> newRootNode != null &&
     @ (newRootNode != null ==> (newRootNode instanceof TRoot));
     @*/

    /**
     * @param e the root node of the gc tree.
     *
     * @return The VcGenerator on which you can call method to get
     * vcs for different syntax
     */
    public VcGenerator(/*@ non_null @*/ProverType p, /*@ non_null @*/ASTNode e, boolean err, boolean warn, boolean info, boolean colors) {
        prover = p;
        oldRootNode = e;

        /*
         * Set different output (or not);
         */
        TDisplay.init(err, warn, info, colors);

        /*
         * Reset static initialization for TNode.
         */
        TNode.init(prover);
    }

    public void getProof(Writer out, String proofName) {

        if (!computationDone) {
            generateIfpTree(oldRootNode, false);
            newRootNode.typeTree();
        }
        
        newRootNode = prover.rewrite(newRootNode);

        prover.getProof(out, proofName, newRootNode);
    }

    /*@
     @ ensures computationDone && oldDot != null;
     @*/
    public String old2Dot() {
        if (oldDot == null) {

            /* call to the fonction that does the job,
             indicating with the second parameter that we want the
             dot representation as well */
            generateIfpTree(oldRootNode, true);

            computationDone = true;
        }

        return oldDot.toString();
    }

    /*@
     @ ensures computationDone;
     @*/
    public String toDot() {
        if (!computationDone)
            generateIfpTree(oldRootNode, false);

        TDotVisitor v = new TDotVisitor();

        newRootNode.accept(v);

        return v.out.toString();
    }

    /**
     * This attribute is used by the next function to save the current parent of the node
     * we may create. Before any call to generateIfpTree, and after the last call has returned
     * this field is always null. This is bizarre, but avoids to add a parameter to this function
     * and makes the code more concise.
     */
    private TFunction currentParent = null;

    // boolean to skip first not
    // Clement's experiment
    private boolean firstNotSkipped = true;

    /**
     * The main goal of this method is to translate the 
     * gc tree (which is still independant from simplify) to a new tree
     * (classes of this new tree are in escjava/vcGeneration which is, by far, 
     * easier to manipulate that the one which is given here (parameter e). 
     *
     * The code is divided in this way :
     *
     * 1/ declarations
     * 2/ switch on the type of the node currently looked at
     * 3/ a/ depending on the type, create a new node for the translation of the tree
     *    b/ if dot output is asked (ie parameter dot != null), do the job
     *
     * 4/ call the method on the sons of the current node.
     *
     * @param p the node (of the old tree) you're visiting.
     * @param dot if true, generate the dot representation of the old tree.
     */
    private void generateIfpTree(/*@ non_null @*/ASTNode n, boolean dot) {

        // declarations & instancations
        int nbChilds = n.childCount();
        Object o = null;

        ASTNode nodeTemp = null;
        boolean outputDot = false;
        StringBuffer name = null;

        if (oldDot == null)
            oldDot = new StringBuffer();

        TFunction saveParent = currentParent;

        /* happen at first call */
        if (currentParent == null)
            currentParent = new TRoot();
        // /declarations & instancations

        if (dot) {
            name = new StringBuffer(getNameASTNode(n));
            name.append(n.dotId);

            /* declaration of the node in dot */
            oldDot.append(name);
            /* label = "xxxx" <- notice the " */
            oldDot.append(" [label = \"" + getNameASTNode(n));
        }

        // 	if(e.getStartLoc() != Location.NULL)
        // 	    dot.append(" "+e.getStartLoc()+"|"+e.getEndLoc());

        // all types checked are in alphabetical order
        if (n instanceof ArrayType) {

            ArrayType m = (ArrayType) n;
            // this represents a type

            String s = m.toString();

            TDisplay.err(s);

            // 		TName newNode = new TName(s);
            // 		// we put it as a variable name with type %Type
            // 		TNode.addName(s, "%Type");

            // 		// we put as a type too
            // 		TNode.addType(s);

            // 		currentParent.addSon(newNode);

            if (dot)
                // fixme, not precise enough maybe
                oldDot.append("\\n" + s + "\"");
        } else if (n instanceof LiteralExpr) {
            LiteralExpr m = (LiteralExpr) n;

            switch (m.getTag()) {
            case TagConstants.STRINGLIT: {
                TString newNode = new TString(null);
                currentParent.addSon(newNode);
                break;
            }
            case TagConstants.BOOLEANLIT: {
                TBoolean newNode = null;

                if (((Boolean) m.value).booleanValue())
                    newNode = new TBoolean(true);
                else
                    newNode = new TBoolean(false);

                currentParent.addSon(newNode);
                break;
            }
            case TagConstants.CHARLIT: {
                TChar newNode = new TChar(((Integer) m.value).intValue());
                currentParent.addSon(newNode);
                break;
            }
            case TagConstants.INTLIT: {
                TInt newNode = new TInt(((Integer) m.value).intValue());
                currentParent.addSon(newNode);
                break;
            }
            case TagConstants.FLOATLIT: {
                TFloat newNode = new TFloat(((Float) m.value).floatValue());
                currentParent.addSon(newNode);
                break;
            }
            case TagConstants.DOUBLELIT: {
                TDouble newNode = new TDouble(((Double) m.value).doubleValue());
                currentParent.addSon(newNode);
                break;
            }
            case TagConstants.NULLLIT: {
                TNull newNode = new TNull();
                currentParent.addSon(newNode);
                break;
            }
            case TagConstants.SYMBOLLIT: {
                //error.println("SYMBOLLIT "+(String)m.value );

                // pass here for ecReturn and ecThrow
                String s = (String) m.value;

                // fixme am I right ?
                TName newNode = new TName(s);
                TNode.addName(s, null);
                currentParent.addSon(newNode);
                break;
            }
            default:
                TDisplay.err("Instanceof LiteralExpr, case missed :"
                                + TagConstants.toString(m.getTag()));
                break;
            }

            if (dot)
                oldDot.append("\\n" + TagConstants.toString(m.getTag()) + "\"");
        }
        // name of a method
        else if (n instanceof NaryExpr) {
            NaryExpr m = (NaryExpr) n;

            String methodName = TagConstants.toString(m.getTag());

            switch (m.getTag()) {
            // boolean operations
            case TagConstants.BOOLIMPLIES: {
                TBoolImplies newNode = new TBoolImplies();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.BOOLAND:
            case TagConstants.BOOLANDX: {
                TBoolAnd newNode = new TBoolAnd();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.BOOLOR: {
                TBoolOr newNode = new TBoolOr();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.BOOLNOT: {
            	//FIXME: I am sorry Clement but I can't prove anything with
            	// your experiment...
                if (firstNotSkipped) {
                    TBoolNot newNode = new TBoolNot();
                    currentParent.addSon(newNode);
                    currentParent = newNode;
                } else
                    firstNotSkipped = true;

                break;
            }
            case TagConstants.BOOLEQ: {
                TBoolEQ newNode = new TBoolEQ();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } 
            case TagConstants.BOOLNE: {
                TBoolNE newNode = new TBoolNE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } 
            case TagConstants.METHODCALL: {
    		    // FIXME I think it is fixed... needs further testing
    		    //TDisplay.err(this, "generateIfpTree", TagConstants.toString(m.getTag()));
    		    TMethodCall newNode = new TMethodCall(m.methodName.toString());
    		    currentParent.addSon(newNode);
    		    currentParent = newNode;
    		    break;
    		}
            // allocation comparisons
            case TagConstants.ALLOCLT: {
                TAllocLT newNode = new TAllocLT();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ALLOCLE: {
                TAllocLE newNode = new TAllocLE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } 
            
            // references
            case TagConstants.ANYEQ: {
                TAnyEQ newNode = new TAnyEQ();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ANYNE: {
                TAnyNE newNode = new TAnyNE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } 
            
            // integral comparisons
            case TagConstants.INTEGRALEQ: {
                TIntegralEQ newNode = new TIntegralEQ();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALGE: {
                TIntegralGE newNode = new TIntegralGE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALGT: {
                TIntegralGT newNode = new TIntegralGT();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALLE: {
                TIntegralLE newNode = new TIntegralLE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALLT: {
                TIntegralLT newNode = new TIntegralLT();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALNE: {
                TIntegralNE newNode = new TIntegralNE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } // integral operation // fixme some missing
            case TagConstants.INTEGRALADD: {
                TIntegralAdd newNode = new TIntegralAdd();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALDIV: {
                TIntegralDiv newNode = new TIntegralDiv();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALMOD: {
                TIntegralMod newNode = new TIntegralMod();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALMUL: {
                TIntegralMul newNode = new TIntegralMul();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.INTEGRALSUB: {
    		    TIntegralSub newNode = new TIntegralSub();
    		    currentParent.addSon(newNode);
    		    currentParent = newNode;
    		    break;
    		}
            // real comparisons
            case TagConstants.FLOATINGEQ: {
                TFloatEQ newNode = new TFloatEQ();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGGE: {
                TFloatGE newNode = new TFloatGE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGGT: {
                TFloatGT newNode = new TFloatGT();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGLE: {
                TFloatLE newNode = new TFloatLE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGLT: {
                TFloatLT newNode = new TFloatLT();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGNE: {
                TFloatNE newNode = new TFloatNE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            // integral operation // fixme some missing
            case TagConstants.FLOATINGADD: {
                TFloatAdd newNode = new TFloatAdd();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGDIV: {
                TFloatDiv newNode = new TFloatDiv();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGMOD: {
                TFloatMod newNode = new TFloatMod();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FLOATINGMUL: {
                TFloatMul newNode = new TFloatMul();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            // lock comparisons
            case TagConstants.LOCKLE: {
                TLockLE newNode = new TLockLE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.LOCKLT: {
                TLockLT newNode = new TLockLT();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } // reference comparisons
            case TagConstants.REFEQ: {
                TRefEQ newNode = new TRefEQ();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.REFNE: {
                TRefNE newNode = new TRefNE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } // type comparison
            case TagConstants.TYPEEQ: {
                TTypeEQ newNode = new TTypeEQ();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.TYPENE: {
                TTypeNE newNode = new TTypeNE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.TYPELE: {
                TTypeLE newNode = new TTypeLE();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } // usual functions, cast is select store typeof 
            case TagConstants.CAST: {
                TCast newNode = new TCast();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.IS: {
                TIs newNode = new TIs();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.SELECT: {
                TSelect newNode = new TSelect();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.STORE: {
                TStore newNode = new TStore();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.UNSET: {
    			TUnset newNode = new TUnset();
    		    currentParent.addSon(newNode);
    		    currentParent = newNode;
    			break;
    		}
            case TagConstants.TYPEOF: {
                TTypeOf newNode = new TTypeOf();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            // allocation
            case TagConstants.ISALLOCATED: {
                TIsAllocated newNode = new TIsAllocated();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ECLOSEDTIME: {
                TEClosedTime newNode = new TEClosedTime();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.FCLOSEDTIME: {
                TFClosedTime newNode = new TFClosedTime();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } // as trick : asElems asField asLockset
            case TagConstants.ASELEMS: {
                TAsElems newNode = new TAsElems();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.SUM: {
            	TSum newNode = new TSum();
            	currentParent.addSon(newNode);
                currentParent = newNode;
            }
            case TagConstants.ASFIELD: {
                TAsField newNode = new TAsField();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ASLOCKSET: {
                TAsLockSet newNode = new TAsLockSet();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            } // array
            case TagConstants.ARRAYFRESH: {
                TArrayFresh newNode = new TArrayFresh();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ARRAYSHAPEONE: {
                TArrayShapeOne newNode = new TArrayShapeOne();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ARRAYSHAPEMORE: {
                TArrayShapeMore newNode = new TArrayShapeMore();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ISNEWARRAY: {
                TIsNewArray newNode = new TIsNewArray();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            case TagConstants.ARRAYLENGTH: {
                TArrayLength newNode = new TArrayLength();
                currentParent.addSon(newNode);
                currentParent = newNode;
                break;
            }
            default:
                TDisplay.err("translating old gc tree, methodName not recognized "
                                + methodName);
            }

            if (dot)
                /* fixme maybe this call can return "?" which means
                 * the name of the function isn't in the original tree
                 */
                oldDot.append("\\n" + TagConstants.toString(m.getTag()) + "\"");

        } else if (n instanceof PrimitiveType) { // javafe/Type
            PrimitiveType m = (PrimitiveType) n;
            String s = javafe.ast.ASTTagConstants.toString(m.getTag());

            // this means this variable represent a type like
            // Java.x.Vector or Java.lang.Object etc...
            TName newNode = new TName(s);
            TNode.addName(s, "%Type");

            currentParent.addSon(newNode);

            if (dot)
                oldDot.append("\\n" + s + "\"");

        } else if (n instanceof QuantifiedExpr) {
            QuantifiedExpr m = (QuantifiedExpr) n;
            String s = TagConstants.toString(m.getTag());

            if (s.equals("\\forall")) {
                TForAll newNode = new TForAll();
                currentParent.addSon(newNode);
                currentParent = newNode;
            } else if (s.equals("\\exist")) {
                TExist newNode = new TExist();
                currentParent.addSon(newNode);
                currentParent = newNode;
            } else
                TDisplay.err("QuantifiedExpr, unhandled tag : " + s);

            if (dot)
                oldDot.append("\\n" + s + "\"");

        } else if (n instanceof SimpleName) {
            SimpleName m = (SimpleName) n;

            // it seems that this node is under a TypeName node all the time
            // and that all the information is in the TypeName node.
            // that's why we don't create a new node here.

            if (dot)
                oldDot.append("\\n" + m.printName() + "\"");
        } else if (n instanceof SubstExpr) {
            SubstExpr m = (SubstExpr) n;

            TDisplay.err("SubstExpr viewed and not handled");
        } else if (n instanceof TypeDecl) {

            TypeDecl m = (TypeDecl) n;
            // this represents a type

            String s = new String(m.id.chars);

            TName newNode = new TName(s);
            // we put it as a variable name with type %Type
            TNode.addName(s, "%Type");

            // we put as a type too
            TNode.addType(s);

            currentParent.addSon(newNode);

            if (dot)
                // fixme, not precise enough maybe
                oldDot.append("\\n" + s + "\"");
        } else if (n instanceof TypeExpr) {
            /* It seems that this kind always has a son that contains 
             the information we want (like TypeSig).
             However from previoius experiment it seems that sometimes
             the sons contains (but is it harmful since name
             have been resoluted during compilation and we know there
             is no ambiguity)
             'Object' and that this one contains 
             'java.lang.Object' so maybe something has to be fixed.
             
             corrected since by adding the double instanceof switch,
             the class avoided here are handled in other else if
             */
            TypeExpr m = (TypeExpr) n;

            if (!(m.type instanceof TypeName || m.type instanceof PrimitiveType || m.type instanceof ArrayType)) {
                String s = m.type.toString(); // here we get the type

                TName newNode = new TName(s);
                TNode.addName(s, "%Type");

                TNode.addType(s);

                currentParent.addSon(newNode);
            }

            if (dot)
                oldDot.append("\\n \"");
        } else if (n instanceof TypeName) { // javafe/Type
            // this represents a type

            TypeName m = (TypeName) n;
            String s = m.name.printName();

            if (s == null)
                TDisplay
                        .err("case n instanceof TypeName, warning null reference not expected");

            // we say that this name represents a type
            TName newNode = new TName(s);
            TNode.addName(s, "%Type");
            currentParent.addSon(newNode);

            if (dot)
                oldDot.append("\\n" + s + "\"");
        } else if (n instanceof TypeSig) {
            TypeSig m = (TypeSig) n;
            // 		// this represents a type

            // 		String s = m.simpleName;

            // 		TName newNode = new TName(s, TNode.$Type);
            // 		// we put as a variable name
            // 		TNode.addName(s, "%Type");

            // 		// we put as a type too
            // 		TNode.addType(s);

            // 		currentParent.addSon(newNode);

            if (dot)
                // fixme, not precise enough maybe
                oldDot.append("\\n" + m.simpleName + "\"");

        } else if (n instanceof VariableAccess) {
            VariableAccess m = (VariableAccess) n;

            String sTemp = Atom.printableVersion(UniqName.variable(m.decl));

            //String sTemp = new String(m.id.chars);

            // fixme the two are not the same all the time
            // 		error.println("\n");
            // 		error.println(m.id.chars);
            // 		error.println(m.decl.id.chars);
            // 		error.println("\n");

            // 		if(m.loc != 0)
            // 		    sTemp = sTemp+":"+UniqName.locToSuffix(m.loc);

            // add it to the global map containg all variables name
            TNode.addName(sTemp, null);

            /* we don't know the type, so null */
            TName nTemp = new TName(sTemp);
            currentParent.addSon(nTemp);

            if (dot) {
                /* display a square box for variable 
                 * + name of the variable 
                 */

                oldDot.append("\\n" + sTemp);

                // append line & column in the old format
                if (m.loc != Location.NULL)
                    oldDot.append(":" + UniqName.locToSuffix(m.loc));

                oldDot.append("\"");

                oldDot.append(", shape = box");
            }
        } else /* add the " */
        if (dot)
            oldDot.append("\"");

        if (dot) {
            oldDot.append("];\n");

            /* declaration of the sons */
            for (int i = 0; i < nbChilds; i++) {
                o = (Object) n.childAt(i);

                if (o instanceof ASTNode) {
                    oldDot.append(name);
                    oldDot.append(" -> ");
                    nodeTemp = (ASTNode) o;
                    oldDot.append(getNameASTNode(nodeTemp));
                    oldDot.append(nodeTemp.dotId);

                    /* red arrow for variables */
                    if (nodeTemp instanceof VariableAccess) {
                        VariableAccess va = (VariableAccess) nodeTemp;
                        oldDot.append(" [color = red]");
                        //		    error.println(va.id);
                        // 		    error.println(va.loc);
                        // 		    error.println(va.decl);

                    }

                    oldDot.append(";\n");
                }
            }

        }

        /* call on all the sons */

        for (int i = 0; i < nbChilds; i++) {
            o = (Object) n.childAt(i);

            if (o instanceof ASTNode) {
                nodeTemp = (ASTNode) o;

                generateIfpTree(nodeTemp, dot);
            }
        }

        if (saveParent == null) { // end of the first call
            newRootNode = currentParent;

            if (newRootNode == null)
                TDisplay.err("root node is null");

            if (!(newRootNode instanceof TRoot))
                TDisplay.err("root node doesn't have type TRoot");

            //
            //firstNotSkipped = false;

            // we type the tree.
            newRootNode.typeTree();
            
            TDisplay.info("tree has been typed");

        }

        // restore the parent
        currentParent = saveParent;

    }

    /**
     * Utility method for creating dot representation of gc tree
     *
     * @return transform escjava.prover.sammy to sammy, ie escjava.x.y to y
     */
    private static String getNameASTNode(ASTNode e) {

        String s = e.getClass().getName();
        int lastDotIndex = s.lastIndexOf('.');

        /* truncate to the class name without package before */
        if (lastDotIndex != -1)
            s = s.substring(lastDotIndex + 1, s.length());

        return s;

    }

    /**
     * Debugging purpose only.
     */
    public void printInfo() {
        if (!computationDone) {
            generateIfpTree(oldRootNode, true);

            computationDone = true;
        }

        TNode nTemp = (TNode) newRootNode;
        nTemp.printInfo();
    }

}
