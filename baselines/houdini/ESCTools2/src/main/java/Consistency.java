package escjava.soundness;



import javafe.ast.*;

import javafe.tc.TypeSig;
import javafe.util.FatalError;
import escjava.ast.DerivedMethodDecl;
import escjava.ast.ExprModifierPragma;
import escjava.translate.InitialState;
import escjava.Main;


public class Consistency {
	
	
    DerivedMethodDecl dmd = null;
    ExprModifierPragma oMP = null;
    ExprModifierPragma tMP = null;
    ExprVec bvec = null;
    String status = "error";
    
    public void consistency(RoutineDecl r, TypeSig sig,
		     InitialState initState, long startTime){
    Object[] obj = r.getDecorations();
    for (int i = 0; i < obj.length; i++){
        if (obj[i] instanceof DerivedMethodDecl){
            dmd = (DerivedMethodDecl)obj[i];
            //System.out.println(dmd.requires.toString());

        }
    }
        
    for (int i = 0; i < dmd.requires.size(); i++){
        oMP = dmd.requires.elementAt(i);
        Visit v = new Visit();
        bvec = v.visit(oMP);
        
        //Remove Precondition
        for (int j = 0; j < bvec.size(); j ++){
            BinaryExpr be = (BinaryExpr) bvec.elementAt(j);

            for (int k = 0; k < 2; k++){
                if (k == 0){
                    tMP = ExprModifierPragma.make(be.left.getTag(), be.left, oMP.getStartLoc() );
                    System.out.println();
                    System.out.println("Rechecking with only the following Specifications (see VariableAccess id):");
                    System.out.println("....................................................................................");
                    System.out.println(be.left.toString());
                    System.out.println("....................................................................................");
                }
                else{
                    tMP = ExprModifierPragma.make(be.right.getTag(), be.right, oMP.getStartLoc() );
                    System.out.println();
                    System.out.println("Rechecking with only the following Specifications (see VariableAccess id):");
                    System.out.println("....................................................................................");
                    System.out.println(be.right.toString());
                    System.out.println("....................................................................................");
                }
                dmd.requires.removeElementAt(i);
                dmd.requires.insertElementAt(tMP, i);
                
                try{
                	                	
                    status = Main.getInstance().processRoutineDecl(r, sig, initState);
                } catch (javafe.util.NotImplementedException e) {
                    // continue - problem already reported
                    status = "not-implemented";
                } catch (FatalError e) {
                    // continue;
                }
                dmd.requires.removeElementAt(i);
                dmd.requires.insertElementAt(oMP, i);
                System.out.println("    [" + javafe.Tool.timeUsed(startTime) + "] Status:  " + status);

            }
        }
    }
    }

}
