package escjava.soundness;

import javafe.ast.ASTNode;
import javafe.ast.BinaryExpr;
import javafe.ast.ExprVec;
import escjava.ast.TagConstants;

public class Visit {
	
	   /*Helper method for retrieving all Binary "AND" and "OR"
     *occurences in the children of a Node. Performs a Depth First
     *Search and returns a vector of the results.
     *Used by the Consistency Checker - Conor 2005
     */      
    protected ExprVec visit(ASTNode mp){
        ExprVec beVec = ExprVec.make();
        
        for(int i = 0;i < mp.childCount();i++){
            if (mp.childAt(i) instanceof BinaryExpr){
                BinaryExpr b = (BinaryExpr)mp.childAt(i);
                if (b.getTag() == TagConstants.OR || b.getTag() == TagConstants.AND){
                    beVec.addElement(b);
                }
            }
             
            if (mp.childAt(i) instanceof ASTNode){
                beVec.append(visit( (ASTNode)mp.childAt(i) ));
                
            }
        }
        return beVec;  
    }

}
