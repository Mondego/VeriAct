package escjava.vcGeneration.coq.visitor;

import java.io.*;

import escjava.vcGeneration.*;
import escjava.vcGeneration.coq.CoqProver;
import escjava.vcGeneration.PrettyPrinter;

public abstract class AIntegralVisitor extends ABasicCoqVisitor {
	
	protected AIntegralVisitor(Writer out, CoqProver p, PrettyPrinter ppout) {
		super(out, p, ppout);
	}
    
	public void visitTIntegralEQ(/*@ non_null @*/ TIntegralEQ n){
    	binOp(" =", n);
    }
    
    public void visitTIntegralGE(/*@ non_null @*/ TIntegralGE n){
    	binOp(">=", n);
    }
    
    public void visitTIntegralGT(/*@ non_null @*/ TIntegralGT n){
    	binOp(">", n);
    }

    public void visitTIntegralLE(/*@ non_null @*/ TIntegralLE n){
    	binOp("<=", n);
    }
    
    public void visitTIntegralLT(/*@ non_null @*/ TIntegralLT n){
    	binOp("<", n);
    }
    
    public void visitTIntegralNE(/*@ non_null @*/ TIntegralNE n){
    	binOp("<>", n);
    }
    
    public void visitTIntegralAdd(/*@ non_null @*/ TIntegralAdd n){
    	binOp("+", n);
    }
    
    public void visitTIntegralDiv(/*@ non_null @*/ TIntegralDiv n){
    	binOp("/", n); 
    }

    public void visitTIntegralMod(/*@ non_null @*/ TIntegralMod n) {
    	binOp("mod", n); 
    }

    public void visitTIntegralMul(/*@ non_null @*/ TIntegralMul n){
    	binOp("*", n);
    }

	public void visitTIntegralSub(/*@non_null*/TIntegralSub n) {
		binOp("-", n);
	}
    public void visitTInt(/*@ non_null @*/ TInt n){
    	out.appendN(n.value + ""); 
    }
}
