package escjava.vcGeneration.coq.visitor;

import java.io.*;

import escjava.vcGeneration.*;
import escjava.vcGeneration.coq.CoqProver;
import escjava.vcGeneration.PrettyPrinter;

/**
 * This class implements all the functions that are not handled
 * by the Coq visitor. 
 * @author J.Charles
 * @version 14/11/2005
 *
 */
public abstract class ANotHandledVisitor extends AArrayOpsVisitor{

	protected ANotHandledVisitor(Writer out, CoqProver prover, PrettyPrinter ppout) {
		super(out, prover, ppout);
	}
	
    // FIXME: LockLE and LockLT have the same symbol
    public void visitTLockLE(/*@ non_null @*/ TLockLE n){
    	binOp("lockLess", n);
    }

    public void visitTLockLT(/*@ non_null @*/ TLockLT n){
    	binOp("lockLess", n);
    }
    
    public void visitTCast(/*@ non_null @*/ TCast n){
    	genericFun("cast", n);
    }
			
    public void visitTIs(/*@ non_null @*/ TIs n){
    	out.appendN("(* TIs *) True");
    }

    /**
     * FIXME: Not handled yet.
     */
    public void visitTExist(/*@ non_null @*/ TExist n){
    	out.appendN("(* TExist *) True");
    }
}
