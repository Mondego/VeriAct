package escjava.vcGeneration.coq.visitor;

import java.io.*;

import escjava.vcGeneration.*;
import escjava.vcGeneration.coq.CoqProver;
import escjava.vcGeneration.PrettyPrinter;

public abstract class ABasicCoqVisitor extends TVisitor{
    protected ABasicCoqVisitor tcbv;
    protected ABasicCoqVisitor tcv;
    
    /**
     * The output buffer.
     */
    protected PrettyPrinter out = null;
    /**
     * The current instance of the CoqProver.
     */
    protected CoqProver p;
    
    
    
    protected ABasicCoqVisitor(Writer out, CoqProver prover, PrettyPrinter ppout) {
        super(out, "  ", "(", ")", "\n");
        if (ppout == null)
            this.out = new PrettyPrinter(out, "  ", "(", ")", "\n");
        else
            this.out = ppout;

        p = prover;
    }
    
    protected void setVisitors(ABasicCoqVisitor tcv, ABasicCoqVisitor tcbv) {
    	this.tcv = tcv;
    	this.tcbv = tcbv;
    }
    /*
     * General Function
     * You give the operator at the first argument and then it outputs
     * op (son1, son2 ...)
     * )
     */
    public void genericFun(/*@ non_null @*/ String s, TFunction n){

    	out.appendI(s+" ");
    	int i =0;
    	for(; i < n.sons.size(); i++) {
    		n.getChildAt(i).accept(tcv);
    		if(i != n.sons.size() - 1)
    			out.appendN(" ");
    	}


    	if((n.getChildAt(--i)) instanceof TName || (n.getChildAt(--i)) instanceof TLiteral)
    		out.reduceIwNl();
    	else
    		out.reduceI();    
    }
    
    public void propFun(/*@ non_null @*/ String s, TFunction n){

    	out.appendI(s+" ");
    	
    	int i =0;
    	for(; i < n.sons.size(); i++) {
    		
    	    n.getChildAt(i).accept(tcbv);

    	    /*
    	     * If not last, output a space
    	     */
    	    if(i != n.sons.size() - 1)
    		out.appendN(" ");
    	}

    	if((n.getChildAt(--i)) instanceof TName || (n.getChildAt(--i)) instanceof TLiteral)
    	    out.reduceIwNl();
    	else
    	    out.reduceI();
    	    
        }
    /*
     * Function/Operator with arity 1 :
     * (op X)
     */
    public void unaryGeneric(/*@ non_null @*/ String s, TFunction n){

	if(n.sons.size() != 1)
	    System.err.println("java.escjava.vcGeneration.TCoqVisitor.unFun : an unary operator named "+s+" has a number of sons equals to "+n.sons.size()+" which is different from 1");
	out.appendI("");
	out.appendN(s);
	n.getChildAt(0).accept(tcv);
	if((n.getChildAt(0)) instanceof TName || (n.getChildAt(0)) instanceof TLiteral)
	    out.reduceIwNl();
	else
	    out.reduceI();
	    
    }

    /*
     * You give the operator at the first argument and then it outputs
     *   (son1 
        op 
	  son2 ... 
	op 
	  sonN)
     */
    public void genericOp(/*@ non_null @*/ String s, TFunction n){

	out.appendI("");
	for(int i =0; i < n.sons.size(); i++) {

	    n.getChildAt(i).accept(tcv);
	    /*
	     * not the last
	     */
	    if(i != n.sons.size() - 1) {
	    	out.appendN("\n");
	    	out.append(s);
	    }
	    
	    out.appendN(" ");
	}
	out.reduceI();
    }
    public void genericPropOp(/*@ non_null @*/ String s, TFunction n){
    	out.appendI("");
    	for(int i =0; i < n.sons.size(); i++) {
    	    n.getChildAt(i).accept(tcbv);
    	    /*
    	     * not the last
    	     */
    	    if(i != n.sons.size() - 1) {
    	    	out.appendN(" " + s + " ");
    	    }

    	}
    	
    	out.reduceI();
        }
    /*
     * Function/Operator with arity 1 :
     * (op X)
     */
    public void unaryProp(/*@ non_null @*/ String s, TFunction n){

	if(n.sons.size() != 1)
	    System.err.println("java.escjava.vcGeneration.TCoqVisitor.unFun : an unary operator named "+s+" has a number of sons equals to "+n.sons.size()+" which is different from 1");
	out.appendI("");
	out.appendN(s);
	n.getChildAt(0).accept(tcbv);
	if((n.getChildAt(0)) instanceof TName || (n.getChildAt(0)) instanceof TLiteral)
	    out.reduceIwNl();
	else
	    out.reduceI();
	    
    }
    /*
     * Function for binary operator
     * You give the operator at the first argument and then it outputs
     * (son1 
     * op 
     * son2)
     * 
     * If son1 is a variable, op isn't on the next line
     * If son2 is a variable, it doesn't go to next line.
     */
    public void binOp(/*@ non_null @*/ String s, TFunction n){

	if(n.sons.size() != 2)
	    System.err.println("java.escjava.vcGeneration.TCoqVisitor : a binary operator named "+s+" has a number of sons equals to "+n.sons.size()+" which is different from 2");

	out.appendI("");
	
	n.getChildAt(0).accept(tcv);

	if(! ((n.getChildAt(0)) instanceof TName || (n.getChildAt(0)) instanceof TLiteral)) {
	    out.appendN("\n");
	    out.append("");
	}
	
	out.appendN(" "+s+" ");
	n.getChildAt(1).accept(tcv);
	if((n.getChildAt(1)) instanceof TName || (n.getChildAt(1)) instanceof TLiteral)
	    out.reduceIwNl();
	else
	    out.reduceI();
	    
    }
    public void spacedBinOp(/*@ non_null @*/ String s, TFunction n){

    	if(n.sons.size() < 2 )
    	    System.err.println("java.escjava.vcGeneration.TCoqVisitor : the spaced out binary operator named "+s+" has a number of sons equals to "+n.sons.size()+" which is different from 2");

    	out.appendI("");
    	for(int i =0; i < n.sons.size() - 1; i++) {
    		
    		if(i != 0) {
    			out.appendN(" /\\ ");
    		}
//    		else
//    			out.appendN("");
    		// Frankie says... no equality on Prop
    		n.getChildAt(i).accept(tcv);    	
    		out.appendN(" "+s+" ");
    		n.getChildAt(i+1).accept(tcv);
//    		out.appendN("");
    	}
    	if((n.getChildAt(1)) instanceof TName || (n.getChildAt(1)) instanceof TLiteral)
    	    out.reduceIwNl();
    	else
    	    out.reduceI();
    	    
        }
}
