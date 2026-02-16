package escjava.vcGeneration.coq.visitor;

import java.io.*;

import escjava.vcGeneration.*;
import escjava.vcGeneration.coq.CoqProver;
import escjava.vcGeneration.coq.TCoqVisitor;


/**
 * In the Coq extension, we differentiate <code>Boolean</code>
 * value and variables from usual <code>Prop</code>.
 * So we have 2 visitors which are coupled. 
 * <p>
 * This is the visitor used for <code>Boolean</code> specific values.  
 * @author J.Charles
 * @version 14/11/2005
 */
public class TCoqBoolVisitor extends TCoqVisitor {
	
	/**
	 * Constructor of the couple  TCoqVisitor/TCoqBoolVisitor.
	 * It should not be called by any other class than TCoqVisitor.
	 * @param v the current visitor.
	 * @param o the current CoqStringBuffer.
	 * @param p the current instance of CoqProver.
	 */
	public TCoqBoolVisitor(Writer out, TCoqVisitor v, PrettyPrinter o, CoqProver p){
    	super(out, v, o, p);	
    }
	
	
	/**
	 * Boolean value seen as a prop boolean value:
	 * <code>True</code> or <code>False</code>.
	 * @see TCoqVisitor#visitTBoolean(TBoolean)
	 */
	public void visitTBoolean(/*@ non_null @*/ TBoolean n){	
		if(n.value)
		    out.appendN("True");
		else
		    out.appendN("False");
	 }
	 
	 /**
	  * Boolean variables that are used within a prop
	  * should be translated in the form <code>myvar = true</code>
	  * @param n the variable to translate.
	  */
	 public void visitTName(/*@ non_null @*/ TName n){
		    VariableInfo vi = TNode.getVariableInfo(n.name);
		    String name = p.getVariableInfo(vi);
		    if(name.equals("Z"))
		    	name = "T_Z";
			out.appendN(name + " = true");
	 }
	 
	 public void visitTSelect(/*@ non_null @*/ TSelect n){
	    	String pre = "";
	    	if(TNode._integer.equals(((TNode)n.sons.get(1)).type))
	    		pre = "arr";
	    	genericFun("BoolHeap." + pre +"select ", n);
	    	out.appendN(" = true");
	    }
	 
	 public void visitTBoolNot(/*@ non_null @*/ TBoolNot n){
	    	unaryProp("not ", n);
	 }
}
