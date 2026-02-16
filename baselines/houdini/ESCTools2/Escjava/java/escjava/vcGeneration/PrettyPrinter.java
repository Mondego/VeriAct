package escjava.vcGeneration;

import java.io.*;

import javafe.util.ErrorSet;

public class PrettyPrinter {
	
	private final String TAB;
	private final String LBR;
	private final String RBR;
	private final String NL;
	
	
	String commentMark = "";
	
	/*@ non_null @*/Writer out = null;
	
	/*@ non_null @*/StringBuffer indentation = null;
	
	public PrettyPrinter(Writer out, String tab, String lbr, String rbr, String nl) {
		this.out = out;
		indentation = new StringBuffer();
		TAB = tab;
		LBR = lbr;
		RBR = rbr;
		NL = nl;
	}
	
	private void write(String s)
	{
		try {
			out.write(s);
		} catch (IOException e) {
			ErrorSet.fatal("internal error: " + e.getMessage());
		}
	}
	
	/**
	 * just append the parameter, N stands for normal
	 */
	public/*@ non_null @*/Writer appendN(/*@ non_null @*/String s) {
		if (!commentMark.equals(""))
			s.replaceAll(NL,NL+commentMark);
		write(s);
		return out;
	}
	
	/**
	 * append indentation and parameter
	 */
	public/*@ non_null @*/Writer append(/*@ non_null @*/String s) {
		if (!commentMark.equals(""))
			s.replaceAll(NL,NL+commentMark);
		write(indentation.toString());
		write(s);
		
		return out;
	}
	
	/**
	 * This function goes to next line, increases indentation by two spaces
	 * and add a '('
	 * If you want to change the indentation style do it here.
	 */
	public/*@ non_null @*/Writer appendI(/*@ non_null @*/String s) {
		if (!commentMark.equals(""))
			s.replaceAll(NL,NL+commentMark);
		if (indentation.length() != 0) { // not first call
			write(NL + commentMark);
			write(indentation.toString());
		}
		else
			write(commentMark);
		
		indentation.append(TAB);
		
		write(LBR + s);
		return out;
	}
	
	/**
	 * This function increases indentation by two spaces
	 * and add a '('
	 * If you want to change the indentation style do it here.
	 */
	public/*@ non_null @*/Writer appendIwNl(/*@ non_null @*/String s) {
		if (!commentMark.equals(""))
			s.replaceAll(NL,NL+commentMark);
		indentation.append(TAB);
		write(indentation.toString());
		write(LBR + s);
		
		return out;
	}
	
	/**
	 * This function goes to new line, add a ')' and
	 * reduces indentation by two spaces
	 * If you want to change the indentation style do it here.
	 */
	public/*@ non_null @*/Writer reduceI() {
		write(NL + commentMark);
		indentation = indentation.delete(0, TAB.length());
		write(indentation + RBR);
		
		return out;
	}
	
	/**
	 * This function add a ')' and
	 * reduces indentation by two spaces
	 * If you want to change the indentation style do it here.
	 */
	public/*@ non_null @*/Writer reduceIwNl() {
		write(RBR);
		indentation = indentation.delete(0, TAB.length());
		
		return out;
	}
	
	public/*@ non_null @*/String toString() {
		return out.toString();
	}
	
	/**
	 * This adds a user-specified comment mark string to the beginning of each
	 * line.  It can be removed by wither recalling setC with the empty
	 * string or calling removeC.  It immediately adds one instance of the
	 * comment marker.
	 */
	public /*@ non_null @*/Writer beginC(/*@ non_null @*/String s) {
		commentMark = s;
		write(s);
		return out;
	}
	
	/**
	 * This function removes the comment mark set by setC() and appends a
	 * newline;
	 */
	public/*@ non_null @*/Writer endC() {
		commentMark = "";
		write(NL);
		return out;
	}
	
	
}
