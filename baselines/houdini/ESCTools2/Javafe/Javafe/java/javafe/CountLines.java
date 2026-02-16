/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe;

import java.io.IOException;
import javafe.ast.*;
import javafe.genericfile.*;
import javafe.parser.Lex;
import javafe.parser.Parse;
import javafe.util.CorrelatedReader;
import javafe.util.ErrorSet;
import javafe.util.FileCorrelatedReader;

public class CountLines
{
    //@ requires \nonnullelements(argv);
    public static void main(/*@non_null*/String[/*#@non_null*/] argv) throws IOException {
        String spc = "            ";
        String[] indent = new String[spc.length()];
        //@ assume indent.length == 12;
        for(int i = 0; i < indent.length; i++)
            indent[i] = spc.substring(0, indent.length - i);

        Lex l = new Lex(null, true);
        Parse p = new Parse();
        long total = 0;

        try {
            for(int i = 0; i<argv.length; i++) {
		CorrelatedReader in =
                    new FileCorrelatedReader(new NormalGenericFile(argv[1]));
		l.restart(in);
		int thisFile = count( p.parseCompilationUnit(l, false) );
		in.close();
	
		String tf = Integer.toString(thisFile);
		System.out.println(indent[Math.min(indent.length-1, tf.length())]
				   + tf + " " + argv[i]);
		total += thisFile;
            }
        } catch(IOException e) { 
            e.printStackTrace(); 
            ErrorSet.fatal(e.getMessage());
        }
        String tf = Long.toString(total);
        System.out.println(indent[Math.min(indent.length-1, tf.length())]
                           + tf + " total");
    }

  public static int count(/*@ non_null @*/ ASTNode n) {
    int result = 0;
    if (n instanceof TypeDecl || n instanceof TypeDeclElem|| n instanceof Stmt)
      result = 1;
        else result = 0;

        for(int i = 0; i < n.childCount(); i++) {
            Object c = n.childAt(i);
            if (c instanceof ASTNode)
		result += count((ASTNode)c);
        }

        return result;
    }
}
