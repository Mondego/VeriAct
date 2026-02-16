/* Copyright 2000, 2001, Compaq Computer Corporation */

package houdini;

import escjava.prover.*;
import java.io.*;


/**
 ** Parsers SExpressions from an input stream
 **/

public class SExpParser {

    private PushbackInputStream in;

    public SExpParser(InputStream in) {
        this.in = new PushbackInputStream(in);
    }

    /** internal buffer. */
    private StringBuffer sb = new StringBuffer();

    /**
     ** returns:
     **   int as Integer
     **   ()  as Character
     **   name as String
     **   eof  as null
     **/
    private Object nextToken() throws IOException {
        sb.setLength(0);
        while (true) {
            int c = in.read();
            if (c == -1) return null;
            char ch = (char)c;
            if (ch == '-') {
                int c2 = in.read();
                in.unread(c2);
                if (c2 == -1 || Character.isDigit((char)c2)) {
                    return scanInt(ch);
                }
            }
            if (Character.isDigit(ch)) {
                return scanInt(ch);
            } else if (ch == ')' || ch == '(') {
                return new Character(ch);
            } else if (ch == ';') {
                while (c != '\n' && c != -1) {
                    c = in.read();
                }
            } else if (Character.isWhitespace(ch)) {
                continue;
            } else {
                return scanName(ch);
            }
        }
    }
    
    private Integer scanInt(int ch) throws IOException {
        sb.append((char)ch);
        while (true) {
            ch = in.read();
            if (ch == -1) break;
            if (Character.isDigit((char)ch)) {
                sb.append((char)ch);
            } else {
                in.unread(ch);
                break;
            }
        }
        return new Integer(sb.toString());
    }
     
    static private String special = "()\n \t";
    
    String scanName(int ch) throws IOException {
        sb.append((char)ch);
        while (true) {
            ch = in.read();
            if (ch == -1) break;
            if (special.indexOf(ch) == -1) {
                sb.append((char)ch);
            } else {
                in.unread(ch);
                break;
            }
        }
        String s = sb.toString();
        if (s.startsWith("|") && s.endsWith("|")) {
            return s.substring(1, s.length() - 1);
        } else {
            return s;
        }
    }        
    
    SExp parseList() throws SExpTypeError {
        SList s = SList.make();
        while (true) {
            SExp e = parse();
            if (e == null) {
                break;
            }
            s = s.addFront(e);
        }
        return SList.reverseD(s);
    }

    /**
     * return the next SExp in the stream or throw an error 
     */
    public SExp parse() throws SExpTypeError {
        Object o;
        try {
            o = nextToken();
        } catch (IOException e) {
            throw new SExpTypeError();
        }

        if (o == null) throw new SExpTypeError();
        
        if (o instanceof Integer || o instanceof String) {
            return SExp.fancyMake(o);
        }
        
        if (o instanceof Character) {
            char ch = ((Character)o).charValue();
            if (ch == ')') return null;
            if (ch != '(') throw new SExpTypeError();
            return parseList();
        }
	throw new SExpTypeError();
    }
    
    public static void main(String s[]) throws IOException, SExpTypeError {
        String ss = "(x 1 2 3)\n;moo\n(xc (1 2 3) <: (|moo))\n|dog|\n";
        SExpParser p = new SExpParser(new StringBufferInputStream(ss));
        SExp e;
        while ((e = p.parse()) != null) {
            e.print();            
            System.out.println();
            SExp.display(e);
        }
    }



}
