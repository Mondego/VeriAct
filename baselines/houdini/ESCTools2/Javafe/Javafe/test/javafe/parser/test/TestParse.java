/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.parser.test;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.io.FileInputStream;
import java.io.PrintStream;

import javafe.ast.CompilationUnit;
import javafe.ast.PrettyPrint;
import javafe.parser.Lex;
import javafe.parser.Parse;
import javafe.util.CorrelatedReader;
import javafe.util.FileCorrelatedReader;
import javafe.util.Assert;
import javafe.util.ErrorSet;
import javafe.tc.TagConstants;

/**
 * Parse Java compilation units in various ways to test the Java
 * front-end parser.
 */

public class TestParse
{
    // The parser and scanner used to process files
    public static /*@ non_null @*/ Lex lexer = new Lex(null, true);

    public static /*@ non_null @*/ Parse parser = new Parse();

    // The following variables are set by command-line options.
    public static boolean check = false;
    public static boolean print = false;
    public static boolean progress = false;
    public static boolean verboseprogress = false;
    public static boolean idempotence = false;

    // Buffers used to hold intermediate results
    public static /*@ non_null @*/ MemoryPipeOutputStream out1 = new MemoryPipeOutputStream();
    public static /*@ non_null @*/ MemoryPipeOutputStream out2 = new MemoryPipeOutputStream();

    /**
     * Fail with a specific error message.
     *
     * @param msg the error message to print to {@link System.err}. 
     */
    //@ ensures false;
    public static void fatal(String msg) {
	System.err.println("Fatal error: " + msg);
	System.exit(99);
    }

    /**
     * @param argv the command-line arguments.
     * @bon_constraint If first command-line argument is "diff" then the
     * second and third command-line arguments must be the name of the
     * two file to parse and compare as a test.
     * @bon_constraint The command-line arguments may only be "diff",
     * "assert", "check", "print", "progress", "silent", or
     * "idempotence".
     * @bon_constraint The command-line argument "diff" may only be the
     * first command-line argument.
     */
    //@ requires \nonnullelements(argv);
    public static void main(String[] argv) throws IOException {

        // Allow for a test of the diff routine:
        if (argv.length > 0 && argv[0].equals("diff")) {
            if (argv.length < 3)
                fatal("Insufficient arguments to diff");
	    if (argv.length >= 4 && argv[3].equals("assert")) {
		if (javafe.Tool.options == null)
			javafe.Tool.options = new javafe.Options();
		javafe.Tool.options.assertIsKeyword = true;
	    }
            diff(argv[1], new FileInputStream(argv[1]),new FileInputStream(argv[2]));
            System.exit(0);
        }

        int exitStatus = 0;
	if (javafe.Tool.options == null)
	    javafe.Tool.options = new javafe.Options();
        for(int i = 0; i < argv.length; i++) {
            if (argv[i].equals("verboseprogress")) { verboseprogress=true;continue; }
            else if (argv[i].equals("check")) { check = true; continue; }
            else if (argv[i].equals("print")) { print = true; continue; }
            else if (argv[i].equals("progress")) { progress = true; continue; }
            else if (argv[i].equals("silent")) { ErrorSet.gag = true; continue; }
            else if (argv[i].equals("assert")) {
                javafe.Tool.options.assertIsKeyword = true; 
                continue;
            } else if (argv[i].equals("idempotence")) { idempotence = true; continue; }

            if (verboseprogress) System.out.println("Checking: " + argv[i]);
            else if (progress) {
                System.out.write('.');
                if (i != 0 && (i % 50) == 0) System.out.write('\n');
                System.out.flush();
            }

            out1.reset();
            out2.reset();
            prettyPrint(argv[i], 
                        new FileInputStream(argv[i]), out1, 
                        check, print);
            if (idempotence) {
                prettyPrint(argv[i] + ":printed", out1.getInputStream(), out2, 
                            false, false);
                if (diff(argv[i], out1.getInputStream(), out2.getInputStream()))
                    exitStatus = 2;
            }
        }
        if (progress) System.out.println("");
        System.exit(exitStatus);
    }

    /**
     * @bon Pretty-print a given parsed Java compilation unit to a specified output stream.
     * @bon Pretty-print a given parsed Java compilation unit to the system output stream.
     * @bon Check the invariant of a parsed Java compilation unit.
     * @param n the name of the input stream.
     * @param src the input stream from which to read the Java compilation unit(s).
     * @param dst the output stream on which to send the
     * pretty-printed parsed Java compilation unit.
     * @param check if <code>true</code> check the invariant of the
     * parsed Java compilation unit using the {@link ASTNode#check()}
     * method.
     * @param print if <code>true</code> pretty-print the parsed Java
     * compilation unit to {@link System.out}.
     */
    public static void prettyPrint(/*@ non_null @*/ String n,
                                   /*@ non_null @*/ InputStream src,
                                   /*@ non_null @*/ OutputStream dst,
                                   boolean check, boolean print) {
        CorrelatedReader in = new FileCorrelatedReader(src, n);
        PrintStream out = new PrintStream(dst);
        lexer.restart(in);
        CompilationUnit cu = parser.parseCompilationUnit(lexer, false);
        if (check) cu.check();
        PrettyPrint.inst.print(out, cu);
        if (print) PrettyPrint.inst.print(System.out, cu);
        in.close();
        out.close();
    }

    /**
     * Compare two input streams and prints a message and returns
     * <code>true</code> if they are different, return
     * <code>false</code> otherwise.
     *
     * @return <code>true</code> iff input stream <var>in1</var> and
     * <var>in2</var> are different.
     * @param n
     * @param in1 the first input stream to compare.
     * @param in2 the second input stream to compare.
     */
    public static boolean diff(String n,
                               /*@ non_null @*/ InputStream in1,
                               /*@ non_null @*/ InputStream in2) {
	StringBuffer sb = new StringBuffer();
        try {
            int c1 = in1.read(), c2 = in2.read();
            int line = 1, col = 1, offset = 1;
            while (c1 == c2 && c1 != -1 && c2 != -1) {
		//sb.append((char)c1);
                if (c1 == '\n') { line++; col = 1; }
                else col++;
                offset++;
                c1 = in1.read();
                c2 = in2.read();
            }
            if (c1 != c2) {
                if (progress) n = "\n" + n;
                System.out.println(n + " failed (" + (char)c1 + " != " + (char)c2
                                   + " at offset " + offset
                                   + ", line " + line + ", col " + col + ")");
		//System.out.println(sb.toString());
                return true;
            } else return false;
        } catch(IOException e) {
            e.printStackTrace();
            System.exit(2);
            return false; // Dummy
        }
    }
}

/**
 * A helper class for the parser test framework.  Values written to
 * this MemoryPipeOutputStream will be readable from its associated
 * MemoryPipeInputStream.
 */

final class MemoryPipeOutputStream extends OutputStream
{
    //@ invariant 16*1024 <= buff.length;
    /*@ non_null @*/ byte[] buff = new byte[16*1024];

    //@ invariant 0 <= ini;
    //@ invariant ini <= buff.length;
    int ini = 0;

    /**
     * Write the given value into the buffer <code>buff</code> at index
     * <code>ini</code>.
     * @param b the value to write into the buffer.
     */
    //@ ensures buff[ini] == b;
    //@ ensures ini == \old(ini + 1);
    public void write(int b) {
        try {
            buff[ini] = (byte)b;	//@ nowarn IndexTooBig;  // caught
            ini++;
        } catch (ArrayIndexOutOfBoundsException e) {
            byte[] newbuff = new byte[2*buff.length];
            System.arraycopy(buff, 0, newbuff, 0, buff.length);
            buff = newbuff;
            buff[ini++] = (byte)b;
        }
    }

    //@ ensures ini == 0;
    void reset() {
      ini = 0;
    }

    //@ ensures \fresh(\result);
    /*@ non_null @*/ MemoryPipeInputStream getInputStream() { 
      return new MemoryPipeInputStream(this);
    }
}

/**
 * A helper class for the parser test framework. 
 */

final class MemoryPipeInputStream extends InputStream
{
    /*@ non_null @*/ MemoryPipeOutputStream src;

    //@ invariant outi >= 0;
    int outi = 0;

    /**
     * Create a new input stream connected to the given output stream.
     * Bytes read from this input stream will be exactly those bytes
     * written to the output stream <code>src</code>, in the same order.
     *
     * @param src the MemoryPipeOutputStream this straem is attached to.
     */
    //@ ensures this.src == src;
    MemoryPipeInputStream(/*@ non_null @*/ MemoryPipeOutputStream src) {
      this.src = src;
    }

    /**
     * @return a -1 if there is no value available on this input stream,
     * otherwise return the next value that is available.
     */
    //@ public normal_behavior
    //@   requires src.ini <= outi;
    //@   ensures \result == -1;
    //@ also
    //@ public normal_behavior
    //@   requires outi < src.ini;
    //@   ensures \result == src.buff[outi];
    //@   ensures outi == \old(outi + 1);
    public int read() {
        int result = (src.ini <= outi) ? -1 : src.buff[outi++];
        return result;
    }
}
