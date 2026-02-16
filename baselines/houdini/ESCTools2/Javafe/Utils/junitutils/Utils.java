/*
 * Copyright (C) 2000-2001 Iowa State University
 *
 * This file is part of mjc, the MultiJava Compiler, adapted for ESC/Java2.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 *
 * $Id$
 * Author: David R. Cok
 */

package junitutils;
import java.io.PrintStream;
import java.io.*;
import java.lang.reflect.Method;

/** This class contains miscellaneous (static) utility functions that are
 useful in writing JUnit functional tests.
 */
public class Utils {
  
  /** Setting this field to true disables the capturing of the output;
   one would do this only for debugging purposes.
   */
  static public boolean disable = false;
  
  /** A cached value of the usual System out stream. */
  //@ spec_public
  final private static /*@non_null*/ PrintStream pso = /*+@ (non_null PrintStream) */ System.out;

  /** A cached value of the usual System err stream. */
  //@ spec_public
  final private static /*@non_null*/ PrintStream pse = /*+@ (non_null PrintStream) */ System.err;
  
  /** Redirects System.out and System.err to the given PrintStream. 
   Note that setStreams/restoreStreams operate on the global
   values of System.out and System.err; these implementations
   are not synchronized - you will need to take care of any
   race conditions if you utilize these in more than one thread.
   
   @param ps The stream that is the new output and error stream
   */
    /*@ public normal_behavior
      @   requires disable;
      @   assignable \nothing;
      @ also
      @ public normal_behavior
      @   requires !disable;
      @   requires ps.isOpen;
      @   assignable System.out, System.err;
      @   ensures System.out == ps 
      @        && System.err == ps;
      @*/
  static public void setStreams(/*@ non_null */ PrintStream ps) {
    if (disable) return;
    System.out.flush();
    System.err.flush();
    System.setOut(ps);
    System.setErr(ps);
  }
  
  /** Creates a new output stream (which is returned) and makes it
   the stream into which the standard and error outputs are captured.
   
   @return an output stream into which standard and error output 
   is captured
   */
    /*@ public normal_behavior
      @   requires disable;
      @   assignable \nothing;
      @   ensures \fresh(\result);
      @ also
      @ public normal_behavior
      @   requires !disable;
      @   assignable System.out, System.err;
      @   ensures \fresh(\result);
      @   ensures \result.isOpen;
      @   ensures System.out.underlyingStream == \result;
      @   ensures System.err.underlyingStream == \result;
      @   ensures System.err == System.out;
      @*/
  static public /*@ non_null */ ByteArrayOutputStream setStreams() {
    ByteArrayOutputStream ba = new ByteArrayOutputStream(10000);
    PrintStream ps = new PrintStream(ba);
    setStreams(ps);
    return ba;
  }
  // TODO - note the hard-coded size of the stream above.  It needs to be
  // variable, or at least be sure to capture overflows.  The size needs to
  // be large enough to hold the output of a test.
  
  /** Restores System.out and System.err to the initial, 
   system-defined values. It is ok to call this method
   even if setStreams has not been called.
   <p>
   Note that setStreams/restoreStreams operate on the global
   values of System.out and System.err; these implementations
   are not synchronized - you will need to take care of any
   race conditions if you utilize these in more than one thread.
   */
    /*@ public normal_behavior
      @   requires disable;
      @   assignable \nothing;
      @ also
      @ public normal_behavior
      @   requires !disable;
      @   assignable System.out, System.err;
      @   ensures System.out == pso
      @        && System.err == pse;
      @*/
  static public void restoreStreams() {
    restoreStreams(false);
  }
  
  /** Restores System.out and System.err to the initial, 
   system-defined values. It is ok to call this method
   even if setStreams has not been called.
   <p>
   Note that setStreams/restoreStreams operate on the global
   values of System.out and System.err; these implementations
   are not synchronized - you will need to take care of any
   race conditions if you utilize these in more than one thread.
   
   * @param close if true, the current output and error streams
   *   are closed before being reset (if they are not currently the
   *   System output and error streams)
   */
    /*@ public normal_behavior
      @   requires disable;
      @   assignable \nothing;
      @ also
      @ public normal_behavior
      @   requires !disable;
      @   assignable System.out, System.err;
      @   ensures System.out == pso
      @        && System.err == pse;
      @   ensures (close && \old(System.out) != pso) ==> \old(System.out).wasClosed;
      @   ensures (close && \old(System.err) != pse) ==> \old(System.err).wasClosed;
      @*/
  static public void restoreStreams(boolean close) {
    if (disable) return;
    if (close) {
      if (pso != System.out) System.out.close();
      if (pse != System.err) System.err.close();
    }
    System.setOut(pso);
    System.setErr(pse);
  }
  
  
  
  /** Parses a string into arguments as if it were a command-line, using
   the QuoteTokenizer to parse the tokens.
   
   @param s The String to parse
   @return The input string parsed into command-line arguments
   */
  //@ ensures \nonnullelements(\result);
  static public /*@ non_null */ String[/*#@non_null*/] parseLine(/*@ non_null */String s) {
    QuoteTokenizer q = new QuoteTokenizer(s);
    java.util.ArrayList args = new java.util.ArrayList();
    while (q.hasMoreTokens()) {
      String a = q.nextToken();
      args.add(a);
    }
    Object[] o = args.toArray(new String[args.size()]);
    return (/*+@non_null*/String[/*#@non_null*/])o; //@ nowarn Cast;
  }
  
  
  /** Parses a string into arguments as if it were a command-line, using
   the QuoteTokenizer to parse the tokens. Adds to the existing list of
   arguments.
   
   @param s The String to parse
   @return The input string parsed into command-line arguments
   */
  //@ ensures \nonnullelements(\result);
  static public /*@ non_null */ String[] parseLineWithArgs(/*@ non_null */String s, /*@ non_null */String[] originalArgs) {
    QuoteTokenizer q = new QuoteTokenizer(s);
    java.util.ArrayList args = new java.util.ArrayList();
    
    // Add in existing arguments
    for (int i=0; i < originalArgs.length; i++) {
      String a = originalArgs[i];
      args.add (a);
    }
    
    // Add in specific arguments from the list of files
    while (q.hasMoreTokens()) {
      String a = q.nextToken();
      args.add(a);
    }
    return (String[])args.toArray(new String[args.size()]);
  }
  
  /** An enumerator that parses a string into tokens, according to the
   rules a command-line would use.  White space separates tokens,
   with double-quoted and single-quoted strings recognized.
   */
  // FIXME - does not handle escape sequences in strings, nor 
  // quoted strings that are not the whole token, e.g.
  // a b c+"asd"  should be three tokens, "a", "b", and  c+"asd" .
  static public class QuoteTokenizer {
    /** The String being tokenized */
    /*@ spec_public */ final private /*@non_null*/ String ss;
    
    /** A char array representation of the String being tokenized */
    /*@ spec_public */ final private /*@non_null*/ char[] cc;
    
    /** The position in the char array */
    /*@ spec_public */ private int pos = 0;
    /*@ invariant pos >= 0;
      @ invariant pos <= cc.length;
      @*/
    
    /** Initializes the tokenizer with the given String
     * @param s the String to be tokenized
     */
    //@ modifies ss,cc;
    //@ ensures s == ss;
    public QuoteTokenizer(/*@ non_null */ String s) {
      ss = s;
      cc = s.toCharArray();
    }

      /*@ public model boolean moreTokens;
	@ public represents moreTokens <-
	@ 	(\exists int i; pos <= i && i < cc.length;
	@		Character.isWhitespace(cc[i]));
	@*/

      /*@ public model boolean moreChar;
	@ public represents moreChar <- pos < cc.length;
	@*/

      //@ public invariant_redundantly moreTokens ==> moreChar;
      /*+@ public invariant_redundantly 
	@ moreChar && !Character.isWhitespace(cc[pos]) ==> moreTokens;
	@*/

    /**
     * @return true if there are more tokens to be returned
     */
      /*@ public normal_behavior
	@ modifies pos;
	@ ensures \result == moreChar;
	@ ensures \result ==> !Character.isWhitespace(cc[pos]); */
     /*+@ ensures_redundantly \result ==> moreTokens;
	@ ensures \result == moreTokens;		// true but esc cannot prove
	@ ensures moreTokens == \old(moreTokens);	// true but esc cannot prove
	@*/
    public boolean hasMoreTokens() {
      while (pos < cc.length && Character.isWhitespace(cc[pos])) pos++;
      return pos < cc.length;
    }

    /**
     * @return the next token if there is one, otherwise null
     */
      /*@ public normal_behavior
	@  requires moreChar;
	@  modifies pos;
	@  ensures  pos > \old(pos);
	@ also public normal_behavior
	@  requires !moreChar;
	@  modifies \nothing;
	@  ensures \result == null;
	@*/
      //@ also signals_only IndexOutOfBoundsException;
     /*+@ also
	@ public normal_behavior
	@   requires moreTokens;
	@   modifies pos;
	@   ensures \result != null;
	@ also public normal_behavior
	@   requires !moreTokens;
	@   modifies pos;
	@   ensures \result == null;
	@*/
    public /*@nullable*/String nextToken() {
      String res = null;
      while (pos < cc.length && Character.isWhitespace(cc[pos])) ++pos;
      if (pos == cc.length) return res;
      int start = pos;
      if (cc[pos] == '"') {
        ++pos;
        while (pos < cc.length && cc[pos] != '"' ) ++pos;
        if (cc[pos] == '"') ++pos; //@ nowarn IndexTooBig;
        res = ss.substring(start+1,pos-1);
      } else if (cc[pos] == '\'') {
        ++pos;
        while (pos < cc.length && cc[pos] != '\'' ) ++pos;
        if (cc[pos] == '\'') ++pos; //@ nowarn IndexTooBig;
        res = ss.substring(start+1,pos-1);
      } else {
        while (pos < cc.length && !Character.isWhitespace(cc[pos])) ++pos;
        res = ss.substring(start,pos);
      }
      return res;
    } //@ nowarn Exception;
  }
  
  /** Deletes the contents of a directory, including subdirectories.  
   If the second argument is true, the directory itself is deleted as well.
   
   @param d The directory whose contents are deleted
   @param removeDirectoryItself if true, the directory itself is deleted
   
   @return  false if something could not be deleted; 
   true if everything was successfully deleted
   */
  static public boolean recursivelyRemoveDirectory(/*@ non_null */ File d, 
      boolean removeDirectoryItself) {
    if (!d.exists()) return true;
    boolean success = true;
    File[] fl = d.listFiles();
    if (fl != null) {
      for (int ff=0; ff<fl.length; ++ff) {
        if (fl[ff].isDirectory()) {
          if (!recursivelyRemoveDirectory(fl[ff],true)) 
            success = false;
        } else {
          if (!fl[ff].delete()) success = false;
        }
      }
    }
    if (removeDirectoryItself) {
      if (!d.delete()) success = false;
    }
    return success;
  }
  
  /** Reads the contents of the file with the given name, returning a String.
   This is an optimized version that uses the byte array provided and 
   presumes that the file is shorter than the length of the array.
   
   @param filename		The file to be read
   @param cb		A temporary byte array to speed reading
   
   @return			The contents of the file
   @throws java.io.IOException
   */
  // FIXME - can we check that the file is too long without losing the efficiency benefits?
  static public /*@ non_null */ String readFile(/*@ non_null */ String filename, /*@ non_null */ byte[] cb) 
  throws java.io.IOException {
    int i = 0;
    int j = 0;
    java.io.FileInputStream f = null;
    try {
      f = new java.io.FileInputStream(filename);
      while (j != -1) {
        i = i + j;
        j = f.read(cb,i,cb.length-i);
      }
    } finally {
      if (f != null) f.close();
    }
    return new String(cb,0,i);
  }
  
  /**
   * Reads a file, returning a String containing the contents
   * @param filename the file to be read
   * @return the contents of the file as a String, or null if the
   *      file could not be read
   */
  static public /*@nullable*/String readFileX(/*@ non_null */ String filename) {
    try {
      return readFile(filename);
    } catch (Exception e ) {
      System.out.println("Could not read file " + filename);
      // FIXME - need a better way to report and catch errors
    }
    return null;
  }
  
  /** Reads the contents of the file with the given name, returning a String. 
   This version presumes the file is shorter than an internal buffer.
   FIXME
   
   @param filename		The file to be read
   @return 				The contents of the file
   @throws IOException
   */
  static public /*@ non_null */ String readFile(/*@ non_null */ String filename) throws java.io.IOException {
    StringBuffer sb = new StringBuffer(10000);
    char[] cb = new char[10000];  // This hard-coded value can be anything;
                // smaller numbers will be less efficient since more reads
                // may result
    int j = 0;
    java.io.FileReader f = null;
    try {
      f = new java.io.FileReader(filename);
      while (j != -1) {
        j = f.read(cb,0,cb.length);
        if (j != -1) sb.append(cb,0,j);
      }
    } finally {
      if (f != null) f.close();
    }
    return sb.toString();
  }
  
  /**
   * Executes the static compile(String[]) method of the given class
   * @param cls	The class whose 'compile' method is be invoked
   * @param args   The String[] argument of the method
   * @return The standard output and error output of the invocation
   */
  //@ requires \nonnullelements(args);
  static public String executeCompile(/*@ non_null */ Class cls, 
				      /*@ non_null */ String[] args) 
    throws SecurityException, NoSuchMethodException
  {
    return executeMethod(cls,"compile",args);
  }
  
  /** Finds and executes the method with the given name in the given class;
   the method must have a single argument of type String[].  The 'args'
   parameter is supplied to it as its argument.
   * @param cls		The class whose method is to be invoked
   * @param methodname The method to be invoked
   * @param args		The argument of the method
   * @return  The standard output and error output of the invocation
   */
  //@ requires \nonnullelements(args);
  static private String executeMethod(/*@ non_null */ Class cls, 
				     /*@ non_null */ String methodname, 
				     /*@ non_null */ String[] args) 
    throws SecurityException, NoSuchMethodException 
  {
    try {
      Method method = cls.getMethod(methodname, new Class[] { String[].class });
      return executeMethod(method,args);
    } catch (NoSuchMethodException e) {
      System.err.println("No method compile in class " + cls);  // FIXME - better error return needed
      e.printStackTrace();
      throw e; // new RuntimeException(e.toString());
    }
  }
  
  /** Calls the given method on the given String[] argument.  Any standard
   output and error output is collected and returned as the String 
   return value.
   * @param method	The static method to be invoked
   * @param args	The argument of the method
   * @return		The standard output and error output of the method
   */
  //@ requires \nonnullelements(args);
  static private String executeMethod(/*@ non_null */ Method method, 
				     /*@ non_null */ String[] args) {
    try {
	ByteArrayOutputStream ba = setStreams();
	/*Object result =*/ method.invoke(null,new Object[]{args});
	// The following line might cause a cast error, but since b is not used it is comment out.
	// boolean b = ((Boolean)result).booleanValue();
	return ba.toString();
    } catch (Exception e) {
      Utils.restoreStreams(); // FIXME - see comment in TestFilesTestSuite.java
      // Need the above restore before we try to print something
      System.out.println(e.toString());  // FIXME - need better error handling
    } finally {
      Utils.restoreStreams();
    }
    return null;
  }
  
  /** The suffix to append to create the golden output filename */
  static private final String ORACLE_SUFFIX = "-expected";
  
  /** The suffix to append to create the actual output filename */
  static private final String SAVED_SUFFIX = "-ckd";
  
  /** Compares the given string to the content of the given file using
   a comparator that ignores platform differences in line-endings.
   The method has the side effect of saving the string value in a file
   for later comparison if the string and file are different, and making
   sure that the actual output file (the -ckd file) is deleted if the
   string and file are the same.  The expected output filename is the
   rootname + "-expected"; the actual output filename is the rootname + "-ckd". 
   *  
   * @param s the String to compare
   * @param rootname the prefix of the file name
   * @return The Diff structure that contains the comparison
   * @throws java.io.IOException
   */
  static public Diff compareStringToFile(/*@ non_null */ String s, String rootname) 
                                               throws java.io.IOException {
    String ss = Utils.readFile(rootname+ORACLE_SUFFIX);
    Diff df = new Diff( "expected", ss, "actual", s );
    
    if (!df.areDifferent()) {
      // If the two strings match, the test succeeds and we make sure
      // that there is no -ckd file to confuse anyone.
      (new File(rootname+SAVED_SUFFIX)).delete();
    } else {
      // If the strings do not match, we save the actual string and 
      // fail the test.
      FileWriter f = null;
      try {
        f = new FileWriter(rootname+SAVED_SUFFIX);
        f.write(s); 
      } finally {
        if (f != null) f.close(); 
      }
    }
    return df;
  }
  
  /** This deletes all files (in the current directory) whose
   names match the given pattern in a regular-expression sense;
   however, it is only implemented for patterns consisting of
   characters and at most one '*', since I'm not going to rewrite
   an RE library.
   * @param pattern the pattern to match against filenames
   */
  static public void removeFiles(/*@ non_null */ String pattern) {
    File[] list;
    int k = pattern.indexOf("*");
    if (k == -1) {
      list = new File[] { new File(pattern) };
    } else if (k == 0) {
      final String s = pattern.substring(1);
      FileFilter ff = new FileFilter() { 
        public boolean accept(File f) { return f.isFile() && f.getName().endsWith(s); }
      };
      list = (new File(".")).listFiles(ff);
    } else if (k+1 == pattern.length()) {
      final String s = pattern.substring(0,k);
      FileFilter ff = new FileFilter() { 
        public boolean accept(File f) { return f.isFile() && f.getName().startsWith(s); }
      };
      list = (new File(".")).listFiles(ff);        
    } else {
      final String s = pattern.substring(0,k);
      final String ss = pattern.substring(k+1);
      final int j = pattern.length()-1;
      FileFilter ff = new FileFilter() { 
        public boolean accept(File f) { 
          return  f.isFile() && 
          f.getName().length() >= j && 
          f.getName().startsWith(s) && 
          f.getName().endsWith(ss); }
      };
      list = (new File(".")).listFiles(ff);        
    }
    
    for (int i = 0; i<list.length; ++i) {
      //System.out.println("DELETING " +list[i].getName());
      list[i].delete();
    }
    
  }
}
