/* Copyright Hewlett-Packard, 2002 */

package escjava.prover;


import java.io.*;

import javafe.util.*;


/**
 ** Instances of this class represent subprocesses. <p>
 **
 ** SubProcess'es are created by invoking a named program with a given
 ** pathname.  Input can be submitted to the resulting subprocess;
 ** routines are provided for use in parsing the subprocesses'
 ** resulting output.<p>
 **
 ** SubProcesses can be closed; afterwards none of their methods
 ** (except close) should ever be called again.<p>
 **
 ** If any errors occur (including parsing errors), a fatal error will
 ** be reported. <p>
 **
 **
 ** As a debugging aid, if Info.on is true, we save all the characters
 ** read by clients from our subprocess since the last resetInfo()
 ** call.  In the event of a parsing error (see handleUnexpected()),
 ** we use this information, if available, to produce a more useful
 ** error message.
 **
 **
 ** Note that a Java application will not exit normally (i.e., when
 ** its <code>main</code> method returns) until all subprocesses have
 ** finished.  In particular, this implies that all
 ** <code>SubProcess</code> instances must be closed before this can
 ** occur.  Alternatively, a Java application can just call
 ** <code>java.lang.System.exit</code> to force an exit to occur
 ** without waiting for subprocesses.<p>
 **/

public class SubProcess {

    /***************************************************
     *                                                 *
     * Instance variables:			       *
     *                                                 *
     ***************************************************/

    /**
     ** The name of the subprocess, suitable for use in error messages.
     **/
    //@ invariant name!=null
    public final String name;


    /**
     ** The actual sub-process or null if we are closed.
     **/
    private Process P = null;

    /**
     ** The PrintStream to the actual sub-process or null if we are
     ** closed.
     **/
    //@ invariant (to==null) == (P==null)
    private PrintStream to = null;

    /**
     ** The PushbackInputStream from the actual sub-process or null if
     ** we are closed.
     **/
    //@ invariant (from==null) == (P==null)
    private PushbackInputStream from = null;


    /**
     ** If non-null, this buffer keeps a record of (some of) the characters
     ** read from our subprocess using getChar(). <p>
     **
     ** In the event of a parsing error (see handleUnexpected()),
     ** we use this information, if available, to produce a more useful
     ** error message.<p>
     **/
    private final StringBuffer readChars = (escjava.Main.trackReadChars ?
					    new StringBuffer() :
					    null);


    /***************************************************
     *                                                 *
     * Creation and destruction:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Create a new invocation of a sub-process. <p>
     **
     ** The resulting invocation is initially open, but may be closed
     ** permanently with the close method.<p>
     **
     ** name should be a unique short name for use in error messages
     ** (E.g., "Simplify"); path is the pathname of the program to
     ** invoke to obtain the subprocess.<p>
     **
     ** This constructor may result in a fatal error if any problems
     ** occur.<p>
     **/
    public SubProcess(/*@ non_null @*/ String name, /*@ non_null @*/ String path) {
	this.name = name;
	try {
	    P = Runtime.getRuntime().exec(path);
	    if (P==null)
		throw new IOException();
	} catch (IOException e) {
	    javafe.util.ErrorSet.fatal("Unable to invoke " + name);
	}

	OutputStream out = P.getOutputStream();
	Assert.notNull(out);   //@ nowarn Pre   //Unsure if this is always true

	if (escjava.Main.sxLog != null) {
	  try {
	    FileOutputStream fos = new FileOutputStream(escjava.Main.sxLog);
	    out = new TeeOutputStream(fos, out);
	  } catch (FileNotFoundException fnfe) {
	    javafe.util.ErrorSet.fatal("error opening sxLog file " +
				       escjava.Main.sxLog);
	  }
	}
	to = new PrintStream(out);
	from = new PushbackInputStream(P.getInputStream());
    }


    /**
     ** Close us. <p>
     **
     ** This destroys the associated subprocess.  Afterwards, none of
     ** our methods should ever be called again.<p>
     **
     ** This method may result in a fatal error if any problems
     ** occur.  Our subprocess is guaranteed to be destroyed afterwards,
     ** regardless of which exit is taken.<p>
     **/
    public void close() {
	try {
	    if (to!=null)
		to.close();
	    if (from!=null)
		from.close();	// may throw IOEXception
	} catch (IOException e) {
	    ErrorSet.fatal("The following I/O error occurred while "
			+ "trying to close the connection to " + name + ": "
			+ e.getMessage());
	} finally {
	    if (P!=null)
		P.destroy();
	    P = null;
	    to = null;
	    from = null;
	}
    }


    /***************************************************
     *                                                 *
     * Sending characters to a subprocess:             *
     *                                                 *
     ***************************************************/

    /**
     ** Send a String to our subprocess.
     **
     ** Precondition: we are not closed.<p>
     **/
    public void send(/*@ non_null @*/ String s) {
	Assert.notNull(P);     //@ nowarn Pre // precondition

	to.println(s);
	to.flush();
    }


    public PrintStream ToStream() {
	return to;
    }

    /*******************************************************
     *                                                     *
     * Reading characters from a subprocesses' output:     *
     *                                                     *
     *******************************************************/

    /**
     ** Return the next output character from our subprocess.  If an
     ** I/O error occurs or there are no more characters available
     ** (i.e., EOF is reached), a fatal error is reported.<p>
     **
     ** Saves the output character (if any) in <code>readChars</code> if
     ** it is non-null.  (This is a private debugging aid.)<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    public char getChar() {
	Assert.notNull(P);     //@ nowarn Pre // precondition

	try {
	    int next = from.read();
	    if (next<0)
		ErrorSet.fatal("Unexpected exit by " + name + " subprocess");
	    char c = (char)next;

	    if (readChars!=null)
		readChars.append(c);

	    return c;
	} catch (IOException e) {
	    handleReadError(e);
	    return 0;	             // Make the compiler happy...
	}
    }

    /**
     ** Like <code>getChar</code> but leaves the character in the
     ** stream.  If an I/O error occurs, a fatal error is reported.
     ** Returns -1 on EOF.<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    public int peekChar() {
	Assert.notNull(P);     //@ nowarn Pre // precondition

	try {
	    int next = from.read();
	    if (next!=-1)
		from.unread(next);
	    return next;
	} catch (IOException e) {
	    handleReadError(e);
	    return -1;			// dummy return
	}
    }


    /**
     ** Reset our memory of the recent output from our subprocess to
     ** remembering nothing. <p>
     ** 
     ** In the event of a parsing error (see handleUnexpected()), we
     ** use any remembered recent output to produce a more useful
     ** error message.<p>
     **/
    public void resetInfo() {
	if (readChars!=null)
	    readChars.setLength(0);
    }


    /**
     ** Turn an IOException from reading <code>from</code> into a fatal
     ** error.
     **/
    //@ ensures false
    private void handleReadError(/*@ non_null @*/ IOException e) {
	ErrorSet.fatal("I/O error encountered while reading "
		       + name + "'s output: " + e.getMessage());
    }


    /***************************************************
     *                                                 *
     * Verifying the subprocesses' output:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Report a fatal error due to unexpected output from the subprocess.<p>
     **
     ** <code>wanted</code> is the output we expected.<p>
     ** 
     ** Use this routine if at all possible, because it provides
     ** additional useful information when Info.on is true about the
     ** output read recently, what was expected, and the remaining
     ** output.<p>
     **/
    //@ ensures false
    public void handleUnexpected(String wanted) {
      
	if (readChars!=null && Info.on) {
	    Info.out("***** Unexpected output encountered while parsing "
		+ name + "'s output");
	    Info.out("Already read: ["+readChars.toString()+"]");
	    Info.out("Expected " + wanted);
	}
	if (readChars != null) {
	    if (P!=null) {
		to.close();
		while (peekChar()!=-1)
		    getChar();
	    }
	}

	String errOut = "";
	if (P != null) {
	  InputStream err = P.getErrorStream();
	  byte[] buf = new byte[1024];
	  try {
	    while (true) {
	      int r = err.read(buf);
	      if (r == -1) {
		break;
	      }
	      errOut += new String(buf, 0, 0, r);
	    }
	  } catch (IOException ioe) {
	    errOut += "<IOException: " + ioe.toString() + ">";
	  }
	}

	close();

	String suffix = null;
	if (readChars != null) {
	  suffix = ":" +
	           "\n----- Begin unexpected output -----\n" +
	           trimNewlines(readChars.toString()) +
	           "\n----- End unexpected output -----";
	}
	if (errOut.length() != 0) {
	  if (suffix == null) {
	    suffix = ":";
	  }
	  suffix += "\n----- Begin stderr of unexpected output -----\n" +
	            errOut +
	            "\n----- End stderr or unexpected output -----";
	}
	ErrorSet.fatal("Unexpected output encountered while parsing " +
		       name + "'s output" +
		       (suffix == null ? "" : suffix));
    }

    /** Returns <code>s</code> but with starting and ending newline characters
      * removed
      **/
    //@ ensures RES != null;
    static private String trimNewlines(/*@ non_null */ String s) {
      int start = 0;
      int end = s.length();
      while (start < end && s.charAt(start) == '\n') {
	start++;
      }
      while (start < end && s.charAt(end-1) == '\n') {
	end--;
      }
      return s.substring(start, end);
    }

    /**
     ** Consume the next output character from our subprocess.
     ** If it is not exactly <code>c</code>, a fatal error is 
     ** reported.<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    public void checkChar(char c) {
	if (c == getChar())
	    return;

	handleUnexpected("the character '" + c + "'");
    }

    /**
     ** Ensure that the next output characters from our subprocess
     ** (consumed in the process) match <code>s</code>.  If this does
     ** not occur, a fatal error is reported.<p>
     **
     ** Precondition: we are not closed.<p>
     **/
    public void checkString(/*@ non_null @*/ String s) {
	for (int i=0; i<s.length(); i++) {
	    checkChar(s.charAt(i));
	}
    }


    /***************************************************
     *                                                 *
     * General purpose parsing routines:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Consume any whitespace (spaces, tabs, and newlines) present in
     ** the subprocesses' output.
     **
     ** Precondition: we are not closed.<p>
     **/
    public void eatWhitespace() {
	for (int next=peekChar(); next==' ' || next=='\t' || next=='\n';
		next=peekChar())
	    getChar();
    }


    /**
     ** Read characters from our subprocess up to, but *not* including
     ** a character from the string <code>stops</code> or an EOF. <p>
     **
     ** The read characters are returned as a <code>String</code>.
     **
     ** Precondition: we are not closed.<p>
     **/
    //@ ensures \result != null;
    public String readWord(/*@ non_null @*/ String stops) {
	StringBuffer soFar = new StringBuffer();

	for (;;) {
	    int next = peekChar();

	    if (next>=0 && stops.indexOf((char)next) != -1)
		break;

	    soFar.append(getChar());
	}

	return soFar.toString();
    }

    /** Reads a (possibly empty) sequence of digits from the subprocess
      * and returns the digits as a number.  Assumes no overflow will
      * occur.
      **/
    //@ ensures 0 <= \result;
    public int readNumber() {
      int n = 0;
      while (Character.isDigit((char)peekChar())) {
	n = 10*n + getChar() - '0';
      }
      return n;
    }

    /***************************************************
     *                                                 *
     * Reading SExps and SLists:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Read a non-null SList from our subprocess.  A fatal error is
     ** reported if any errors occur. <p>
     **
     ** Note: see the warnings on readSExp()!
     **
     ** Precondition: we are not closed.<p>
     **/
    //@ ensures RES!=null
    public SList readSList() {
	eatWhitespace();
	checkChar('(');

	SList l = SList.make();
	for (;;) {
	    eatWhitespace();
	    if (peekChar()==')')
		break;

	    l = new SPair(readSExp(),l);
	}
	l = SList.reverseD(l);
	checkChar(')');
	return l;
    }


    /**
     ** Read a non-null SExp from our subprocess.  A fatal error is
     ** reported if any errors occur. <p>
     **
     ** We use the following grammar for Atoms and SInts:
     **
     **     SInt  ::= ['+'|'-']<digit><not one of "() \n\t">*
     **
     **     Atom ::= <not one of "() \n\t">+ modulo it's not a SInt as
     **              defined above
     **
     ** We do further processing as follows:
     **
     **   integers are parsed to ints using the java.lang.Integer class.
     **
     **   Atoms are parsed by removing the first and last characters
     **   iff the first character is a '|'.
     **
     **
     ** WARNINGS:
     **
     **   This means we do not handle correctly atoms containing the
     ** characters ' ', '\n', '\\', '(', ')', and '|' (the last inside
     ** the atom).  Likewise, we do not handle Atoms that start with
     ** '|' but do not end with a different '|'.  For speed reasons,
     ** this limitation may be left in the final version.<p>
     **
     **
     ** Precondition: we are not closed.<p>
     **/
    //@ ensures RES!=null
    public SExp readSExp() {
	eatWhitespace();
	if (peekChar()=='(')
	    return readSList();

	/*
	 * It's an Atom or integer, not a SList:
	 */
	String token = readWord("() \n\t");
	if (token.length()==0)
	    handleUnexpected("SExp");

	// Handle integers:
	char first = token.charAt(0);
	if (first=='+' || first=='-') {
	    if (token.length()>1)
		first = token.charAt(1);
	}
	if (Character.isDigit(first)) {
	    try {
		return SExp.make(new Integer(token).intValue());
	    } catch (NumberFormatException e) {
		handleUnexpected("valid integer");		
	    }
	}

	// Handle atoms:
	if (token.charAt(0)=='|')
	    token = token.substring(1, token.length()-1);

	return Atom.fromString(token);
    }



    /**
     ** Read a (possibly empty) series of SLists from our subprocess.  A
     ** fatal error is reported if any errors occur. <p>
     **
     ** I.e., "(a b) (c d (e) f)" returns the SExp ((a b) (c d (e) f)).
     ** This routine never returns null.<p>
     **
     ** Note: see the warnings on readSExp()!
     **
     ** Precondition: we are not closed.<p>
     **/
    //@ ensures RES!=null
    public SList readSLists() {
	SList l = SList.make();

	for (;;) {
	    eatWhitespace();
	    if (peekChar()!='(')
		break;

	    l = new SPair(readSList(),l);
	}
        l = SList.reverseD(l);
	return l;
    }


    /**
     ** Read a (possibly empty) series of SExps from our subprocess,
     ** until the stop character is seen (but not read).  A
     ** fatal error is reported if any errors occur. <p>
     **
     ** I.e., "e (a b) (c d (e) f)$(1 3)" where $ is the stop character
     ** returns the SExp (e (a b) (c d (e) f)), leaving the remaining
     ** unprocessed output "$(1 3)".  If the stop character
     ** occurs in the middle of atoms or SLists, it will not cause
     ** processing to stop.<p>
     **
     ** Note: see the warnings on readSExp()!
     **
     ** Precondition: we are not closed, the stop character is not
     ** whitespace.<p>
     **/
    //@ ensures RES!=null
    public SList readSExps(char stop) {
	SList l = SList.make();

	for (;;) {
	    eatWhitespace();
	    if (peekChar()==stop)
		break;

	    l = new SPair(readSExp(),l);
	}
	l = SList.reverseD(l);
	return l;
    }
}
