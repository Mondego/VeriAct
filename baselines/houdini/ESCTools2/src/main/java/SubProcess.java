/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

import java.io.*;

import javafe.util.*;

/**
 * Instances of this class represent subprocesses.
 * 
 * <p> <code>SubProcess</code>es are created by invoking a named
 * program with a given pathname.  Input can be submitted to the
 * resulting subprocess; routines are provided for use in parsing the
 * subprocesses' resulting output. </p>
 * 
 * <p> <code>SubProcess</code>es can be closed; afterwards none of
 * their methods (except close) should ever be called again. </p>
 *
 * <p> If any errors occur (including parsing errors), a fatal error
 * will be reported. </p>
 *
 * <p> As a debugging aid, if {@link Info#on} is true, we save all the
 * characters read by clients from this subprocess since the last
 * {@link #resetInfo()} call.  In the event of a parsing error (see
 * {@link #handleUnexpected(String)}), we use this information, if
 * available, to produce a more useful error message. </p>
 *
 * <p> Note that a Java application will not exit normally (i.e., when
 * its <code>main()</code> method returns) until all subprocesses have
 * finished.  In particular, this implies that all
 * <code>SubProcess</code> instances must be closed before this can
 * occur.  Alternatively, a Java application can just call {@link
 * java.lang.System#exit(int)} to force an exit to occur without
 * waiting for subprocesses. </p>
 *
 * @todo Introduce a model variable notClosed that relates to P for
 * clearer preconditions.
 */

public class SubProcess {
    static public class Died extends RuntimeException {
      private static final long serialVersionUID = 2624260378471410994L;
    }

    final static public /*@ non_null @*/ Died DIED = new Died();

    /**
     * The name of the subprocess, suitable for use in error messages.
     */
    public final /*@ non_null @*/ String name;

    /**
     * The actual subprocess, or <code>null</code> if we are closed.
     */
    //@ spec_public
    private /*@nullable*/ Process P = null;

    //@ public model boolean closed;
    //@ private represents closed <- P == null;

    /**
     * The {@link PrintStream} to the actual subprocess, or
     * <code>null</code> if we are closed.
     */
    //@ invariant (to == null) <==> (P == null);
    //@ invariant (to == null) ==> closed;
    //@ spec_public
    private /*@nullable*/ PrintStream to = null;

    /**
     * The {@link PushbackInputStream} from the actual subprocess, or
     * <code>null</code> if we are closed.
     */
    //@ invariant (from == null) <==> (P == null);
    //@ spec_public
    private /*@nullable*/ PushbackInputStream from = null;

    /**
     * If non-<code>null</code>, this buffer keeps a record of (some
     * of) the characters read from this subprocess using {@link
     * #getChar()}.
     *
     * <p> In the event of a parsing error (see {@link
     * #handleUnexpected(String)}), we use this information, if
     * available, to produce a more useful error message. </p>
     */
    //@ spec_public
    private final StringBuffer readChars = (escjava.Main.options().trackReadChars ?
                                            new StringBuffer() :
                                            null);

    /**
     * Instantiate a subprocess.
     *
     * <p> The resulting invocation is initially open, but may be closed
     * permanently with the {@link #close()} method. </p>
     *
     * <p> This constructor may result in a fatal error if any
     * problems occur. </p>
     *
     * @param name should be a unique short name for use in error
     * messages (E.g., "Simplify").
     * @param pathAndArgs is an array containing the full pathname of the program to execute to
     * obtain the subprocess (e.g., "/usr/bin/emacs") and any command-line arguments.
     */
    /*@ public normal_behavior
      @   assignable this.name, to, from, P;
      @   ensures this.name == name;
      @   ensures to != null;
      @   ensures from != null;
      @   ensures !closed;
      @*/
    public SubProcess(/*@ non_null @*/ String name, 
                      /*@ non_null @*/ String[] pathAndArgs,
		      String[] envp) {
	this.name = name;
	try {
	    P = Runtime.getRuntime().exec(pathAndArgs,envp);
	    if (P == null)
		throw new IOException();
	} catch (IOException e) {
	    javafe.util.ErrorSet.fatal("Unable to invoke " + name);
	}

	OutputStream out = P.getOutputStream();
	Assert.notNull(out);   //@ nowarn Pre;   //Unsure if this is always true

	if (escjava.Main.options().sxLog != null) {
            try {
                OutputStream fos = new FileOutputStream(escjava.Main.options().sxLog);
		if (escjava.Main.options().prettyPrintVC)
			fos = new PPOutputStream(fos);

                out = new TeeOutputStream(fos, out);
            } catch (FileNotFoundException fnfe) {
                javafe.util.ErrorSet.fatal("error opening sxLog file " +
                                           escjava.Main.options().sxLog);
            }
	}
	to = new PrintStream(out);
	from = new PushbackInputStream(P.getInputStream());
    }


    /**
     * Close this subprocess.
     *
     * <p> This destroys the associated subprocess.  Afterwards, none
     * of the methods of this object should ever be called again. </p>
     *
     * <p> This method may result in a fatal error if any problems
     * occur.  This subprocess is guaranteed to be destroyed on
     * completion, regardless of which exit is taken. </p>
     */
    /*@ public normal_behavior
      @   modifies from, to, P;
      @   ensures P == null;
      @   ensures to == null;
      @   ensures from == null;
      @   ensures closed;
      @*/
    public void close() {
	try {
	    if (to != null)
		to.close();
	    if (from != null)
		from.close();	// may throw IOException
	} catch (IOException e) {
	    ErrorSet.fatal("The following I/O error occurred while "
                           + "trying to close the connection to " + name + ": "
                           + e.getMessage());
	} finally {
	    if (P != null)
		P.destroy();
	    P = null;
	    to = null;
	    from = null;
	}
    }

    // Sending characters to a subprocess.

    /**
     * Send a string to this subprocess.
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies to;
    public void send(/*@ non_null @*/ String s) {
	Assert.notNull(P);     //@ nowarn Pre; // precondition
	to.println(s);
	to.flush();
    }

    //@ public normal_behavior
    //@   requires !closed;
    //@   ensures (\result == null) <==> (P == null);
    public /*@ pure @*/ PrintStream ToStream() {
	return to;
    }

    // Reading characters from a subprocess's output.

    /**
     * @return the next output character from this subprocess.  If an
     * I/O error occurs or there are no more characters available
     * (i.e., EOF is reached), a fatal error is reported and a -1 is
     * returned.
     *
     * <p> Saves the output character (if any) in {@link #readChars} if
     * it is non-null.  (This is a private debugging aid.) </p>
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies readChars.*;
    //@   ensures \result >= 0;
    //@ also
    //@ public exceptional_behavior
    //@   requires P == null;
    //@   modifies \everything;
    //@   signals_only Died;
    //@   signals (Died) true;
    public int getChar() {
	if (P == null) throw new Died();

	try {
	    int next = from.read();
	    if (next < 0)
		ErrorSet.fatal("Unexpected exit by " + name + " subprocess");
	    char c = (char)next;

	    if (readChars != null)
		readChars.append(c);

	    return c;
	} catch (IOException e) {
	    handleReadError(e);
	    return -0;	             // Make the compiler happy...
	}
    }

    /**
     * Like {@link #getChar()}, but leaves the character in the stream.
     * If an I/O error occurs, a fatal error is reported.
     *
     * @return A -1 is returned on EOF, otherwise the next character
     * read from the subprocess is returned as an integer.
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    //@   ensures \result >= -1;
    //@ also
    //@ public exceptional_behavior
    //@   requires P == null;
    //@   modifies \everything;
    //@   signals_only Died;
    //@   signals (Died) true;
    public int peekChar() {
	// P may have been closed on us 
	if (P == null) throw new Died();

	try {
	    int next = from.read();
	    if (next != -1)
		from.unread(next);
	    return next;
	} catch (IOException e) {
	    handleReadError(e);
	    return -1;			// dummy return
	}
    }

    /**
     * Reset the memory of the recent output from this subprocess.
     * 
     * <p> In the event of a parsing error (see {@link
     * #handleUnexpected(String)}), we use any remembered recent
     * output to produce a more useful error message. </p>
     */
    //@ public normal_behavior
    //@   requires !closed;
    //@   modifies readChars.*;
    //@   ensures (readChars != null) ==> readChars.length() == 0;
    public void resetInfo() {
	if (readChars != null)
	    readChars.setLength(0);
    }

    /**
     * Turn an {@link IOException} resulting from a read on {@link
     * #from} into a fatal error.
     */
    /*@ private behavior
      @   requires true;
      @   diverges true;
      @   assignable \everything;
      @   ensures false;
      @   signals(Exception) false;
      @*/
    private void handleReadError(/*@ non_null @*/ IOException e) {
	ErrorSet.fatal("I/O error encountered while reading "
		       + name + "'s output: " + e.getMessage());
    }


    // Verifying the subprocesses' output.

    /**
     * Report a fatal error due to unexpected output from the subprocess.
     *
     * <p> Use this routine if at all possible, because it provides
     * additional useful information (when {@link javafe.util.Info#on}
     * is true) about the output read recently, what was expected, and
     * the remaining output. </p>
     *
     * @param wanted is the output we expected.
     */
    //@ behavior
    //@   modifies \everything;
    //@   ensures closed;
    public void handleUnexpected(/*@ non_null @*/ String wanted) {
	if (readChars != null && Info.on) {
	    Info.out("***** Unexpected output encountered while parsing "
                     + name + "'s output");
	    Info.out("Already read: [" + readChars.toString() + "]");
	    Info.out("Expected " + wanted);
	}
	if (readChars != null) {
	    if (P != null) {
		to.close();
		while (peekChar() != -1)
		    getChar();
	    }
	}

	StringBuffer errOut = new StringBuffer();
	if (P != null) {
            InputStream err = P.getErrorStream();
		// FIXME - change this to read characters?
            byte[] buf = new byte[1024];
            try {
                while (true) {
                    int r = err.read(buf);
                    if (r == -1) {
                        break;
                    }
                    errOut.append(new String(buf, 0, r));
                }
            } catch (IOException ioe) {
                errOut.append("<IOException: ");
		errOut.append(ioe.toString());
		errOut.append(">");
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

    /**
     * @param s the string to trim.
     * @return the parameter <code>s</code>, but with starting and
     * ending newline characters removed
     */
    static private /*@ pure non_null @*/ String trimNewlines(/*@ non_null @*/ String s) {
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
     * Consume the next output character from this subprocess.  If it
     * is not exactly <code>c</code>, a fatal error is reported.
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    //@ also
    //@ public behavior
    //@   modifies \everything;
    //@   ensures closed;
    public void checkChar(char c) {
	if (c == getChar())
	    return;

	handleUnexpected("the character '" + c + "'");
    }

    /**
     * Ensure that the next output characters from this subprocess
     * (consumed in the process) matches the provided string.  If this
     * does not occur, a fatal error is reported.
     *
     * @param s the string to match.
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    //@ also
    //@ public behavior
    //@   modifies \everything;
    //@   ensures closed;
    public void checkString(/*@ non_null @*/ String s) {
	for (int i = 0; i < s.length(); i++) {
	    checkChar(s.charAt(i));
	}
    }

    // General purpose parsing routines.

    /**
     * Consume any whitespace (spaces, tabs, and newlines) present in
     * the subprocesses' output.
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    public void eatWhitespace() {
	for (int next = peekChar();
             next == ' ' || next == '\t' || next == '\n';
             next = peekChar())
	    getChar();
    }

    /**
     * Read characters from this subprocess up to, but <em>not</em>
     * including a character from the provided string
     * <code>stops</code>, or an EOF.
     *
     * @param stops a string containing all characters on which to
     * stop reading.
     * @return the read characters, not including any character from
     * <code>stops</code>.
     */
    /*@ public normal_behavior
      @   requires P != null;
      @   requires !closed;
      @   modifies \everything;
      @   ensures (\forall int i; 0 <= i && i <= \result.length();
      @                    stops.indexOf(\result.charAt(i)) == -1);
      @   ensures \result != null;
      @*/
    public /*@ non_null @*/ String readWord(/*@ non_null @*/ String stops) {
	StringBuffer soFar = new StringBuffer();

        while (true) {
	    int next = peekChar();

	    if (next >= 0 && stops.indexOf((char)next) != -1)
		break;

	    soFar.append((char)getChar());
	}

	return soFar.toString();
    }

    /**
     * Reads a (possibly empty) sequence of digits from the subprocess
     * and returns the digits as a number.  Assumes no overflow will
     * occur.
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    //@   ensures 0 <= \result;
    public int readNumber() {
        int n = 0;
        while (Character.isDigit((char)peekChar())) {
            n = 10*n + getChar() - '0';
        }
        return n;
    }


    // Reading SExps and SLists

    /**
     * @return a non-null {@link SList} from this subprocess.  A fatal
     * error is reported if any errors occur.
     *
     * @note See the warnings on {@link #readSExp()}!
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    public /*@ non_null @*/ SList readSList() {
	eatWhitespace();
	checkChar('(');

	SList l = SList.make();
        while (true) {
	    eatWhitespace();
	    if (peekChar()==')')
		break;
	    l = new SPair(readSExp(), l);
	}
	l = SList.reverseD(l);
	checkChar(')');
	return l;
    }

    /**
     * @return a non-null {@link SExp} read from this subprocess.  A
     * fatal error is reported if any errors occur.
     *
     * <p> We use the following grammar for {@link Atom}s and {@link
     * SInt}s:
     * <pre>
     *     SInt  ::= ['+'|'-']<digit><not one of "() \n\t">*
     * 
     *     Atom ::= <not one of "() \n\t">+ modulo it's not a SInt as
     *              defined above.
     * </pre>
     *
     * <p> We do further processing as follows:
     * <ul>
     * <li> integers are parsed to ints using the {@link
     * java.lang.Integer} class. </li>
     * <li> {@link Atom}s are parsed by removing the first and last
     * characters iff the first character is a '|'. </li>
     * </ul>
     *
     * @warning This means we do not handle correctly atoms containing
     * the characters ' ', '\n', '\\', '(', ')', and '|' (the last
     * inside the atom).  Likewise, we do not handle {@link Atom}s
     * that start with '|' but do not end with a different '|'.  For
     * speed reasons, this limitation may be left in the final
     * version.
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    public /*@ non_null @*/ SExp readSExp() {
	eatWhitespace();
	if (peekChar() == '(')
	    return readSList();

	// It's an Atom or integer, not a SList.
	String token = readWord("() \n\t");
	if (token.length() == 0)
	    handleUnexpected("SExp");

	// Handle integers.
	char first = token.charAt(0);
	if (first == '+' || first == '-') {
	    if (token.length() > 1)
		first = token.charAt(1);
	}
	if (Character.isDigit(first)) {
	    try {
		return SExp.make(new Integer(token).intValue());
	    } catch (NumberFormatException e) {
		handleUnexpected("valid integer");		
	    }
	}

	// Handle atoms.
	if (token.charAt(0) == '|')
	    token = token.substring(1, token.length() - 1);

	return Atom.fromString(token);
    }

    /**
     * Read a (possibly empty) series of {@link SList}s from this
     * subprocess.  A fatal error is reported if any errors occur.
     *
     * <p> I.e., "(a b) (c d (e) f)" returns the SExp ((a b) (c d (e)
     * f)).  This routine never returns <code>null</code>.
     *
     * @note See the warnings on {@link #readSExp()}!
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   modifies \everything;
    public /*@ non_null @*/ SList readSLists() {
	SList l = SList.make();

        while (true) {
	    eatWhitespace();
	    if (peekChar() != '(')
		break;
	    l = new SPair(readSList(), l);
	}
        l = SList.reverseD(l);
	return l;
    }

    /**
     * @param stop the character that causes reading to stop.
     * @return the slist read from this subprocess.
     *
     * <p> Read a (possibly empty) series of {@link SExp}s from this
     * subprocess, until the supplied stop character is seen (but not
     * read).  A fatal error is reported if any errors occur. </p>
     *
     * <p> I.e., "e (a b) (c d (e) f)$(1 3)" where $ is the stop
     * character returns the {@link SExp} (e (a b) (c d (e) f)),
     * leaving the remaining unprocessed output "$(1 3)".  If the stop
     * character occurs in the middle of atoms or {@link SList}s, it
     * will not cause processing to stop. </p>
     *
     * @note See the warnings on {@link #readSExp()}!
     */
    //@ public normal_behavior
    //@   requires P != null;
    //@   requires !closed;
    //@   requires stop != ' ' && stop != '\t' && stop != '\n';
    //@   modifies \everything;
    public /*@ non_null @*/ SList readSExps(char stop) {
	SList l = SList.make();

        while (true) {
	    eatWhitespace();
	    if (peekChar() == stop)
		break;

	    l = new SPair(readSExp(), l);
	}
	l = SList.reverseD(l);
	return l;
    }
}
