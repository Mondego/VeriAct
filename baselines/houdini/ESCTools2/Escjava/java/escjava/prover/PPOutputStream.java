/* $Id$ */
/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

import java.io.OutputStream;
import java.io.FilterOutputStream;
import java.io.IOException;


/**
  * This class is a FilterOutputStream class designed for LISP-like
  * input; it pretty prints the output by inserting spaces and newlines
  * into the stream.
  *
  * @author  David Cok
  */

public class PPOutputStream extends FilterOutputStream {

    private int parenDepth = 0;
    private boolean inComment = false;
    private boolean recentNL = false;

    static final int lp = '(';
    static final int rp = ')';

    /**
     * Creates an output stream filter built on top of an
     * underlying output streams.
     *
     * @param   out   the underlying output stream
     */
    public PPOutputStream(/*@ non_null */ OutputStream out) {
        super(out);
    }

    /**
     * Writes the specified <code>byte</code> to this output stream. 
     * <p>
     * Implements the abstract <tt>write</tt> method of <tt>OutputStream</tt>. 
     *
     * @param      b   the <code>byte</code>.
     * @exception  IOException  if an I/O error occurs.
     */
    //@ also private normal_behavior
    //@   assignable inComment, parenDepth, recentNL;
    public void write(int b) throws IOException {
	if (inComment) {
		super.write(b);
		if (b == '\n' || b == '\r') {
			inComment = false;
		} else {
			return;
		}
	}
	if (b == ';') {
		inComment = true;
		super.write(b);
		return;
	}
	if (b == lp) {
	    if (!recentNL) {
		super.write('\n'); //FIXME - multiplatform ???
		int n = 2*parenDepth;
		while (--n >= 0) super.write(' ');
	    }
            ++parenDepth;
        }

	if (b == rp) {
	    super.write(b);
            super.write('\n'); //FIXME - multiplatform ???
            --parenDepth;
            int n = 2*parenDepth;
            while (--n >= 0) super.write(' ');
	    recentNL = true;
	} else if (b == ' ' || b == '\t') {
	    if (!recentNL) super.write(b);
	} else if (b == '\n' || b == '\r') {
	    if (!recentNL) {
		super.write(b);
		int n = 2*parenDepth;
		while (--n >= 0) super.write(' ');
		if (parenDepth > 0) recentNL = true;
	    }
	} else {
	    super.write(b);
	    recentNL = false;
        }

    }

}
