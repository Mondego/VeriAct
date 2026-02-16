/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.util;

import java.io.*;

/**
 * An output stream that splits into two.
 */
public class SplitPrintStream extends PrintStream
{
    private PrintStream os2;

    /**
     * Pass in the two print streams to be fed from
     * the same source.  Use the new stream, and it will
     * forward all output to both os1 and os2.
     */
    public SplitPrintStream(PrintStream os1, PrintStream os2) {
        super(os1);
        this.os2 = os2;
    }


    public void flush()  {
        super.flush();
        this.os2.flush();
    }

    public void write (int b) {
        super.write(b);
        os2.write(b);
    }

    public void write (byte[] b) throws IOException {
        super.write(b);
        os2.write(b);
    }

    public void write (byte[] b, int off, int len) {
        super.write(b, off, len);
        os2.write(b, off, len);
    }

}
