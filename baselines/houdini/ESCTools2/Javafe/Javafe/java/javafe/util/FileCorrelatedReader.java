/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import java.io.*; 

import javafe.genericfile.*;
import javafe.util.Assert;

/** 
 * A <code>FileCorrelatedReader</code> is a
 * <code>CorrelatedReader</code> that reads its characters from a
 * stream that corresponds to a file.
 *
 * <p> The class also provides a method to create a new
 * <code>CorrelatedReader</code> for the text between the marked
 * position and the current point in the stream.
 *
 * @see javafe.util.Location
 * @author Cormac Flanagan, Rustan Leino
 */

public class FileCorrelatedReader extends LocationManagerCorrelatedReader
{
    // Class variables & constants

    /** How big a block to read from a stream at a time */

    private static final int READBLOCKSIZE = 256;

    /** The initial size for buf */

    private static final int DEFAULTBUFSIZE = READBLOCKSIZE;

    // Instance variables

    /**
     * The stream for this CorrelatedReader if it is open and not a
     * subReader.  Null otherwise.
     *
     * (A CorrelatedReader is defined to be open iff buf is non-null.)
     */
    //@ invariant (buf != null) == (stream != null);

    //@ spec_public
    private InputStream stream;

    /**
     * The GenericFile for this CorrelatedReader.
     */

    private /*@ non_null @*/ GenericFile file;

    // Creation

    /**
     * Constructs a correlated input stream that reads its input from
     * the specified GenericFile.
     */
    public FileCorrelatedReader(/*@ non_null @*/ GenericFile file)
            throws java.io.IOException {
        this(file.getInputStream(), file);
    }

    /**
     * This is a specialized constructor used for InputStreams that
     * are not reopenable such as stdin.<p>
     *
     * streamName is the human readable name of the stream.
     */
    public FileCorrelatedReader(/*@ non_null @*/ InputStream in,
                                /*@ non_null @*/ String streamName) {
        this(in, new UnopenableFile(streamName));
    }

    /**
     * Constructs a correlated input stream that reads its input from
     * the specified input stream.
     *
     * file is an associated GenericFile that may be used to reopen
     * the specified input stream
     */

    private FileCorrelatedReader(/*@ non_null @*/ InputStream in,
                                 /*@ non_null @*/ GenericFile file) {
        super();  // allocate location numbers, etc.

        this.stream = in;
        this.file = file;

        this.buf = new byte[DEFAULTBUFSIZE];
        this.endBufNdx = 0;
	//System.out.println("NEW STREAM " + streamid + " " + file.getHumanName());
    }

    /**
     * Create a whole file location for a given GenericFile. <p>
     *
     * file need not be openable.
     */
    //@ ensures \result != Location.NULL;

    static int createWholeFileLoc(/*@ non_null @*/ GenericFile file) {
        // TBW:  This implementation leaves something to be desired.  It would be
        // nice if whole-file correlated readers could be their own subclass of
        // LocationManagerCorrelatedReader.
        try {
            /*
             * Create a CorrelatedReader for file, but substituting
             * a dummy InputStream instead of reading from file.
             * (file may not be openable.)
             */
            FileCorrelatedReader R =
                new FileCorrelatedReader(new ByteArrayInputStream(new byte[10]), file);

            // Then read the first character so it has a valid location:
            R.read();

            // Close it and turn it into a whole file CorrelatedReader:
            R.close();
            R.isWholeFile = true;

            // Finally, return its location:
            return R.getLocation();

        } catch (IOException e) {
            Assert.fail("I/O error reading from a ByteArrayInputStream");
            return Location.NULL;   // dummy return...
        }
    }

    /**
     * Closes us.  No other I/O operations may be called on us after
     * we have been closed.
     */
    //@ also modifies stream;

    public void close() {
        super.close();
        if (stream != null) {
            try {
                stream.close();
            } catch (IOException e) {
                // skip
            }
            stream = null;
        }
    }

    // The methods

    /** See spec in the abstract <code>CorrelatedReader</code> class.
     */

    public int read() throws IOException {
        if (buf == null) {
            Assert.precondition("read() called on a closed CorrelatedReader");
        }

        if (curNdx == endBufNdx) {
            // refill buffer
            if (!refillBuf()) {
                // EOF
                // save position of last character
                lastCharNdx = curNdx;
                return -1;
            }
        }

        // save the position of this character, before it is read:
        lastCharNdx = curNdx;

        int c = buf[curNdx++];

        if (c=='\\' && oddSlashLoc+1 != getLocation()) {
            // Can be a unicode escape sequence (is an odd slash!)
            if (peek() == 'u') {
                // is a unicode escape sequence
                while (peek() == 'u') {
                    curNdx++;
                }
                char s[] = new char[4];
                for (int i=0; i<4; i++) {
                    s[i] = (char)readRaw(); 
                    if (Character.digit(s[i],16)==-1) 
                        throw new IOException("Bad unicode character sequence");
                }
                c = Integer.parseInt( new String(s), 16 );
            } else {
                // not a unicode
                oddSlashLoc = getLocation();
            }
        } else if (c=='\n') {
            recordNewLineLocation();
            totalLinesRead++;
            curLineNo++;
        }

        return c;
    }

    /**
     * Refills the buffer. <p>
     *
     * If the mark is set, and there is less that READBLOCKSIZE space
     * left in the buffer, it allocates a larger buffer and copies
     * markNdx..curNdx from the old buffer into the new one. If the
     * mark is not set, then it clears the buffer.  It then tries to
     * read READBLOCKSIZE from the underlying input stream.<p>
     *
     * Returns true iff not end-of-file, and at least one character
     * was read from the file.  Throws an IOException if no
     * characters could be read from the stream.<p>
     *
     * Requires we are open.<p>
     */

    protected boolean refillBuf() throws IOException {
        // Make sure need more characters in buffer:
        Assert.notFalse( curNdx == endBufNdx );

        // Slide buf down, keep from min( lastCharNdx, markNdx )
        int minNdx = marked ? Math.min( lastCharNdx, markNdx ) : lastCharNdx;

        if (buf.length - (curNdx - minNdx) < READBLOCKSIZE) {
            // Allocate a new buffer
            byte[] newBuf = new byte[ curNdx - minNdx + READBLOCKSIZE ];
            System.arraycopy( buf, minNdx, newBuf, 0, curNdx - minNdx );
            buf = newBuf;
        } else {
            System.arraycopy( buf, minNdx, buf, 0, curNdx-minNdx );
        }
        curNdx       -= minNdx;
        markNdx      -= minNdx;       //@ nowarn Unreadable;
        lastCharNdx  -= minNdx;
        beforeBufLoc += minNdx;
        endBufNdx    = curNdx;

        Assert.notFalse( endBufNdx+READBLOCKSIZE <= buf.length );

        int got = 0;
        do {
            got = stream.read( buf, endBufNdx, READBLOCKSIZE );
        } while (got == 0 );

        if (got == -1) {
            return false;
        } else {
            Assert.notFalse( curNdx == endBufNdx );
            endBufNdx += got;
            Assert.notFalse( endBufNdx <= buf.length );
            Assert.notFalse( curNdx < endBufNdx );
            return true;
        }
    }

    // Misc

    /**
     * Returns the file underlying this correlated reader.
     */

    public /*@non_null*/GenericFile getFile() {
        return file;
    }
}
