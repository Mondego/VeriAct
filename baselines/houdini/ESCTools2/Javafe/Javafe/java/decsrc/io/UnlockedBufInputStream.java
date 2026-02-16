package decsrc.io;
import java.io.*;

/** Buffered input stream without synchronization, and use of a user
    supplied buf. */
public class UnlockedBufInputStream extends InputStream {
    private byte[] buf;
    private InputStream source;
    private int pos;
    private int count;

    /** Construct an unsynchronized input stream that reads from "s"
	and uses "b" as a temporary buffer for read data. */
    public UnlockedBufInputStream(InputStream s, byte[] b) {
	source = s;
	buf = b;
	pos = 0;
	count = 0;
	// xxx set isOpen = true; //prj 15may2006 isOpen not defined
    }

    //@ also private behavior
    //@ assignable pos;
    public int read() throws IOException {
	if (pos >= count) {
	    if (!fill()) {
		return -1;
	    }
	}
	return buf[pos++] & 0xff;
    }

    //@ also private behavior
    //@ assignable pos;
    public int read(byte[] b, int off, int len) throws IOException {
	if (pos >= count) {
	    if (!fill()) {
		return -1;
	    }
	}
	int n = count - pos;
	int ret = (n < len) ? n : len;
	System.arraycopy(buf, pos, b, off, ret);
	pos += ret;
	return ret;
    }

    //@ also private behavior
    //@ assignable count, pos;
    public long skip(long n) throws IOException {
	if (n > 0) {
	    long buffered = count - pos;
	    if (n >= buffered) {
		pos = 0;
		count = 0;
		return buffered + source.skip(n - buffered);
	    } else {
		pos += n;
		return n;
	    }
	} else {
	    return 0;
	}
    }

    public int available() throws IOException {
	return source.available() + (count - pos);
    }

    public void close() throws IOException {
	source.close();
    }

    private boolean fill() throws IOException {
	pos = 0;
	int n = source.read(buf, 0, buf.length);
	if (n > 0) {
	    count = n;
	    return true;
	} else {
	    count = 0;
	    return false;
	}
    }
}
