package decsrc.io;

import java.io.Writer;
import decsrc.container.CharVec;

/** A "Writer" that sends its output to a "CharVec".  This writer is
    <strong>not</strong> synchronized.  Therefore callers should arrange
    that multiple threads do not invoke methods in this writer
    concurrently.  This class never throws IO exceptions. */
public class UnlockedCharVecWriter extends Writer {
    private CharVec buffer;

    public UnlockedCharVecWriter(CharVec v) {
	buffer = v;
    }

    public void write(int c) {
	buffer.append((char) c);
    }

    public void write(char[] buf, int start, int num) {
	buffer.appendArray(buf, start, num);
    }

    public void write(String s) {
	buffer.appendString(s);
    }

    public void flush() {
    }

    public void close() {
	// What should we do?  The "Writer" spec claims that further
	// write/flush calls should cause an IOException, but it
	// does not make sense for a writer into a character-vector
	// to generate such exceptions.
    }
}
