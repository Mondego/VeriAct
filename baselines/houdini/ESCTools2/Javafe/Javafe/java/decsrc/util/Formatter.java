package decsrc.util;
import java.text.NumberFormat;
import java.io.Writer;
import java.io.IOException;

/** Class for formatting various data types to a Writer.
    <ul>
    <li> A Formatter provides limited printf-style formatting.
    <li> A Formatter does not perform its own synchronization.
         Therefore clients are required to arrange that
         only a single thread uses the same "Formatter" object
	 at a time.
    </ul>

    Extensions from C version:
    <ul>
    <li> <code>%b</code> prints numbers in binary.
    <li> <code>%p</code> prints the result of <code>arg.toString()</code>
    </ul>

    Limitations compared to C version:
    <ul>
    <li> Upto seven object arguments can be passed directly.  An
         array can be passed if more data has to be printed.
    <li> A single scalar argument can be supplied to printf
         directly.  If multiple arguments have to be printed,
	 all of the scalar arguments have to be turned into
	 the corresponding object type before being printed.
    <li> h/l size specifiers are ignored.  The actual type of the
         argument is used instead.
    <li> %e/%g are handled just like %f
    <li> positional parameters are not handled
    <li> indirect field width specification is not handled
    <li> repeated use of the same format specifier for multiple
         arguments is not handled.
    </ul>
*/
public class Formatter {
    // The following objects are used to record the type of
    // the argument when a scalar argument is supplied.
    private static Byte		byte_type	= new Byte((byte) 0);
    private static Character	char_type	= new Character((char) 0);
    private static Short	short_type	= new Short((short) 0);
    private static Integer	int_type	= new Integer((int) 0);
    private static Long		long_type	= new Long((long) 0);
    private static Float	float_type	= new Float(0.0);
    private static Double	double_type	= new Double(0.0);

    private Writer w;
    private Object[] vlist;
    private char[] tmp;
    private NumberFormat nf;
    private boolean had_exc;

    // If "scalar" is true, the singleton argument is stored in
    // either "long_value" or "double_value" depending on the
    // type recorded in the argument list.
    private boolean	scalar;
    private long	long_value;
    private double	double_value;

    private boolean	autoflush;

    /** Create a new auto-flushing formatter that sends its output
	to <code>w</code>. */
    public Formatter(Writer w) {
	this(w, true);
    }

    /** Create a new formatter that sends its output to <code>w</code>.
	The output is automatically flushed iff <code>autoflush</code>
	is true. */
    public Formatter(Writer w, boolean autoflush) {
	this.w = w;
	this.vlist = new Object[7];
	this.tmp = new char[65];
	this.nf = null;
	this.had_exc = false;
	this.autoflush = autoflush;
    }

    /** Returns true iff output is automatically flushed. */
    public boolean autoflush() {
	return this.autoflush;
    }

    /** Change the auto-flush mode so that output is automatically
	flushed iff <code>autoflush</code> is true. */
    public void set_autoflush(boolean autoflush) {
	this.autoflush = autoflush;
    }

    public void flush() {
	try {
	    w.flush();
	} catch (IOException x) {
	    record_exc();
	}
    }

    public void print(String s) {
	try {
	    w.write(s);
	    if (autoflush) w.flush();
	} catch (IOException x) {
	    record_exc();
	}
    }

    public void print(int c) {
	try {
	    w.write(c);
	    if (autoflush) w.flush();
	} catch (IOException x) {
	    record_exc();
	}
    }

    public void print(char[] buf) {
	print(buf, 0, buf.length);
    }

    public void print(char[] buf, int start, int length) {
	try {
	    w.write(buf, start, length);
	    if (autoflush) w.flush();
	} catch (IOException x) {
	    record_exc();
	}
    }

    // Some simple printf functions with a single argument
    public void printf(String fmt, byte v) {
	long_printf(fmt, (long) v, byte_type);
    }

    public void printf(String fmt, char v) {
	long_printf(fmt, (long) v, char_type);
    }

    public void printf(String fmt, short v) {
	long_printf(fmt, (long) v, short_type);
    }

    public void printf(String fmt, int v) {
	long_printf(fmt, (long) v, int_type);
    }

    public void printf(String fmt, long v) {
	long_printf(fmt, (long) v, long_type);
    }
    
    public void printf(String fmt, float v) {
	double_printf(fmt, (double) v, float_type);
    }

    public void printf(String fmt, double v) {
	double_printf(fmt, v, double_type);
    }

    private void long_printf(String fmt, long v, Object type) {
	try {
	    long_value = v;
	    scalar = true;
	    vlist[0] = type;
	    process(fmt, vlist, 0, 1);
	} finally {
	    scalar = false;
	}
    }
    
    private void double_printf(String fmt, double v, Object type) {
	try {
	    double_value = v;
	    scalar = true;
	    vlist[0] = type;
	    process(fmt, vlist, 0, 1);
	} finally {
	    scalar = false;
	}
    }

    // printf functions that take "Object" arguments
    public void printf(String fmt) {
	plist(fmt, 0, null, null, null, null, null, null, null);
    }

    public void printf(String fmt, Object x1) {
	plist(fmt, 1, x1, null, null, null, null, null, null);
    }

    public void printf(String fmt, Object x1, Object x2) {
	plist(fmt, 2, x1, x2, null, null, null, null, null);
    }

    public void printf(String fmt, Object x1, Object x2, Object x3) {
	plist(fmt, 3, x1, x2, x3, null, null, null, null);
    }

    public void printf(String fmt, Object x1, Object x2, Object x3,
		       Object x4) {
	plist(fmt, 4, x1, x2, x3, x4, null, null, null);
    }

    public void printf(String fmt, Object x1, Object x2, Object x3,
		       Object x4, Object x5) {
	plist(fmt, 5, x1, x2, x3, x4, x5, null, null);
    }

    public void printf(String fmt, Object x1, Object x2, Object x3,
		       Object x4, Object x5, Object x6) {
	plist(fmt, 6, x1, x2, x3, x4, x5, x6, null);
    }

    public void printf(String fmt, Object x1, Object x2, Object x3,
		       Object x4, Object x5, Object x6, Object x7) {
	plist(fmt, 7, x1, x2, x3, x4, x5, x6, x7);
    }

    public void printf_list(String fmt, Object[] list) {
	process(fmt, list, 0, list.length);
    }

    private void plist(String fmt, int n,
		       Object x1, Object x2, Object x3, Object x4,
		       Object x5, Object x6, Object x7) {
	vlist[0] = x1;
	vlist[1] = x2;
	vlist[2] = x3;
	vlist[3] = x4;
	vlist[4] = x5;
	vlist[5] = x6;
	vlist[6] = x7;
	process(fmt, vlist, 0, n);
    }

    private void process(String fmt, Object[] list, int start, int limit) {
	try {
	    int value_index = start;
	    int n = fmt.length();
	    int i = 0;
	    while (i < n) {
		char c = fmt.charAt(i++);
		if (c != '%') {
		    w.write(c);
		    continue;
		}
		
		// "%%"?
		if (fmt.charAt(i) == '%') {
		    w.write('%');
		    i++;
		    continue;
		}

		// Left-justify?
		boolean lpad = true;
		if (fmt.charAt(i) == '-') {
		    lpad = false;
		    i++;
		}
		
		// What character should be used for padding?
		char padding = ' ';
		boolean zero_pad = (fmt.charAt(i) == '0');
		
		// Read width
		int width = 0;
		while (Character.isDigit((c = fmt.charAt(i)))) {
		    int digit = Character.digit(c, 10);
		    width = width * 10 + digit;
		    i++;
		}
		
		// Read precision
		int precision = -1;
		if (c == '.') {
		    i++;
		    precision = 0;
		    while (Character.isDigit((c = fmt.charAt(i)))) {
			int digit = Character.digit(c, 10);
			precision = precision * 10 + digit;
			i++;
		    }
		}
		
		// Unparse the type into either a "String", or a char array
		String str = null;
		char[] buf = null;
		int nchars = 0;

		// Get the object
		if (value_index >= limit) {
		    format_error("too few arguments for format string");
		}
		Object x = list[value_index++];
		
		char t = fmt.charAt(i++);
		if ((t == 'h') || (t == 'l')) {
		    // Ignore the size specification
		    t = fmt.charAt(i++);
		}
		if (t == 'c') {
		    // Check argument type and then get the value
		    char cval = ((Character) x).charValue();
		    if (scalar) cval = (char) long_value;
		    buf = tmp;
		    buf[0] = cval;
		    nchars = 1;
		} else if (t == 's') {
		    str = (String) x;
		} else if (t == 'd') {
		    buf = tmp;
		    nchars = unparse_decimal(buf, x, precision);
		    if (zero_pad) padding = '0';
		} else if (t == 'x') {
		    buf = tmp;
		    nchars = unparse_bits(buf, x, 16, 4, false, precision);
		    if (zero_pad) padding = '0';
		} else if (t == 'X') {
		    buf = tmp;
		    nchars = unparse_bits(buf, x, 16, 4, true, precision);
		    if (zero_pad) padding = '0';
		} else if (t == 'o') {
		    buf = tmp;
		    nchars = unparse_bits(buf, x, 8, 3, false, precision);
		    if (zero_pad) padding = '0';
		} else if (t == 'b') {
		    buf = tmp;
		    nchars = unparse_bits(buf, x, 2, 1, false, precision);
		    if (zero_pad) padding = '0';
		} else if (t == 'f') {
		    str = unparse_f_format(x, precision);
		} else if ((t == 'g') || (t == 'e')) {
		    // just use f-format for now
		    str = unparse_f_format(x, precision);
		} else if (t == 'p') {
		    if (scalar) format_error("invalid argument for \"%p\"");
		    str = (x == null) ? "null" : x.toString();
		} else {
		    format_error("invalid format character");
		}
		
		if (str != null) {
		    nchars = str.length();
		}
		int npad = (nchars < width) ? (width - nchars) : 0;
		
		// Left padding
		if (lpad && (npad > 0)) {
		    while (npad-- > 0) w.write(padding);
		}
		
		// The actual value
		if (str != null) {
		    w.write(str);
		} else {
		    w.write(buf, 0, nchars);
		}
		
		// Right padding
		if (!lpad && (npad > 0)) {
		    while (npad-- > 0) w.write(' ');
		}
	    }
	    if (autoflush) w.flush();
	} catch (IOException x) {
	    record_exc();
	} catch (ClassCastException x) {
	    format_error("illegal argument type");
	} catch (StringIndexOutOfBoundsException x) {
	    format_error("incomplete format specification");
	}
    }

    private int unparse_decimal(char[] buf, Object x, int precision) {
	long value = ((Number) x).longValue();
	if (scalar) {
	    // The real value is actually stored elsewhere
	    value = long_value;
	}

	int n = 0;
	int digit_start = 0;

	if (value < 0) {
	    // Negative decimal number: use sign
	    buf[n++] = '-';
	    value = -value;
	    digit_start = 1;
	}

	// Use negative version while unparsing to handle 2-complement
	// overflow problems for MIN_LONG
	value = -value;

	// Unparse the value with lsb first
	while (value != 0) {
	    int digit = (int) -(value % 10);
	    value = value / 10;
	    char c = Character.forDigit(digit, 10);
	    buf[n++] = c;
	}

	return (fix_number(buf, digit_start, n, precision));
    }

    private int unparse_bits(char[] buf, Object x, int radix, int bits,
			    boolean upcase, int precision) {
	long value = ((Number) x).longValue();
	if (scalar) {
	    // The real value is actually stored elsewhere
	    value = long_value;
	}

	// Chop off unnecessary bits
	if (x instanceof Byte) {
	    value = value & 0xffl;
	} else if ((x instanceof Short) || (x instanceof Character)) {
	    value = value & 0xffffl;
	} else if (x instanceof Integer) {
	    value = value & 0xffffffffl;
	}

	// Unparse the value with lsb first
	int n = 0;
	while (value != 0) {
	    int digit = (int) (value & (radix - 1));
	    value >>>= bits;
	    char c = Character.forDigit(digit, radix);
	    if (upcase) {
		c = Character.toUpperCase(c);
	    }
	    buf[n++] = c;
	}

	return fix_number(buf, 0, n, precision);
    }

    private static int fix_number(char[] buf, int digit_start, int n,
				  int precision) {
	// Handle precision (also takes care of unparsing zeros)
	if (precision == -1) {
	    // Output at least one digit (for "x == 0")
	    precision = 1;
	}
	if (precision > (n - digit_start)) {
	    // Not enough digits.  Append zeros until the precision
	    // constraint is specified.  These zeros will become
	    // leading zeroes after the string reversal below.
	    if (precision > (buf.length - digit_start)) {
		// Not enough space for requested precision.
		// Lower precision to meet space requirements
		precision = buf.length - digit_start;
	    }
	    while ((n - digit_start) < precision) {
		// Append a zero (the zero will become a leading zero
		// after the string reversal below.)
		buf[n++] = '0';
	    }
	}

	// Reverse "buf[digit_start..n-1]"
	int i = digit_start;
	int j = n - 1;
	while (i < j) {
	    char tmp = buf[i];
	    buf[i] = buf[j];
	    buf[j] = tmp;
	    i++;
	    j--;
	}

	return n;
    }

    private String unparse_f_format(Object x, int precision) {
	double value = ((Number) x).doubleValue();
	if (scalar) {
	    // The real value is actually stored elsewhere
	    value = double_value;
	}

	if (nf == null) {
	    nf = NumberFormat.getInstance();
	}

	if (precision == -1) {
	    precision = 6;
	}

	// Print it
	nf.setMaximumFractionDigits(precision);
	nf.setMinimumFractionDigits(precision);
	return nf.format(value);
    }

    private void format_error(String msg) {
	throw new IllegalArgumentException(msg);
    }

    private void record_exc() {
	had_exc = true;
    }
}
