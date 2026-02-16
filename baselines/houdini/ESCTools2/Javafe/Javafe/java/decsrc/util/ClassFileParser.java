package decsrc.util;
import java.io.DataInput;
import java.io.IOException;

/** A "ClassFileParser" parses the contents of a class-file.
    Various abstract methods are invoked as pieces of the class-file
    are parsed.  Subclasses should override these methods to
    provide specific functionality.

    A subclass of this abstraction can depend on the following
    order for callbacks.  Parenthesis are used for grouping,
    asterisks for repetition, vertical bars for alternation,
    and question marks for optional calls:
    <pre>
    |	set_version
    |
    |	set_num_constants
    |	set_const*
    |	set_const_ref*
    |
    |	set_modifiers
    |	set_this_class
    |	set_super_class
    |
    |	set_num_interfaces
    |	set_interface*
    |
    |	set_num_fields
    |	( set_field (set_field_initializer | set_field_attribute)* )*
    |
    |	set_num_methods
    |	( set_method ( set_method_attribute |
    |	               ( set_method_body
    |			   ( set_method_handler | set_method_body_attribute )*
    |		       )
    |		     )*
    |	)*
    |
    |	set_class_attribute*
    </pre>
    
    @deprecated Use BCELReader for Java 1.5 and later versions
*/
public abstract class ClassFileParser implements ClassFileConstants {
    //**** methods to be overridden ****

    /** Set the version numbers for this class. */
    protected abstract void set_version(int major, int minor)
	throws ClassFormatError;

    /** Set the class modifier flags. */
    protected abstract void set_modifiers(int modifiers)
	throws ClassFormatError;

    /** Set the number of constants in the constant pool.  The
	constants will be numbered 1..cnum-1.  (The constant at entry
	zero is missing.) */
    protected abstract void set_num_constants(int cnum)
	throws ClassFormatError;

    /** Set number of interfaces implemented by this class. */
    protected abstract void set_num_interfaces(int n)
	throws ClassFormatError;

    /** Set number of fields declared in this class. */
    protected abstract void set_num_fields(int n)
	throws ClassFormatError;

    /** Set number of methods declared in this class. */
    protected abstract void set_num_methods(int n)
	throws ClassFormatError;

    /** Set the class name.  "cindex" is the constant pool entry that
	contains the "CONSTANT_Class" entry corresponding to this
	class.  This routine is called after "set_num_constants",
	and after "set_const(cindex)". */
    protected abstract void set_this_class(int cindex)
	throws ClassFormatError;

    /** Set the super class name.  If "cindex" is zero, this class has
	no super class.  Otherwise "cindex" is the constant pool entry
	that contains the "CONSTANT_Class" entry corresponding to this
	class.  This routine is called after "set_num_constants", and
	if "cindex" is non-zero, after "set_const(cindex)". */
    protected abstract void set_super_class(int cindex)
	throws ClassFormatError;

    /** Set the name of the "index"th interface implemented by this
	class.  This routine is called exactly once for each interface
	implemented by this class.  This routine is called after
	"set_num_constants", after "set_num_interfaces", and after
	"set_const(cindex)". */
    protected abstract void set_interface(int index, int cindex)
	throws ClassFormatError;

    /** Set the type and value of the "i"th constant pool entry.
	This routine is called exactly once for each constant pool
	entry of one of the following types:
	<pre>
    		constant type	Type of "value"
    		--------------------------------
    		UTF8		String
    		String		String
    		Class		String
    		Integer		Integer
    		Float		Float
    		Long		Long
    		Double		Double
	</pre>
	This routine is called after "set_num_constants". */
    protected abstract void set_const(int i, int ctype, Object value)
	throws ClassFormatError;

    /** Set the type and value of the "i"th constant pool entry.
	This routine is called exactly once for each constant pool
	entry of one of the following types:  Fref, Mref, IMref.
	<p>
	This routine is called after "set_num_constants".  It is
	called after all calls to "set_const". */
    protected abstract void set_const_ref(int i, int ctype,
					  int class_index,
					  String field_name,
					  String type)
	throws ClassFormatError;

    /** Record unparsed class file attribute.  This routine is called
        for all class-file attributes. The default implementation just
        skips the "n" bytes of data that define the attribute value.
        If a subclass overrides this method, it must read exactly "n"
        bytes from the input stream. */
    protected void set_class_attribute(/*@ non_null */ String aname,
				       /*@ non_null */ DataInput stream, int n)
	throws IOException, ClassFormatError
    {
	stream.skipBytes(n);
    }

    /** Set the "i"th field defined by this class.  This routine
	is called exactly once for each field.  It is called after
	"set_num_fields". */
    protected abstract void set_field(int i, String fname, String type,int mod)
	throws ClassFormatError;

    /** Set the initializer for "i"th field defined by this class.
	This routine is called at most once for each field.  It is
	called after "set_field(i)". */
    protected abstract void set_field_initializer(int i, Object value)
	throws ClassFormatError;

    /** Record unparsed field attribute for field number "i".  This
        routine is called for all field attributes except for
        "ConstantValue". The default implementation just skips the "n"
        bytes of data that define the attribute value.  If a subclass
        overrides this method, it must read exactly "n" bytes from the
        input stream.  This routine is called after "set_field(i)". */
    protected void set_field_attribute(int i, String aname,
				       DataInput stream, int n)
	throws IOException, ClassFormatError
    {
	stream.skipBytes(n);
    }

    /** Set the "i"th field defined by this class.  This routine
	is called exactly once for each method.  It is called after
	"set_num_methods". */
    protected abstract void set_method(int i, String mname, String sig,
				       int mod)
	throws ClassFormatError;

    /** Set the body for "i"th method defined by this class.
	This routine is called at most once for each method.  It is
	called after "set_method(i)". */
    protected abstract void set_method_body(int i,
					    int max_stack,
					    int max_local,
					    byte[] code,
					    int num_handlers)
	throws ClassFormatError;

    /** Set the body for "j"th handler in the "i"th method defined by this
	class.  This routine is called exactly once for each handler
	in the "i"th method.  It is called after "set_method_body(i)". */
    protected abstract void set_method_handler(int i,
					       int j,
					       int start_pc, int end_pc,
					       int handler_pc,
					       int catch_index)
	throws ClassFormatError;

    /** Record unparsed method attribute for method number "i".  This
        routine is called for all method attributes except for
        "Code". The default implementation just skips the "n" bytes of
        data that define the attribute value.  If a subclass overrides
        this method, it must read exactly "n" bytes from the input
        stream.  This routine is called after "set_method(i)". */
    protected void set_method_attribute(int i, String aname,
					DataInput stream, int n)
	throws IOException, ClassFormatError
    {
	stream.skipBytes(n);
    }

    /** Record unparsed method-body attribute for method number "i".
        This routine is called for all method-body attributes.  The
        default implementation just skips the "n" bytes of data that
        define the attribute value.  If a subclass overrides this
        method, it must read exactly "n" bytes from the input stream.
        This routine is called after "set_method_body(i)" and the
	corresponding "set_method_handler(i)" calls. */
    protected void set_method_body_attribute(int i, String aname,
					     DataInput stream, int n)
	throws IOException, ClassFormatError
    {
	stream.skipBytes(n);
    }

    // Fields used during parsing
    private DataInput	stream;
    private byte[]	tags;
    private Object[]	clist;

    /** Parse class file from a supplied stream.  Callbacks are made
	to register the contents of the parsed class.  The order
	of the callbacks is described in the class header. */
    protected void parse_file(DataInput stream)
	throws ClassFormatError, IOException
    {
	try {
	    this.stream = stream;
	    parse_file();
	} finally {
	    this.stream = null;
	    this.tags = null;
	    this.clist = null;
	}
    }

    private void parse_file() throws ClassFormatError, IOException {
	DataInput stream = this.stream;

	// Read header
	int magic = stream.readInt();
	if (magic != MAGIC) error("bad magic number");
	int minor = stream.readUnsignedShort();
	int major = stream.readUnsignedShort();
	if ((major != MAJOR) || (minor > MINOR)) error("illegal version");
	set_version(major, minor);

	// Parse constants
	int cnum = stream.readUnsignedShort();
	if (cnum < 1) error("invalid constant pool count");
	Object[] clist = new Object[cnum];
	byte[] tags = new byte[cnum];
	this.tags = tags;
	this.clist = clist;
	
	tags[0] = (byte) CONSTANT_Unused;
	int c = 1;
	while (c < cnum) {
	    int tag = stream.readUnsignedByte();
	    tags[c] = (byte) tag;
	    switch ((int)tag) {
	      case CONSTANT_Utf8:
		clist[c] = stream.readUTF();
		break;
	      case CONSTANT_Integer:
		clist[c] = new Integer(stream.readInt());
		break;
	      case CONSTANT_Float:
		clist[c] = new Float(Float.intBitsToFloat(stream.readInt()));
		break;
	      case CONSTANT_Long:
		clist[c] = new Long(stream.readLong());
		clist[c+1] = null;
		tags[c+1] = (byte)CONSTANT_Unused;
		c++;
		break;
	      case CONSTANT_Double:
		clist[c] = new Double(Double.longBitsToDouble
				      (stream.readLong()));
		clist[c+1] = null;
		tags[c+1] = (byte)CONSTANT_Unused;
		c++;
		break;
	      case CONSTANT_Class:
	      case CONSTANT_String:
		clist[c] = new Integer(stream.readUnsignedShort());
		break;
	      case CONSTANT_Fref:
	      case CONSTANT_Mref:
	      case CONSTANT_IMref:
	      case CONSTANT_NameType:
		int i1 = stream.readUnsignedShort();
		int i2 = stream.readUnsignedShort();
		clist[c] = new Integer((i1 << 16) | i2);
		break;
	      default:
		error("unknown constant type: " + tag);
	    }
	    c++;
	}

	// Register constants: pass 1
	set_num_constants(cnum);
	for (int i = 1; i < cnum; i++) {
	    Object x = clist[i];
	    int tag = (int) tags[i];
	    switch (tag) {
	      case CONSTANT_Utf8:
	      case CONSTANT_Integer:
	      case CONSTANT_Float:
	      case CONSTANT_Long:
	      case CONSTANT_Double:
		set_const(i, tag, x);
		break;
	      case CONSTANT_Class:
	      case CONSTANT_String: {
		  int index = ((Integer) x).intValue();
		  set_const(i, tag, get_utf(index));
		  break;
	      }
	    }
	}

	// Register constants: pass 2
	for (int i = 1; i < cnum; i++) {
	    Object x = clist[i];
	    int tag = (int) tags[i];
	    switch (tag) {
	      case CONSTANT_Fref:
	      case CONSTANT_Mref:
	      case CONSTANT_IMref: {
		  int v = ((Integer) x).intValue();

		  // Get class name
		  int ci = v >>> 16;
		  check(ci, CONSTANT_Class);

		  // Get name/type
		  int nti = v & 0xffff;
		  check(nti, CONSTANT_NameType);
		  int nt = ((Integer) clist[nti]).intValue();
		  int ni = nt >>> 16;
		  int ti = nt & 0xffff;
		  String id = get_utf(ni);
		  String sig = get_utf(ti);

		  set_const_ref(i, tag, ci, id, sig);
		  break;
	      }
	    }
	}

	// Read miscellaneous info
	set_modifiers(stream.readUnsignedShort());
	int this_index = stream.readUnsignedShort();
	check(this_index, CONSTANT_Class);
	set_this_class(this_index);
	int super_index = stream.readUnsignedShort();
	if (super_index != 0) {
	    check(super_index, CONSTANT_Class);
	}
	set_super_class(super_index);

	// Read interface list
	int inum = stream.readUnsignedShort();
	set_num_interfaces(inum);
	for (int i = 0; i < inum; i++) {
	    int iindex = stream.readUnsignedShort();
	    check(iindex, CONSTANT_Class);
	    set_interface(i, iindex);
	}

	// Read field list
	int fnum = stream.readUnsignedShort();
	set_num_fields(fnum);
	for (int i = 0; i < fnum; i++) {
	    read_field(i);
	}

	// Read method list
	int mnum = stream.readUnsignedShort();
	set_num_methods(mnum);
	for (int i = 0; i < mnum; i++) {
	    read_method(i);
	}

	// File attributes
	int nfile = stream.readUnsignedShort();
	for (int i = 0; i < nfile; i++) {
	    String name = get_utf(stream.readUnsignedShort());
	    int length = stream.readInt();	// Get length
	    set_class_attribute(name, stream, length);
	}

	/* XXX We should check that we are at the end of the file,
	   XXX but that will require generating and catching an
	   XXX EOFException.
		try {
		    stream.readByte();
		    error("garbage at end of class file");
		} catch (EOFException x) { }
	*/
    }

    /** Check that constant number "index" has type "tag". */
    private void check(int index, int tag) {
	if ((index <= 0) || (index >= tags.length)) {
	    error("constant pool index is out of range");
	}
	if (tags[index] != tag) {
	    error("constant pool entry has unexpected type");
	}
    }

    /** Read a class name. */
    protected String get_class(int i) {
	check(i, CONSTANT_Class);
	int index = ((Integer) clist[i]).intValue();
	return get_utf(index);
    }

    /** Read a utf8. */
    private String get_utf(int i) {
	check(i, CONSTANT_Utf8);
	return ((String) clist[i]);
    }


    private void read_field(int f) throws ClassFormatError, IOException {
	int access = stream.readUnsignedShort();
	String name = get_utf(stream.readUnsignedShort());
	String sig = get_utf(stream.readUnsignedShort());
	set_field(f, name, sig, access);

	// Read field attributes
	int n = stream.readUnsignedShort();
	for (int i = 0; i < n; i++) {
	    String aname = get_utf(stream.readUnsignedShort());
	    int length = stream.readInt();
	    if (aname.equals("ConstantValue")) {
		int index = stream.readUnsignedShort();
		Object initializer;
		if (tags[index] == CONSTANT_String) {
		    int uindex = ((Integer) clist[index]).intValue();
		    initializer = get_utf(uindex);
		} else {
		    initializer = clist[index];
		}
		set_field_initializer(f, initializer);
	    } else {
		set_field_attribute(f, aname, stream, length);
	    }
	}
    }

    private void read_method(int m) throws ClassFormatError, IOException {
	int mod = stream.readUnsignedShort();
	String name = get_utf(stream.readUnsignedShort());
	String sig = get_utf(stream.readUnsignedShort());
	set_method(m, name, sig, mod);

	// Read method attributes
	int n = stream.readUnsignedShort();
	for (int i = 0; i < n; i++) {
	    String aname = get_utf(stream.readUnsignedShort());
	    int length = stream.readInt();
	    if (aname.equals("Code")) {
		read_method_body(m);
	    } else {
		set_method_attribute(m, aname, stream, length);
	    }
	}
    }

    private void read_method_body(int m) throws ClassFormatError, IOException {
	int ms = stream.readUnsignedShort();
	int ml = stream.readUnsignedShort();
	byte[] code = new byte[stream.readInt()];
	stream.readFully(code);
	int nh = stream.readUnsignedShort();
	set_method_body(m, ms, ml, code, nh);
	for (int h = 0; h < nh; h++) {
	    int spc = stream.readUnsignedShort();
	    int epc = stream.readUnsignedShort();
	    int hpc = stream.readUnsignedShort();
	    int ct = stream.readUnsignedShort();
	    if (ct != 0) {
		check(ct, CONSTANT_Class);
	    }
	    set_method_handler(m, h, spc, epc, hpc, ct);
	}

	// Read method body attributes
	int nb = stream.readUnsignedShort();
	for (int j = 0; j < nb; j++) {
	    String bname = get_utf(stream.readUnsignedShort());
	    int blength = stream.readInt();
	    set_method_body_attribute(m, bname, stream, blength);
	}
    }

    private static void error(String str) throws ClassFormatError {
	throw new ClassFormatError(str);
    }
}
