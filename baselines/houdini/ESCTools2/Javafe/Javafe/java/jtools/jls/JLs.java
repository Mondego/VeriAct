package jtools.jls;

import java.util.Enumeration;
import java.util.Vector;

import javafe.filespace.*;

/**
 * The master module for the jls program.
 *
 * <p> See the man page for details of its functionality. </p>
 */

public class JLs
{
    /***************************************************
     *                                                 *
     * Handling jls for a series of package names:     *
     *                                                 *
     ***************************************************/

    /**
     ** Perform jls's functionality for a series of package names:
     ** names[start], names[start+1], ...<p>
     **
     ** The current program settings are taken into account.  An empty
     ** series of names is considered equivalent to the series
     ** consisting of just the empty package name.<p>
     **
     ** names must be non-null and start>=0<p>
     **/
    //@ requires \nonnullelements(names);
    //@ requires start >= 0;
    public static void handlePackageNames(String[] names, int start) {
	int count = names.length - start;	// # of names to process

	// no names case:
	if (count<1) {
	    handlePackageName("", false);
	    return;
	}

	// single names case:
	if (count==1) {
	    handlePackageName(names[start], false);
	    return;
	}

	// multiple names case:
	for (int i=start; i<names.length; i++)
	    handlePackageName(names[i], true);
    }


    /***************************************************
     *                                                 *
     * Handling separators:			       *
     *                                                 *
     ***************************************************/

    /** Has anything been output yet by jls? **/
    private static boolean anyOutputYet = false;

    /** All jls routines should call this instead of System.out.print **/
    public static void print(String s) {
	System.out.print(s);
	anyOutputYet = true;
    }

    /** All jls routines should call this instead of System.out.println **/
    public static void println(String s) {
	System.out.println(s);
	anyOutputYet = true;
    }

    /** All jls routines should call this instead of System.out.println **/
    public static void println() {
	System.out.println();
	anyOutputYet = true;
    }

    /**
     ** Print a blank line as a seperator on System.out.print, unless no
     ** output has been printed yet, in which case, do nothing.
     **/
    public static void printSeparator() {
	if (anyOutputYet)
	    System.out.println();
    }


    /***************************************************
     *                                                 *
     * Handling jls for a single package name:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Should we descend recursively?
     **/
    public static boolean recurse = false;

    /**
     ** Should we display reference types?
     **/
    public static boolean displayTypes = true;

    /**
     ** Should we display packages?
     **/
    public static boolean displayPackages = true;

    /**
     ** Should we ignore empty packages?
     **/
    public static boolean ignoreEmpty = false;


    /**
     ** Display a package name via println.  If isHeader is set,
     ** proceeds it by a separator line and puts a ':' at the end of the
     ** package name.
     **/
    public static void displayPackageName(/*@ non_null @*/ Tree P,
					  boolean isHeader) {
	if (isHeader)
	    printSeparator();

	println(PkgTree.getPackageName(P) + (isHeader ? ":" : ""));
    }


    /**
     ** Perform jls's functionality for a single package name.<p>
     **
     ** The current program settings are taken into account.  A header
     ** is displayed before any other normal output, regardless of the
     ** program settings, if alwaysDisplayHeader is true.<p>
     **
     ** packageName must be non-null<p>
     **/
    public static void handlePackageName(/*@ non_null @*/ String packageName,
					 boolean alwaysDisplayHeader) {
	// Look up the package name:
	Tree P = lookupPackage(Resolve.namespace, packageName);
	if (P==null)
	    return;

	// Display it as a header if needed:
	if (alwaysDisplayHeader)
	    displayPackageName(P, true);

	if (!recurse) {
	    listPackage(P, displayTypes, displayPackages);
	    return;
	}

	/*
	 * The recursive case:
	 */
	Enumeration E = PkgTree.packages(P);

	/*
	 * The first element (always P) is not part of the contents of P
	 * and, thus, does not need to be displayed.  Its types do,
	 * however, if we are displaying types.
	 */
	//@ assume E.moreElements;
	E.nextElement();
	listPackage(P, displayTypes, false);

	while (E.hasMoreElements()) {
	    Tree C = (Tree)E.nextElement();

	    // If ignoreEmpty is true, skip empty subpackages:
	    if (ignoreEmpty && isEmpty(C))
		continue;

	    // Display the subpackname name, if displaying packages:
	    // (it acts as a header if we are also displaying types)
	    if (displayPackages)
		displayPackageName(C, displayTypes);

	    // Followed by the types in it, if displaying types:
	    listPackage(C, displayTypes, false);
	}
	
    }


    /**
     ** lookup a package in namespace using a package-name String.<p>
     **
     **    - handles errors by complaining to System.err
     **        then returning null.<p>
     **
     **    - ignores types while searching; in particular, unlike
     **      Resolve.lookup, ambiguous names (aka, names for both a type
     **      and a package) do not cause any problems.<p>
     **/
    public static Tree lookupPackage(Tree namespace,
				     /*@ non_null @*/ String packageName) {
	Tree P = Resolve.namespace.getQualifiedChild(packageName, '.');
	if (P==null) {
	    System.err.println(packageName + ": no such package");
	    return null;
	};

	return P;
    }


    /***************************************************
     *                                                 *
     * Listing a package:			       *
     *                                                 *
     ***************************************************/

    /**
     ** List the contents of a package without any header.<p>
     **
     ** Displays types and/or packages depending on its arguments.<p>
     **
     ** Uses the current program settings except for displayTypes and
     ** displayPackages, which are overridden by the arguments.<p>
     **/
    public static void listPackage(/*@ non_null @*/ Tree P, boolean displayTypes,
				   boolean displayPackages) {
	if (!displayTypes && !displayPackages)
	    return;

	Vector contents = packageContents(P, displayTypes,
					  displayPackages);

	printList(contents);
    }


    /***************************************************
     *                                                 *
     * Displaying a list of Strings:		       *
     *                                                 *
     ***************************************************/

    /**
     ** How wide a display to produce when outputing lists.<p>
     **
     ** If -1, always display 1 item/line.<p>
     **
     ** Otherwise, gives the # of characters we will attempt to limit
     ** output to.  In the case of items wider than this, we will exceed
     ** this limit, using 1 item/line.<p>
     **/
    public static int outputLimit = 80;

    /**
     ** The separator to use between columns in multi-column displays.
     **/
    //@ public invariant separator.length() > 0;
    public static final String separator = "  ";

    /**
     ** Compute the size of the longest String in a Vector of non-null
     ** Strings.
     **/
    //@ requires list.elementType == \type(String);
    //@ requires !list.containsNull;
    //@ ensures \result >= 0;
    public static int maxItem(/*@ non_null @*/ Vector list) {
	int maxSize = 0;

	for (Enumeration E=list.elements(); E.hasMoreElements(); )
	    maxSize = Math.max(maxSize, ((String)E.nextElement()).length());

	return maxSize;
    }

    /**
     ** Print n spaces quickly.
     **/
    private static final String spaces =
	"                                        ";	// 40 spaces

    protected static void printSpaces(int n) {
	while (n>0) {
	    int inc = Math.min(n,40);
	    print(spaces.substring(0,inc));
	    n -= inc;
	}
    }


    /**
     ** Display a Vector of non-null Strings, using multiple columns if
     ** possible on System.out, so as to produce the shortest output
     ** possibile consistent with the variable outputLimit above.
     **/
    //@ requires list.elementType == \type(String);
    //@ requires !list.containsNull;
    public static void printList(/*@ non_null @*/ Vector list) {
	int size = list.size();
	if (size==0)
	    return;		// No need to display the empty list

	/*
	 * Display using a single column list if outputLimit == -1 or
	 * there is an item too big to fit (cf. outputLimit):
	 */
	int itemWidth = maxItem(list);
	if (itemWidth>=outputLimit) {
	    Enumeration E = list.elements();
	    while (E.hasMoreElements())
		println((String)E.nextElement());
	    return;
	}

	/*
	 * Compute # of columns possible & how many lines of output
	 * needed:
	 */
	/*@ assume (outputLimit-itemWidth) >= 0 &&
          @        (itemWidth + separator.length()) >= 0 ==>
          @          ((outputLimit-itemWidth) /
          @           (itemWidth + separator.length())) >= 0;
          @*/
	int columns = 1 + ((outputLimit-itemWidth) /
                           (itemWidth + separator.length()));
	int offset = size / columns;	// rounds down...
	while (columns*offset < size)	// correct for that
	    offset++;

        //	System.out.println(itemWidth + " " + columns);
        //	System.out.println(list.size() + " " + offset);


	/*
	 * Display the strings in the resulting # of columns:
	 */
	for (int base=0; base<offset; base++) {
	    for (int c=0; c<columns; c++) {
		int i = base + offset*c;
		if (i>=size)
		    continue;

		String element = (String)list.elementAt(i);
		if (c != 0)
		    print(separator);
		print(element);
		printSpaces(itemWidth - element.length());
	    }

	    println();
	}
    }


    /***************************************************
     *                                                 *
     * Computing the contents of a package:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Should we use absolute package and type names when computing
     ** the contents of a package?
     **/
    public static boolean useAbsolute = false;

    /**
     ** Should we include a type if we have source for the type?
     **/
    public static boolean displaySources = true;

    /**
     ** Should we include a type if we have a binary for the type?
     **/
    public static boolean displayBinaries = true;

    /**
     ** Should entries in a package's contents be marked with a
     ** descriptive symbol at the end to denote what kind of entry they
     ** are?
     **/
    public static boolean markEntries = false;


    /**
     ** Compute the contents of a package for display.  Returns a Vector
     ** of Strings, where each String represents one entry of the
     ** package.  The entries are in sorted order.<p>
     **
     ** Includes package entries if includePackages is set.  Includes
     ** types with source if includeTypes and includeSource are
     ** true.  Includes types with binaries if includeTypes and
     ** includeBinary are true.<p>
     **
     ** useAbsolute and markEntries are taken into account.<p>
     **/
    //@ ensures \result != null;
    //@ ensures \result.elementType == \type(String);
    //@ ensures !\result.containsNull;
    public static Vector packageContents(/*@ non_null @*/ Tree P,
					 boolean includeTypes,
					 boolean includePackages) {
	Vector R = new Vector(10);
	//@ set R.elementType = \type(String);
	//@ set R.containsNull = false;
	String namePrefix = useAbsolute ? P.getQualifiedName(".") : "";

	Enumeration E = TreeWalker.sortedChildren(P);
	while (E.hasMoreElements()) {
	    Tree component = (Tree)E.nextElement();

	    String label = component.getLabel(); 
	    //@ assume label != null;
   	    String extension = Extension.getExtension(label);
   	    String relName = Extension.getBasename(label);
   	    String name = Resolve.combineNames(namePrefix, relName, ".");

	    // Package case:
	    if (includePackages && extension.equals("")
		&& (!ignoreEmpty || !isEmpty(component)))
		R.addElement(name + (markEntries ? "/" : ""));

	    // Type source case:
	    if (includeTypes && displaySources &&
                extension.equals(".java")) {
		if (P.getChild(relName+".class")==null)
		    R.addElement(name);		// source, no binary
		else {
		    // source + binary
		    R.addElement(name + (markEntries ? "&" : ""));
		}
	    }

	    // Type binary case:
	    if (includeTypes && displayBinaries &&
                extension.equals(".class")) {
		if (P.getChild(relName+".java")==null)
		    // binary, no source
		    R.addElement(name + (markEntries ? "*" : ""));
		else {
		    // source + binary
		    if (!displaySources)	// avoid duplication
			R.addElement(name + (markEntries ? "&" : ""));
		}
	    }
	}

	return R;
    }


    /***************************************************
     *                                                 *
     * Determining if a package is empty:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Is package P empty?<p>
     **
     ** I.e., does P contain any source or binary files or any
     ** subpackages?
     **/
    public static boolean isEmpty(/*@ non_null @*/ Tree P) {
	if (PkgTree.components(P,".class").hasMoreElements())
	    return false;

	if (PkgTree.components(P,".java").hasMoreElements())
	    return false;

	// Check for subpackages:
	Enumeration E = PkgTree.components(P,"");
	while (E.hasMoreElements()) {
	    Tree child = (Tree)E.nextElement();
	    if (!child.isLeaf())
		return false;
	}

	return true;
    }


    /***************************************************
     *                                                 *
     * Command line parsing:			       *
     *                                                 *
     ***************************************************/

    /**
     ** The list of our command line arguments
     **/
    //@ invariant \nonnullelements(arguments);
    static String[] arguments;

    /**
     ** An index into arguments, pointing to the next argument to be
     ** processed:
     **/
    //@ invariant argIndex >= 0;
    static int argIndex = 0;


    /** The path specified by the -classpath option or null if none **/
    static String userPath   = null;

    /** The path specified by the -bootclasspath option or null if none **/
    static String sysPath    = null;


    /** Print out our usage information on System.err **/
    public static void usage() {
	System.err.println(
                           "jls: usage: [-abcEFpRs1] [-C<width>] [-bootclasspath <classpath>]"
                           + " [-classpath <classpath>] <package_name>...");
	return;
    }


    /**
     ** Code to handle 1 flag (a char).  If the flag is a legal one,
     ** then the current settings are set appropriately and true is
     ** returned.  Otherwise, false is returned and the settings are
     ** left unchanged.
     **/
    static boolean handleFlag(char flag) {
	switch (flag) {
	    case 'a':
		useAbsolute = true;
		return true;

	    case 'b':
		displaySources = false;
		displayBinaries = true;
		return true;

	    case 'c':
		displayTypes = true;
		displayPackages = false;
		return true;

	    case 'E':
		ignoreEmpty = true;
		return true;

	    case 'F':
		markEntries = true;
		return true;

	    case 'p':
		displayTypes = false;
		displayPackages = true;
		return true;

	    case 'R':
		recurse = true;
		return true;

	    case 's':
		displaySources = true;
		displayBinaries = false;
		return true;

	    case '1':
		outputLimit = -1;
		return true;

	    default:
		return false;
	}
    }


    /**
     ** Code to handle one option (anything starting with a '-').<p>
     **
     ** May consume additional arguments by advancing argIndex.<p>
     **/
    static void handleOption(/*@ non_null @*/ String arg) {
	if (arg.equals("-bootclasspath")) {
	    if (argIndex<arguments.length)
		sysPath = arguments[argIndex++];

	} else if (arg.equals("-classpath")) {
	    if (argIndex<arguments.length)
		userPath = arguments[argIndex++];

	} else if (arg.length()>2 && arg.charAt(1)=='C') {
	    try {
		outputLimit = (new Integer(arg.substring(2)).intValue());
	    } catch (NumberFormatException n) {
		System.err.println("Unparsable integer: " 
                                   + n.getMessage());
		System.exit(1);
	    }

	} else {
	    boolean legalArg = true;
	    for (int i=1; i<arg.length(); i++)
		legalArg = legalArg && handleFlag(arg.charAt(i));
	    if (!legalArg) {
		usage();
		System.exit(1);
	    }
	}

	return;
    }


    /**
     ** Compute the classpath to use from userPath, sysPath, and the
     ** default value for our bootclasspath.  A value of "." is
     ** hard-wired for the classpath default value.
     **
     ** (We can't read environment variables, so we have to default to
     **  "." for the classpath.  Ideally, the script calling us will
     **  add "-classpath <user's CLASSPATH value or "." if unset>" in
     **  front of the user supplied arguments so we will do the right
     **  thing.)
     **/
    //@ ensures \result != null;
    static String getClassPath(/*@ non_null @*/ String defaultBootClassPath) {
	String result = sysPath;
	if (result==null)
	    result = defaultBootClassPath;
	
	String user = userPath != null ? userPath : ".";

	if (!user.equals("")) {
	    if (!result.equals(""))
		result += ":";
	    result += user;
	}

	return result;
    }


    /**
     ** Attempt to guess the default bootclasspath for the version of
     ** java we are running on.<p>
     **
     ** We do this via black magic and cooperation with the script
     ** running us.  This is necessary because of behavior differences
     ** between Java 1.2 and earlier versions.
     **
     ** There are 3 cases depending on what JVM we are running on:
     **
     **   (1) First, we lookup the property sun.boot.class.path (should
     **       be defined for all Sun Javas 1.2 and higher); if
     **       defined, we use its value.
     **
     **   (2) If that fails, we lookup the java.lang.path property.
     **       JVMs set it to the classpath specified for use in
     **       finding the jls application classes plus (for Java 1.1.*
     **       and earlier only) the default bootclasspath.
     **
     **       We remove the components of java.lang.path specifying
     **       where to find the jls application classes (the script
     **       passes the number of components to remove via the
     **       java.lang.path.skip property).  If a non-empty path
     **       remains (aka, we are not running under Java 1.2 and
     **       later), we use it.
     **
     **   (3) We give up and use the empty path ("").
     **
     ** Thus, in summary, this routine should work except for non-Sun
     ** JVMs implementing Java 1.2 and later.  If such a JVM is being
     ** used, the jls script must be modified to pass in a hardwired
     ** default bootclasspath to use via a -bootclasspath argument to
     ** jls placed in front of any user arguments.
     **/
    //@ ensures \result != null;
    static String getDefaultBootClassPath() {
	String result = System.getProperty("sun.boot.class.path", null);
	if (result==null)
	    result = javafe.filespace.ClassPath.current();

	return result;
    }


    /**
     ** The main procedure for the jls command
     **/
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
	// Setup processing of arguments:
	arguments = args;

	// Process any options:
	argIndex=0;
	String arg;
	while (argIndex<arguments.length && (arg=args[argIndex]).length()>0
               && arg.charAt(0)=='-') {
	    argIndex++;
	    handleOption(arg);
	}


	/*
	 * Setup namespace to use:
	 */
	String classPath = getClassPath(getDefaultBootClassPath());	
	Resolve.set(classPath, false);


	// Process the remaining arguments as package names:
	handlePackageNames(args, argIndex);
    }
}
