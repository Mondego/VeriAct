package jtools.jwhich;


import java.util.Vector;

import javafe.genericfile.*;
import javafe.filespace.*;


/**
 ** The master module for the jwhich program.<p>
 **
 ** See the man page for details of its functionality.<p>
 **/

public class JWhich {

    /***************************************************
     *                                                 *
     * Resolving a reference-type name:		       *
     *                                                 *
     ***************************************************/

    /**
     ** findType: look up a reference-type name using
     ** Resolve.lookupName in the current namespace (cf. Resolve).  Uses
     ** Resolve.ensureType to make sure that the name names an actual
     ** reference type.
     **/
    //@ ensures \result != null ==> \result.myTypeName != null;
    //@ ensures \result != null ==> \result.myPackage != null;
    public static Resolve_Result findType(/*@ non_null @*/ String typeName) {
	try {
	    return Resolve.ensureType(Resolve.lookupName(typeName));
	} catch (Resolve_AmbiguousName name) {
	    System.err.println(typeName + ": " + name.getMessage());
	}

	return null;
    }


    /***************************************************
     *                                                 *
     * Displaying information about a reference type:  *
     *                                                 *
     ***************************************************/

    /*
     * Various display options for these functions:
     */
    public static boolean listSources	   = true;
    public static boolean listBinaries	   = true;
    public static boolean listXtras	   = true;
    public static boolean listDuplicates   = false;

    /**
     ** The set of extensions for extra files so far:
     **/
    //@ invariant extraExtensions.elementType == \type(String);
    //@ invariant !extraExtensions.containsNull;
    public static /*@ non_null @*/ Vector extraExtensions = new Vector();


    /**
     ** List the source, binary and/or Xtra files (depending on the
     ** settings of listSources, listBinaries, and listXtras) of a
     ** particular reference type in a package P.
     **/
    static void displayType(/*@ non_null @*/ Tree P, String typeName) {
	if (listSources) {
	    Tree sources = P.getChild(typeName + ".java");
	    displayFiles(sources);
        }

	if (listBinaries) {
	    Tree binaries = P.getChild(typeName + ".class");
	    displayFiles(binaries);
        }

	if (listXtras)
	    for (int i=0; i<extraExtensions.size(); i++) {
		String extension = (String)extraExtensions.elementAt(i);
		Tree extras = P.getChild(typeName + extension);
		displayFiles(extras);
	    };
    }

    /**
     ** List the filename(s) associated with a given UnionTree filespace
     ** node.<p>
     **
     ** Lists only the first one, unless listDuplicates is set.<p>
     **
     ** Does nothing if the node passed in is null.<p>
     **/
    static void displayFiles(Tree node) {
	if (node==null)
	    return;

	if (!listDuplicates || !(node instanceof UnionTree))
	    displayFile(node);
	else {
	    Tree[] duplicates = ((UnionTree)node).duplicates();
	    for (int i=0; i<duplicates.length; i++)
	        displayFile(duplicates[i]);
        }
    }

    /**
     ** List the filename associated with a given filespace
     ** node.
     **/
    static void displayFile(/*@ non_null @*/ Tree node) {
	System.out.println(((GenericFile)node.data).getHumanName()); //@ nowarn Cast, Null;
    }


    /***************************************************
     *                                                 *
     * Computing classpaths:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Our default bootclasspath; set by main before calling any of
     ** the option handling code.
     **/
    static /*@ non_null @*/ String defaultBootClassPath = "";


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
	    "jwhich: usage: ( [-bdsx] | -X<extension> | -bootclasspath <path>"
	        + " | -classpath <path> | <typename> )...");

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
	    case 's':
		listSources  = true;
		listBinaries = false;
		listXtras    = false;
		return true;

	    case 'b':
		listSources  = false;
		listBinaries = true;
		listXtras    = false;
		return true;

	    case 'x':
		listSources  = false;
		listBinaries = false;
		listXtras    = true;
		return true;

	    case 'd':
		listDuplicates = true;
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
	    if (argIndex<arguments.length) {
		sysPath = arguments[argIndex++];
		Resolve.set(getClassPath(defaultBootClassPath), false);
	    }

	} else if (arg.equals("-classpath")) {
	    if (argIndex<arguments.length) {
		userPath = arguments[argIndex++];
		Resolve.set(getClassPath(defaultBootClassPath), false);
	    }

	} else if (arg.startsWith("-X")) {
	    String extension = arg.substring("-X".length());
	    extraExtensions.addElement(extension);

	} else {
	    boolean legalArg = true;
	    for (int i=1; i<arg.length(); i++)
		legalArg = legalArg && handleFlag(arg.charAt(i));
	    if (!legalArg)
		usage();
	}
    }


    /** The main procedure for the jwhich command **/
    //@ requires \nonnullelements(args);
    public static void main(String[] args) {
	// Setup processing of arguments:
	arguments = args;
	int typesProcessed = 0;

	// Setup initial namespace:
	defaultBootClassPath = getDefaultBootClassPath();
	Resolve.set(getClassPath(defaultBootClassPath), false);

	// Process each argument in turn:
	while (argIndex<arguments.length) {
	    String arg = args[argIndex++];

	    if (arg.length()>0 && arg.charAt(0)=='-')
		handleOption(arg);
	    else {
		find(arg);
	        typesProcessed++;
	    }
        }

	// If no type names given, print a usage message:
	if (typesProcessed==0)
	    usage();
    }


    /**
     ** find: implement jwhich for 1 reference-type name.
     **/
    public static void find(/*@ non_null @*/ String typeName) {
	Resolve_Result search = findType(typeName);
	if (search != null)
	    displayType(search.myPackage, search.myTypeName);
    }
}
