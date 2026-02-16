package jtools.jcheck;


import java.util.Enumeration;
import java.util.Hashtable;
import java.io.*;

import javafe.filespace.*;
import javafe.genericfile.*;

import jtools.jcheck.parseTypes.*;


/**
 ** The master module for the jcheck program.<p>
 **
 ** See the man page for details of its functionality.<p>
 **/

public class JCheck {

    /***************************************************
     *                                                 *
     * Handling jcheck for a series of package names:  *
     *                                                 *
     ***************************************************/

    /**
     ** Perform jcheck's functionality for a series of package names:
     ** names[start], names[start+1], ...<p>
     **
     ** The current program settings are taken into account.  An empty
     ** series of names is considered equivalent to the series
     ** consisting of just the empty package name.<p>
     **
     ** names must be non-null and start>=0<p>
     **/
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

    /** Has anything been output yet by jcheck? **/
    private static boolean anyOutputYet = false;

    /** All jcheck routines should call this instead of System.out.print **/
    public static void print(String s) {
	System.out.print(s);
	anyOutputYet = true;
    }

    /** All jcheck routines should call this instead of System.out.println **/
    public static void println(String s) {
	System.out.println(s);
	anyOutputYet = true;
    }

    /** All jcheck routines should call this instead of System.out.println **/
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
     * Handling jcheck for a single package name:      *
     *                                                 *
     ***************************************************/

    /**
     ** Should we descend recursively?
     **/
    public static boolean recurse = false;

    /**
     ** Display a header via println, unless it is a duplicate of the
     ** last header printed.  Calling this routine with null resets
     ** things so the next header is always printed.
     **/
    private static Tree lastPackage = null;

    public static void displayHeader(Tree P) {
	if (P==lastPackage)
	    return;

	lastPackage = P;
	if (P==null)
	    return;

	printSeparator();
	println(PkgTree.getPackageName(P)+":");
    }


    /**
     ** Perform jcheck's functionality for a single package name.<p>
     **
     ** The current program settings are taken into account.  A header
     ** is displayed before any other normal output, regardless of the
     ** program settings, if alwaysDisplayHeader is true.<p>
     **
     ** packageName must be non-null<p>
     **/
    public static void handlePackageName(String packageName,
					 boolean alwaysDisplayHeader) {
	// Look up the package name:
	Tree P = lookupPackage(Resolve.namespace, packageName);
	if (P==null)
	    return;

	// Display it as a header if needed:
	displayHeader(null);
	if (alwaysDisplayHeader)
	    displayHeader(P);

	if (!recurse) {
	    checkPackage(P);
	    return;
	}

	/*
	 * The recursive case:
	 */
	Enumeration E = PkgTree.packages(P);
	while (E.hasMoreElements())
	    checkPackage((Tree)E.nextElement());
    }


    /**
     ** lookup a package in namespace using a package-name String.<p>
     **
     **    - handles errors (including failure to resolve to a package name
     **        and empty identifier parts) by complaining to System.err
     **        then returning null.<p>
     **
     **    - ignores types while searching; in particular, unlike
     **      Resolve.lookup, ambiguous names (aka, names for both a type
     **      and a package) do not cause any problems.<p>
     **/
    public static Tree lookupPackage(Tree namespace, String packageName) {
	Tree P = Resolve.namespace.getQualifiedChild(packageName, '.');
	if (P==null) {
	    System.err.println(packageName + ": no such package");
	    return null;
	};

	return P;
    }


    /***************************************************
     *                                                 *
     * Utility routines for use in checking a package: *
     *                                                 *
     ***************************************************/

    /**
     ** This is set by complain when an error is encountered during a
     ** checking phase.
     **/
    private static boolean error = false;

    /**
     ** The current package we are checking.
     **/
    private static Tree currentPackage = null;


    /**
     ** Complain about the current package.  Displays a header if one
     ** has not already been printed.  Sets error.
     **/
    public static void complain(String complaint) {
	displayHeader(currentPackage);
	println("  "+complaint);
	error = true;
    }

    /**
     ** Warn about the current package.  Displays a header if one has
     ** not already been printed.<p>
     **
     ** Does not set error.  No output is displayed if noWarn is set.<p>
     **/
    public static void warn(String complaint) {
	if (noWarn)
	    return;
	displayHeader(currentPackage);
	println("  Warning: "+complaint);
    }

    /**
     ** Display debugging information about the current package.  Displays a
     ** header if one has not already been printed.<p>
     **
     ** Does not set error and ignores noWarn.<p>
     **/
    public static void debug(String complaint) {
	displayHeader(currentPackage);
	println("  "+complaint);
    }


    /**
     ** Display the human-readable location of a node.<p>
     **
     ** Precondition: complain/warn/debug has already been called for
     ** the current package.<p>
     **/
    public static void listLocation(Tree node) {
	println("    " + ((GenericFile)node.data).getHumanName());
    }


    /***************************************************
     *                                                 *
     * Checking a package:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Check the contents of a package.<p>
     **
     ** Uses the current program settings.<p>
     **/
    public static void checkPackage(Tree P) {
	// Set up complain/warn/debug utilities:
	error = false;
	currentPackage = P;

	if (verbosity>0)
	    displayHeader(P);

	checkSourceDuplication();	// Phase I
	if (!error)
	    checkSources();		// Phase II
	if (!error)
	    checkBinaries();		// Phase III
    }


    /**
     ** Checks the current package to see if any of its units have more
     ** than one source files.  Complains about all such units.
     **/
    public static void checkSourceDuplication() {
	if (verbosity>1)
	    debug("Phase I: checking for duplicate source files");

	Enumeration E = PkgTree.components(currentPackage, ".java");
	while (E.hasMoreElements()) {
	    Tree S = (Tree)E.nextElement();
	    String name = Extension.getBasename(S.getLabel());
	    int count = ((UnionTree)S).countDuplicates();

	    if (count>1) {
		complain(name + ": duplicate source files exist:");
		Tree[] duplicates = ((UnionTree)S).duplicates();
		for (int i=0; i<duplicates.length; i++)
		    listLocation(duplicates[i]);
	    }
	}
    }


    /***************************************************
     *                                                 *
     * Mapping units to source files:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Running checkSources sets sources to a map from (simple)
     ** class/interface names to JCheck_SourceInfo objects.
     **/
    public static Hashtable sources = null;

    /**
     ** Checks the current package's source files for various errors.
     ** Complains about all errors found.  Throws Fail if
     ** it has to complain.  Fills in sources for the current package.
     **/
    public static void checkSources() {
	sources = new Hashtable();

	if (verbosity>1)
	    debug("Phase II: checking source files");

	Enumeration E = PkgTree.components(currentPackage, ".java");
	while (E.hasMoreElements()) {
	    Tree S = (Tree)E.nextElement();
	    String name = Extension.getBasename(S.getLabel());

	    if (verbosity>2)
		debug("  checking " + S.getLabel());

	    // Check for a subpackage with the same name:
	    Tree P = currentPackage.getChild(name);
	    if (P != null) {
		complain( name + ": is both a class/interface and a subpackage name:");
		listLocation(P);
		listLocation(S);
		continue;
	    }

	    // If that is not true, scan the file and add its
	    // information to sources, generating complaints as needed:
	    checkSource(S);
	}
    }

    /***************************************************
     *                                                 *
     * Scanning a source file:			       *
     *                                                 *
     ***************************************************/

    public static Tree currentSource = null;
    public static String declaredPackageName = null;


    // The class JCheck_SourceInfo would be here as the static class
    // SourceInfo if inner classes were being used.<p>
    //
    // See its comments.<p>


    public static void checkSource(Tree S) {
	GenericFile srcFile = (GenericFile) S.data;
	currentSource = S;
	declaredPackageName = PkgTree.rootPackageName;
	String name = Extension.getBasename(S.getLabel());

	try {
	    InputStream stream  = srcFile.getInputStream();

	    ParseFindTypes parser = new ParseFindTypes(stream);

	    try {
		parser.CompilationUnit();
	    } finally {
		try { stream.close(); } catch (IOException e) {
		    complain("I/O error: " + e.getMessage());
		};
	    }

	    if (verbosity>3)
		debug("    found package " + declaredPackageName + ";");

	    /*
	     * Check if a unit of our name has not been declared:
	     */
	    if (!sources.containsKey(name))
		complain(srcFile.getHumanName()
		    + ": a unit " + name + " must be declared in this file");

	    /*
	     * Check that we have declared our package name correctly:
	     */
	    if (!declaredPackageName
		    .equals(PkgTree.getPackageName(currentPackage)))
		complain(identifyUnit(name)
		    + ": is declared to be in the package "
		    + declaredPackageName
		    + " rather than in the correct package "
		    + PkgTree.getPackageName(currentPackage));

	} catch (IOException e) {
	    complain("I/O error: " + e.getMessage());
	} catch (ParseError e) {
	    complain(srcFile.getHumanName()
		+ ": error encountered while parsing file");
	}
    }



    public static void registerSourceUnit(String unitname,
				boolean isInterface, boolean isPublic) {
	if (verbosity>3)
	    debug("    found " + (isPublic ? "public " : "")
		+ (isInterface ? "interface " : "class ")
		+ unitname);

	/*
	 * Complain & exit if this unit is already declared previously.
	 */
	if (sources.containsKey(unitname)) {
	    complain("unit " + unitname + " declared multiple times in:");
	    listLocation(currentSource);
	    listLocation(((JCheck_SourceInfo)sources.get(unitname))
				.sourceNode);
	    return;
	}

	// Otherwise, record the information in sources:
	sources.put(unitname, new JCheck_SourceInfo(currentSource,
					isInterface, isPublic));

	/*
	 * Is there a subpackage with the same name?
	 */
	Tree P = currentPackage.getChild(unitname);
	if (P != null) {
	    complain(unitname +
		  ": is both a class/interface and a subpackage name:");
		listLocation(P);
		listLocation(currentSource);
	}

	/*
	 * Are we declared public in a file not of the same name?
	 */
	if (isPublic) {
	    if (!(unitname+".java").equals(currentSource.getLabel()))
		complain(identifyUnit(unitname) + ": may not be"
		    + " declared public unless placed in a file named "
		    + unitname + ".java");
	}
    }

    public static String identifyUnit(String unitname) {
//	if (unitname+".java" == currentSource.getLabel())
//	    return "unit " + unitname;

	return "unit " + unitname + " in "
	    + ((GenericFile)currentSource.data).getHumanName();
    }


    /***************************************************
     *                                                 *
     *						       *
     *                                                 *
     ***************************************************/

    /**
     ** Checks the current package's binary files for various errors.
     ** Complains about all errors found.
     **/
    public static void checkBinaries() {
	if (verbosity>1)
	    debug("Phase III: checking binary files");

	Enumeration E = PkgTree.components(currentPackage, ".class");
	while (E.hasMoreElements()) {
	    Tree S = (Tree)E.nextElement();
	    String name = Extension.getBasename(S.getLabel());

	    if (verbosity>3)
		debug("  checking " + S.getLabel());

	    // Check for a subpackage with the same name:
	    Tree P = currentPackage.getChild(name);
	    if (P != null) {
		error = true;
		complain(name + ": is both a class/interface and a subpackage name:");
		listLocation(P);
		listLocation(S);
	    }

	    // ...
	}
    }


    /***************************************************
     *                                                 *
     * Command line parsing:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Should we supress warning messages?
     **/
     public static boolean noWarn = false;

    /**
     ** Our verbosity level.  The more non-zero, the more detailed
     ** information is to be displayed.  0 is the default.
     **/
     public static int verbosity = 0;


    // The list of our command line arguments
    static String[] arguments;

    // An index into arguments, pointing to the next argument to be
    // processed:
    static int argIndex = 0;


    /** Print out our usage information on System.err **/
    public static void usage() {
	System.err.println(
	    "jcheck: usage: [-nRv] [-classpath <classpath>] <package_name>...");
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
	    case 'n':
		noWarn = true;
		return true;

	    case 'R':
		recurse = true;
		return true;

	    case 'v':
		verbosity++;
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
    static void handleOption(String arg) {
	if (arg.equals("-classpath")) {
	    if (argIndex<arguments.length)
		Resolve.set(arguments[argIndex++], false);

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

    /** The main procedure for the jcheck command **/
    public static void main(String[] args) {
	// Setup processing of arguments:
	arguments = args;

	// Setup initial namespace:
	Resolve.init(false);


	// Process any options:
	argIndex=0;
	String arg;
	while (argIndex<arguments.length && (arg=args[argIndex]).length()>0
		&& arg.charAt(0)=='-') {
	    argIndex++;
	    handleOption(arg);
	}

	// Process the remaining arguments as package names:
	if (argIndex==args.length)
	    recurse = true;
	handlePackageNames(args, argIndex);
    }
}
