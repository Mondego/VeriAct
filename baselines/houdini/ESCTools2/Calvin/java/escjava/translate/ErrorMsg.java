/* Copyright Hewlett-Packard, 2002 */

package escjava.translate;

import java.io.*;

import javafe.util.Assert;
import javafe.util.Info;
import javafe.util.ErrorSet;
import javafe.util.Location;
import javafe.util.Set;

import javafe.ast.RoutineDecl;

import escjava.Main;
import escjava.prover.*;
import escjava.ast.TagConstants;


/**
 ** Provides printing of error messages to the user.
 **/

public final class ErrorMsg {
  
    /** Prints an error message for proof obligation <code>name</code>,
     * where <code>labelList</code> and <code>counterexampleContext</code>
     * are labels and counterexample from Simplify.
     * Error messages are printed to <code>out</code>.
     **/
  
    public static void print(String name,
			     SList labelList,
			     SList counterexampleContext,
			     /*@ non_null */ RoutineDecl rd,
			     /*@ non_null */ Set directTargets,
			     /*@ non_null */ PrintStream out) {
	try {
	    int cLabels = labelList.length();
	    int iErrorLabel = -1;
	    for (int i = 0; i < cLabels; i++) {
		String s = labelList.at(i).getAtom().toString();
		if (isErrorLabel(s)) {
		    printErrorMessage(s, counterexampleContext, rd, directTargets, out);
		    iErrorLabel = i;
		    break;
		}	  
	    }
	    if (iErrorLabel == -1) {
		//@ unreachable
		ErrorSet.error("Unknown cause!");
	    }

	    // print the execution trace info if requested
	    if (Main.traceInfo > 0) {
		// copy the trace labels to a String array
		String traceLabels[] = new String[cLabels];
		int index = 0;
		for (int i = 0; i < cLabels; i++) {
		    String s = labelList.at(i).getAtom().toString();
		    if (isTraceLabel(s)) {
			traceLabels[index] = s.substring(6);
			index++;
		    }
		}
		if (index > 0) {
		    sortTraceLabels(traceLabels, index);
		    out.println("Execution trace information:");
		    for (int i = 0; i < index; i++)
			printTraceInfo(traceLabels[i], out);
		    System.out.println();
		}
	    }
      
	    if (Main.counterexample) {
		out.println("Counterexample context:");
		SList prunedCC = pruneCC(counterexampleContext);
		int n = prunedCC.length();
		for (int i = 0; i < n; i++) {
		    out.print("    ");
		    prunedCC.at(i).prettyPrint(out);
		    out.println();
		}
		out.println();
	    }     

	    if (Info.on || Main.pcc) {
		Assert.notFalse(counterexampleContext.length() > 1 &&
				counterexampleContext.at(0).toString().equals("AND"));
		out.println("Full counterexample context:");
		int n = counterexampleContext.length();
		for (int i = 2; i < n; i++) {
		    out.print("    ");
		    counterexampleContext.at(i).print(out);
		    out.println();
		}
	    }

	    boolean userLabels = false;
	    for (int i = 0; i < cLabels; i++) {
		String label = labelList.at(i).getAtom().toString();
		if (i == iErrorLabel || label.startsWith("vc.") ||
		    (Main.traceInfo > 0 && isTraceLabel(label))) {
		    continue;
		}
		if (!userLabels) {
		    out.println("Counterexample labels:");
		    userLabels = true;
		}
		out.print("    " + label);
	    }
	    if (userLabels) {
		out.println();
		out.println();
	    }
	} catch (escjava.prover.SExpTypeError s) {
	    Assert.fail("unexpected S-expression exception");
	}
	printSeparatorLine(out);
    }

    public static void printSeparatorLine(/*@ non_null */ PrintStream out) {
	if (!Main.quiet) {
	    out.println("------------------------------------------------------------------------");
	}
    }

    /** Returns whether or not <code>s</code> is string that indicates
     * which ESC/Java check the program violates.
     *
     * Requires <code>s</code> to be a label output by an ESC/Java
     * run of Simplify.
     **/

    public static boolean isErrorLabel(String s) {
	return s.indexOf('@') != -1;
    }

    public static boolean isMoverLabel(String s) {
	int k = s.indexOf('@');
	Assert.notFalse(k != -1);
	s = s.substring(0, k);

	k = s.indexOf(':');
	if (k != -1) {
	    s = s.substring(0, k);
	}

	k = s.indexOf('/');
	if (k != -1) {
	    s = s.substring(0, k);
	}
    
	int tag = TagConstants.checkFromString(s);

	if (tag == TagConstants.CHKRIGHTMOVER ||
	    tag == TagConstants.CHKLEFTMOVER) {
	    return true;
	} else {
	    return false;
	}	
    }

    /** Returns whether or not <code>s</code> is string that indicates
     * information about the execution trace in the counterexample context.
     *
     * Requires <code>s</code> to be a label output by an ESC/Java
     * run of Simplify.
     **/

    static boolean isTraceLabel(/*@ non_null */ String s) {
	return s.startsWith("trace.");
    }


    /** Parses <code>s</code> and prints a nice error message to
     * <code>out</code>.
     **/
  
    private static void printErrorMessage(String s,
					  SList counterexampleContext,
					  /*@ non_null */ RoutineDecl rd,
					  /*@ non_null */ Set directTargets,
					  /*@ non_null */ PrintStream out)
	throws SExpTypeError {

	if (counterexampleContext == null) {
	    counterexampleContext = SList.make();
	}

	int k = s.indexOf('@');
	Assert.notFalse(k != -1);
	int locUse = getLoc(s, k+1);
	s = s.substring(0, k);


	k = s.indexOf(':');
	boolean hasAssocDecl = false;
	int locDecl =  Location.NULL;
	if (k != -1) {
	    hasAssocDecl = true;
	    locDecl = getLoc(s, k+1);
	    s = s.substring(0, k);
	}


	k = s.indexOf('/');
	int auxID = -1;  // -1 means none
	if (k != -1) {
	    auxID = toNumber(s, k+1);
	    s = s.substring(0, k);
	}
    
	int tag = TagConstants.checkFromString(s);

	String msg = chkToMsg( tag, hasAssocDecl );

	ErrorSet.warning( locUse, 
			  msg+" (" + TagConstants.toString(tag) + ")" );

	if( locDecl != Location.NULL && !Location.isWholeFileLoc(locDecl)) {
	    System.out.println("Associated declaration is "
			       + Location.toString(locDecl) + ":");
	    ErrorSet.displayColumn(locDecl, assocDeclClipPolicy);
	}

	switch (tag) {
	case TagConstants.CHKLOOPOBJECTINVARIANT:
	case TagConstants.CHKOBJECTINVARIANT:
	    displayInvariantContext(counterexampleContext, out);
	}

	if (Main.suggest) {
	    Object auxInfo;
	    if (auxID == -1) {
		auxInfo = null;
	    } else {
		auxInfo = AuxInfo.get(auxID);
	    }
	    String sug = Suggestion.generate(tag, auxInfo, rd, directTargets, 
					     counterexampleContext, locDecl);
	    if (sug == null) {
		sug = "none";
	    }
	    System.out.println("Suggestion [" + Location.toLineNumber(locUse)+
			       "," + Location.toColumn(locUse) + "]: " + sug);
	}
    }



    
    private static final AssocDeclClipPolicy assocDeclClipPolicy =
	new AssocDeclClipPolicy();


    private static String chkToMsg( int tag, boolean hasAssocDecl ) {
	String r = null;
	// Finally, describe the error
	switch (tag) {
	case TagConstants.CHKARITHMETIC:
	    r = ("Possible division by zero");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKARRAYSTORE:
	    r = ("Type of right-hand side possibly not a subtype of " +
		 "array element type");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKASSERT:
	    r = ("Possible assertion failure");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKCLASSCAST:
	    r = ("Possible type cast error");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKCODEREACHABILITY:
	    r = ("Code marked as unreachable may possibly be reached");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKDEFINEDNESS:
	    r = ("Read of variable when its value may be meaningless");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKGLOBALINVARIANT:
	    r = ("Global invariant possibly does not hold"); 
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKINDEXNEGATIVE:
	    r = ("Possible negative array index");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKINDEXTOOBIG:
	    r = ("Array index possibly too large");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKLOOPINVARIANT:
	    r = ("Loop invariant possibly does not hold");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKLOOPOBJECTINVARIANT:
	    r = ("Object invariant possibly does not hold on loop boundary");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKINITIALIZATION:
	    r = ("Possible dynamic use before meaningful assignment of local " +
		 "variable");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKLOCKINGORDER:
	    r = ("Possible deadlock");

	    break;
	case TagConstants.CHKMODIFIES:
	    r = ("Possible violation of modifies clause");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKNEGATIVEARRAYSIZE:
	    r = ("Possible attempt to allocate array of negative length");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKNONNULL:
	    r = ("Possible assignment of null to variable declared non_null");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKNONNULLINIT:
	    r = ("Field declared non_null possibly not initialized");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKNULLPOINTER:
	    r = ("Possible null dereference");
	    Assert.notFalse(!hasAssocDecl);
	    break;
	case TagConstants.CHKOBJECTINVARIANT:
	    // It would be nice to split this error into two:
	    //  *  Possible violation of object invariant at end of body (InvPost)
	    //  *  Possible violation of object invariant before call (InvPre)
	    r = ("Possible violation of object invariant");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKOWNERNULL:
	    r = ("Owner ghost field of constructed object possibly non-null");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKOTI:
	    r = ("OTI possibly does not hold"); 
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKOTIREFL:
	    r = ("OTI may not be reflexive"); 
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKOTISTUTTER:
	    r = ("{Stutter;OTI} may not be transitive"); 
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKOTITRANS:
	    r = ("OTI may not be transitive"); 
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKPERFORMS:
	    r = ("\"Performs\" predicate possibly does not hold"); 
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKPERFORMSEND:
	    r = ("\"Performs\" predicate possibly does not hold at the end"); 
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKPOSTCONDITION:
	    r = ("Postcondition possibly not established");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKPRECONDITION:
	    r = ("Precondition possibly not established");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKSHARING:
	    r = ("Possible race condition");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKWRITABLE:
	    r = ("Write of variable when disallowed");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKRIGHTMOVERENV:
	    r = ("Violates assumption of another thread for right mover");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKLEFTMOVERENV:
	    r = ("Violates assumption of another thread for left mover");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKGUARDED:
	    r = ("Access of variable when not allowed");
	    Assert.notFalse(hasAssocDecl);
	    break;
	case TagConstants.CHKUNEXPECTEDEXCEPTION:
	    r = ("Possible unexpected exception");
	    // "strLocDecl" may or may not be null
	    break;
	case TagConstants.CHKCONSTRUCTORLEAK:
	case TagConstants.CHKINITIALIZERLEAK:
	case TagConstants.CHKUNENFORCEBLEOBJECTINVARIANT:
	case TagConstants.CHKWRITABLEDEFERRED:
	case TagConstants.CHKMODIFIESEXTENSION:
	    // this  a syntactic warning, not a semantic check
	    //@ unreachable
	    Assert.fail("unexpected error name tag");
	    r = TagConstants.toString(tag);
	    break;
	default:
	    //@ unreachable
	    Assert.fail("Bad tag");
	    break;
	}
	return r;
    }


    /** Sort the trace labels.  Labels are assumed to have the form
	trace.Name^n,r.c[#i]
	Here [] denotes optional (zero or one occurrence), n is an ordering
	number used for the sorting, r is a row number, c is a column number, and
	i is an iteration number within a loop.  Only n is used for sorting;
	the rest are only for printing purposes.
    **/
    private static void sortTraceLabels(/*@ non_null */ String[] labels, 
					int size) {
	// create an array containing the n-value (see comment above) of 
	// each label
	int traceLocs[] = new int[size];
	for (int i = 0; i < size; i++) {
	    int caret = labels[i].indexOf("^");
	    Assert.notFalse(caret != -1);
	    int comma = labels[i].indexOf(",");
	    Assert.notFalse(comma != -1);
	    Assert.notFalse(caret < comma);
	    String n = labels[i].substring(caret + 1, comma);
	    traceLocs[i] = toNumber(n, 0);
	}
	
	// bubble sort the labels
	int sizeMinusOne = size - 1;
	for (int i = 0; i < sizeMinusOne; i++) {
	    for (int j = sizeMinusOne; i < j; j--) {
		if (traceLocs[j] < traceLocs[j-1]) {
		    // swap the j-1 and jth elements in both arrays
		    int tempLoc;
		    String tempLabel;
		    tempLoc = traceLocs[j];
		    tempLabel = labels[j];
		    traceLocs[j] = traceLocs[j-1];
		    labels[j] = labels[j-1];
		    traceLocs[j-1] = tempLoc;
		    labels[j-1] = tempLabel;
		}
	    }
	}
    }


    /** Parses <code>s</code> and prints execution trace information to
     * <code>out</code>.
     **/
  
    private static void printTraceInfo(/*@ non_null */ String s,
				       /*@ non_null */ PrintStream out) {
	// first parse the String to get its location (the last row-col pair
	//   in its sequence of numbers)
	int caret = s.indexOf("^");
	Assert.notFalse(caret != -1);
	int comma = s.indexOf(",");
	Assert.notFalse(comma != -1);
	String temp = s.substring(comma + 1);
	int hash = temp.indexOf("#");
	int iteration = -1;
	if (hash != -1) {
	    iteration = toNumber(temp, hash + 1);
	    temp = temp.substring(0, hash);
	}
	int loc = getLoc(temp,0);
	String location = Location.toString(loc);

	s = s.substring(0,caret);
	if (s.equals("RoutineException")) {
	    out.println("    Routine call returned exceptionally in " +
			location + ".");
	} else if (s.equals("FirstOpOnly")) {
	    out.println("    Short circuited boolean operation in " +
			location + ".");
	} else if (s.equals("LoopIter")) {
	    if (iteration == -1) {
		out.println("    At the top of arbitrary loop iteration at "
			    + location + ".");		
	    } else {
		out.println("    Reached top of loop after " + iteration +
			    " iteration"
			    + (iteration == 1 ? "" : "s") + " in "
			    + location + ".");
	    }
	} else if (s.equals("FinallyBegin")) {
	    out.println("    Abnormal entry to finally clause at " +
			location + ".");
	} else if (s.equals("FinallyEnd")) {
	    out.println("    Resuming abnormal execution path after finally clause at " +
			location + ".");
	} else if (s.equals("CallReturn")) {
	    out.println("    Returned from inlined call at " + location + ".");
	} else {
	    if (s.equals("Then") || s.equals("Else")) {
		s += " branch";
	    } else if (s.equals("Switch")) {
		s += " case";
	    } else if (s.equals("ImplicitReturn")) {
		s = "implicit return";
	    }
	    
	    out.println("    Executed " + s.toLowerCase() +" in " +
			location + ".");
	}
    }


    /** Converts string <code>s</code>, beginning at index <code>k</code>,
     * to a pair of locations, represented as a String.
     **/

    private static String getLocation(String s, int k) {
	String suffix = s.substring(k);
	return UniqName.suffixToString(suffix);
    }

    /** Converts string <code>s</code>, beginning at index <code>k</code>,
     * into a location.
     **/

    private static int getLoc(String s, int k) {
	String suffix = s.substring(k);
	return UniqName.suffixToLoc(suffix);
    }

    /** Converts the substring beginning at <code>k</code> of <code>s</code>
     * into a number.
     **/
  
    private static int toNumber(String s, int k) {
	int n = 0;
	while (k < s.length()) {
	    n = 10*n + s.charAt(k) - '0';
	    k++;
	}
	return n;
    }

    private static void displayInvariantContext(/*@ non_null */ SList counterexampleContext,
						/*@ non_null */ PrintStream out)
	throws SExpTypeError {
	if (Main.plainWarning)
	    return;

	boolean headerDisplayed = false;

	int n = counterexampleContext.length();	
	for (int i = 0; i < n; i++) {
	    SExp contextLine = counterexampleContext.at(i);
	    if (contextLine.toString().indexOf("brokenObj")== -1)
		continue;

	    if (!headerDisplayed) {
		out.println("Possibly relevant items from the counterexample context:  ");
		headerDisplayed = true;
	    }

	    out.print("  ");
	    contextLine.prettyPrint(out);
	    System.out.println();
	}

	if (headerDisplayed) {
	    System.out.println("(brokenObj* refers to the object for"
			       + " which the invariant is broken.)");
	    System.out.println();
	}
    }


    /** Prune out s-expressions from the counterexample context that are 
	almost certainly irrelevant. **/
    //@ requires cc != null
    //@ ensures RES != null
    private static SList pruneCC(SList cc) throws SExpTypeError {
	SList copy = SList.make();
	SExp cur;
	String curS;
	int n = cc.length();
	for (int i = 0; i < n; i++) {
	    cur = cc.at(i);
	    curS = cur.toString();
	    if (!cur.isList())
		continue;
	    // get rid of s-expressions that either tell you the type
	    // of a variable or that one type subtypes another
	    if ((curS.startsWith("(is ") || curS.startsWith("(<: ")) &&
		(curS.indexOf("|T_") != -1))
		continue;
	    if(curS.indexOf("(store ") != -1 ||
	       curS.indexOf("(classDown ") != -1 ||
	       curS.indexOf("(asChild ") != -1 ||
	       curS.indexOf("(asLockSet ") != -1 ||
	       curS.indexOf("(asField ") != -1 ||
	       curS.indexOf("(asElems ") != -1 ||
	       curS.startsWith("(isAllocated ") ||
	       curS.startsWith("(DISTINCT "))
		continue;

	    copy = copy.addEnd(cur);
	}
	return copy;
    }


    /********** Houdini Support ***********/

    /**
     * return the string rep of the location loc when it is used
     * as an associated decl location.  fileNames is the CorrelatedReader
     * file mappings.
     */
    private static String declToFileLocStr(String loc, String fileNames[]) {
	int p = loc.indexOf(".");
	String fNum = loc.substring(0, p);
	int file = Integer.parseInt(fNum);
	loc = loc.substring(p+1);

	p = loc.indexOf(".");
	String line = loc.substring(0,p);
	loc = loc.substring(p+1);

	//	return fileNames[file] + ":"+line;
	//	return "\"" + fileNames[file] + "\"" + ":"+loc;
	return fileNames[file] + " " + line + " " + loc;
    }

    /**
     * return the string rep of the location loc when it is used
     * as a use.  fileNames is the CorrelatedReader
     * file mappings.
     */
    private static String useToFileLocStr(String loc, String fileNames[]) {
	int p = loc.indexOf(".");
	String fNum = loc.substring(0, p);
	int file = Integer.parseInt(fNum);
	loc = loc.substring(p+1);
	p = loc.indexOf(".");
	return fileNames[file] + ":"+loc.substring(0,p);
    }

    /**
     * Prints a houdini error message to out.
     */
    public static void houdiniPrint(String label,
			     /*@ non_null */ PrintStream out,
			     String fileNames[]) {
	try {
	    if (isErrorLabel(label)) {
		houdiniPrintErrorMessage(label, out, fileNames);
	    }
	} catch (escjava.prover.SExpTypeError s) {
	    Assert.fail("unexpected S-expression exception");
	}
    }
    
    /** Parses <code>s</code> and prints an error message for the
     ** houdini log to out.
     **/
    private static void houdiniPrintErrorMessage(String s,
					  /*@ non_null */ PrintStream out,
					  String map[])
	throws SExpTypeError {

	
	int k = s.indexOf('@');
	Assert.notFalse(k != -1);
	String locUse = useToFileLocStr(s.substring(k+1), map);
	s = s.substring(0, k);

	k = s.indexOf(':');
	String strLocDecl = null;
	if (k != -1) {
	    strLocDecl = declToFileLocStr(s.substring(k+1), map);
	    s = s.substring(0, k);
	}

	k = s.indexOf('/');
	if (k != -1) {
	    s = s.substring(0, k);
	}
    
	int tag = TagConstants.checkFromString(s);
	String msg = "Warning: " + chkToMsg( tag, strLocDecl!=null );
	out.println(strLocDecl + " " + locUse + ": " + msg);
    }

    

}
