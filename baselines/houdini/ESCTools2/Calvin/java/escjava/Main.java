/* Copyright Hewlett-Packard, 2002 */

package escjava;


import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Vector;
import java.io.*;

import javafe.ast.*;
import escjava.ast.*;
import escjava.ast.EscPrettyPrint;
import escjava.ast.TagConstants;

import escjava.backpred.FindContributors;

import javafe.reader.StandardTypeReader;
import escjava.reader.EscTypeReader;

import javafe.parser.PragmaParser;

import escjava.sp.*;

import escjava.translate.*;
//import escjava.pa.*;

import javafe.tc.TypeSig;
import escjava.tc.TypeCheck;
import escjava.tc.Types;

import escjava.prover.*;

import javafe.util.*;


/**
 ** Top level control module for ESC for Java. <p>
 **
 ** This class (and its superclasses) handles parsing
 ** <code>escjava</code>'s command-line arguments and orchestrating the
 ** other pieces of the front end and escjava to perform the requested
 ** operations.<p>
 **
 ** @see javafe.Tool
 ** @see javafe.SrcTool
 **/

public class Main extends javafe.SrcTool {

     /** Our version number - Changed by Sanjit **/
     public final static String version = "1.2.3d, 13 June 2001";


    /***************************************************
     *                                                 *
     * Global escjava flags:			       *
     *                                                 *
     ***************************************************/

    /*
     * These are set by the command-line switches of the same name.
     * They default to false (unset).
     */

    public static boolean quiet	= false;
    public static boolean suggest = false;

    public static boolean pjt	= false; // print java w/ types
    public static boolean nvu	= false;
    public static boolean pgc	= false;
    public static boolean pdsa	= false;
    public static boolean pcc	= false;
    // a pruned, pretty printed version of pcc:
    public static boolean counterexample = false;
    public static boolean stats = false;
    public static boolean plainWarning = false;

    // trace info: (0) no, (1) default, (2) verbose
    public static int traceInfo = 1;
    //@ invariant 0 <= traceInfo && traceInfo < 3;

    public static boolean spvc = true;
    public static boolean dsa = true;
    //@ invariant spvc ==> dsa  // spvc requires dsa for soundness
    public static boolean passify = false;
    public static boolean wpp = false;
    public static boolean useDefpred = false;
    public static int wpnxw = 0;
    public static int namePCsize = 0;

    public static boolean peepOptE=true;
    public static boolean peepOptGC=true;
    public static boolean lazySubst = false;
    public static boolean mergeInv=false;

    public static boolean useAllInvPostCall = false;
    public static boolean useAllInvPostBody = false;
    public static boolean useAllInvPreBody = false;
    public static boolean allocUseOpt = true;

    public static final int LOOP_FAST = 0;
    public static final int LOOP_SAFE = 1;
    public static final int LOOP_FALL_THRU = 2;
    public static int loopTranslation = LOOP_FAST;
    //@ invariant LOOP_FAST <= loopTranslation && loopTranslation <= LOOP_FALL_THRU;

    // The default loop unrolling is: -loop 1.5 
    public static int loopUnrollCount = 1;
    public static boolean loopUnrollHalf = true;

    public static boolean predAbstract = false;
    public static boolean inferPredicates = false;

    public static boolean noDirectTargetsOpt = false;
    public static boolean nestQuantifiers = false;

    public static boolean lastVarUseOpt = true;

    public static boolean useIntQuantAntecedents = false;

    public static boolean excuseNullInitializers = false;

    /** The following values are used: <br>
      *   0-never, 1-atomic (default), 2-always
      **/
    public static int strongAssertPost = 1;

    public static boolean showCallDetails = true;
    public static boolean showLoopDetails = false;
    public static boolean showYieldDetails = false;
    public static boolean bubbleNotDown=false;
    public static int startLine = -1;    // starting line # for processing
    public static int vclimit = 500000;

    // flags primarily used by Houdini
    public static boolean printAssumers = false;
    public static boolean noPeepOptGCAssertFalse = false;
    public static boolean assertContinue = false;
    public static boolean trackReadChars = true;
    public static boolean guardedVC = false;
    public static String guardedVCDir;
    public static String guardedVCFileNumbers = "files.txt";
    public static String guardedVCGuardFile = "guards.txt";
    public static String guardedVCPrefix = "G_";
    public static String guardedVCFileExt = "sx";
    public static Set guardVars = new Set();  
    public static String ClassVCPrefix = "*Class*";
    public static String MethodVCPrefix = "*Method*";

    public static Set ignoreAnnSet = null;

    public static boolean printCompilationUnitsOnLoad = false;

    // filter invariants and axioms
    public static boolean filterInvariants = false;
    // filter method specifications
    public static boolean filterMethodSpecs = false;

    public static Set routinesToCheck = null;  // null means "check everything"

    // do the inlined constructor experiment?
    public static boolean inlineConstructors = false;
    // do some other inlining experiment, using at least one of the two
    // inline depth flags?
    public static boolean inlineDepthFlags = false;
    // when checking a constructor of a class T, inline every call
    // this.m(...) transitively from within the constructor, whenever m
    // statically resolves to a non-overridable instance method defined in T.
    public static boolean inlineFromConstructors = false;

    // Check code in a multithreaded context
    public static boolean multiThreaded = false;
    // Whether reduction should be used to check multithreaded code
    public static boolean reduce = false;
    // Version of OTI inference to be used
    public static boolean otiInferOpt = false; 
    // Whether to havoc "final" fields
    public static boolean havocFinal = false;
    // Whether to use invariants declared only in the class
    public static boolean restrictScope = false;
    // Whether to put a yield after an access (or before)
    public static boolean yieldAfter = false;
    // print out info about reduction
    public static boolean showReduce = false;


    public static String sxLog = null;

    // The following switch hardcodes whether or not also_requires is to
    // be part of the annotation language
    public static boolean allowAlsoRequires = true;

    /**
     ** Number of stages to run.  The stages currently in order are:
     **     1. loading, parsing, and type checking
     **     2. generating the type-specific background predicate
     **	    3. translating from Java to GC
     **	    4. translating from GC to DSA
     **     5. generating the VC from the GC
     **     6. calling the theorem prover to verify the VC<p>
     **
     ** defaults to running all stages (6); must be positive.<p>
     **
     ** -nocheck sets <code>stages</code> to run all but the last
     ** stage (5).  The -stages option can be used to set
     ** <code>stages</code> to a particular value.<p>
     **/
    public static int stages	= 6;


    //* Which file to obtain the universal background predicate from.
    public static String univBackPredFile = null;


    /***************************************************
     *                                                 *
     * Generating an options message:		       *
     *                                                 *
     ***************************************************/

    /**
     ** Return the name of this tool.  E.g., "ls" or "cp".<p>
     **
     ** Used in usage and error messages.<p>
     **/
    public String name() { return "escjava"; }

    /**
     ** Print option option information to
     ** <code>System.err</code>. <p>
     **/
    public void showOptions() {
      super.showOptions();
      /* Note:  The following should list all of the "public" options, not
       * debugging options that are present only experimental versions at SRC.
       */
      System.err.println("  -counterexample -loopSafe -loop <iterCount>[.0|.5] -nocheck");
      System.err.println("  -notrace -nowarn <category> -plainwarning -quiet");
      System.err.println("  -routine <routineId> -routine <fullyQualifiedRoutineSignature>");
      System.err.println("  -routineIndirect <routineFile> -start <line#> -suggest");
      System.err.println("  -vclimit <n> -warn <category>");

      /************************************
      System.err.println("  -quiet -plainwarning -nowarn <category> -warn <category>");
      System.err.println("  -start <line#> -nocheck -stages <n> -vclimit <n> -suggest");
      System.err.println("  -notrace -pcc -pgc -pjt -pdsa -stats -routineIndirect <routineFile>");
      System.err.println("  -routine <routineId> -routine <fullyQualifiedRoutineSignature>");
      System.err.println("  -hidecall -showloop -nvu -noEpeepopt -noGCpeepopt -lazySubst -mergeinv");
      System.err.println("  -nospvc -nodsa -bubbleNotDown -noAllocUseOpt -passify -wpp -useDefpred -wpnxw <int> -namePCsize <int>");
      System.err.println("  -useAllInvPostCall -useAllInvPostBody -useAllInvPreBody");
      System.err.println("  -noDirectTargetsOpt -nestQuantifiers -noLastVarUseOpt");
      System.err.println("  -useIntQuantAntecedents -counterexample -sxLog <file>");
      System.err.println("  -strongAssertPost{Never,Atomic,Always} -backpredfile <file>");
      System.err.println("  -loop <iterCount>[.0|.5] -loopFallThru -printAssumers");
      System.err.println("  -guardedVC <dirname> -ignoreAnnFile <file>");
      System.err.println("  -inlineConstructors -inlineCheckDepth <n> -inlineDepthPastCheck <n>");
      System.err.println("  -assertContinue -noPeepOptGCAssertFalse -noTrackReadChars");
      System.err.println("  -filterInvariants -filterMethodSpecs -inlineFromConstructors -verboseTrace");
      ************************************/
    }


    /***************************************************
     *                                                 *
     * Option processing:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Process next tool option. <p>
     **
     ** See <code>Tool.processOption</code> for the complete
     ** specification of this routine.<p>
     **/
    public int processOption(String option, String[] args, int offset) {
	if (option.equals("-nocheck")) {
	    stages = 5;		// run all but last stage
	    return offset;
	} else if (option.equals("-stages")) {
	    if (offset == args.length) {
		usage();
		System.exit(usageError);
	    }
	    try {
	        stages = new Integer(args[offset]).intValue();
		if (stages<1)
		   throw new NumberFormatException();
	    } catch (NumberFormatException e) {
	        badOptionUsage("illegal argument to -stages: " +
			       args[offset]);
	    }
	    return offset+1;
	} else if (option.equals("-start")) {
	    if (offset>=args.length) {
		usage();
		System.exit(usageError);
	    }
	    try {
	        startLine = new Integer(args[offset]).intValue();
	    } catch (NumberFormatException e) {
	        badOptionUsage("illegal argument to -start: " + args[offset]);
	    }
	    return offset+1;
	} else if (option.equals("-vclimit")) {
	    if (offset>=args.length) {
		usage();
		System.exit(usageError);
	    }
	    try {
	        vclimit = new Integer(args[offset]).intValue();
	    } catch (NumberFormatException e) {
	        badOptionUsage("illegal argument to -vclimit: " + args[offset]);
	    }
	    return offset+1;
	} else if (option.equals("-inlineFromConstructors")) {
	  inlineFromConstructors = true;
	  return offset;
	} else if (option.equals("-inlineCheckDepth")) {
	    inlineDepthFlags = true;
	    // can't use this along with the -inlineConstructors flag
	    if (inlineConstructors)
		badOptionUsage("can't use -inlineConstructors in combination with either -inlineCheckDepth or -inlineDepthPastCheck");
	    if (offset == args.length) {
		usage();
		System.exit(usageError);
	    }
	    try {
	        gctranslator.inlineCheckDepth =
		    new Integer(args[offset]).intValue();
		if (gctranslator.inlineCheckDepth < 0)
		    throw new NumberFormatException();
	    } catch (NumberFormatException e) {
	        badOptionUsage("illegal argument to -inlineDepth: " +
			       args[offset]);
	    }
	    return offset+1;
	} else if (option.equals("-inlineDepthPastCheck")) {
	    inlineDepthFlags = true;
	    // can't use this along with the -inlineConstructors flag
	    if (inlineConstructors)
		badOptionUsage("can't use -inlineConstructors in combination with either -inlineCheckDepth or -inlineDepthPastCheck");
	    if (offset == args.length) {
		usage();
		System.exit(usageError);
	    }
	    try {
	        gctranslator.inlineDepthPastCheck =
		    new Integer(args[offset]).intValue();
		if (gctranslator.inlineDepthPastCheck < 0)
		    throw new NumberFormatException();
	    } catch (NumberFormatException e) {
	        badOptionUsage("illegal argument to -inlineDepth: " +
			       args[offset]);
	    }
	    return offset+1;
	} else if (option.equals("-inlineConstructors")) {
	    // can't use this along with either of the inline depth flags
	    if (inlineDepthFlags)
		badOptionUsage("can't use -inlineConstructors in combination with either -inlineCheckDepth or -inlineDepthPastCheck");
	    inlineConstructors = true;
	    return offset;
	} else if (option.equals("-suggest")) {
	    suggest = true;
	    return offset;
	} else if (option.equals("-pgc")) {
	    pgc = true;
	    return offset;
	} else if (option.equals("-pcc")) {
	    pcc = true;
	    return offset;
	} else if (option.equals("-nospvc")) {
	  spvc = false;
	  return offset;
	} else if (option.equals("-passify")) {
	  passify = true;
	  return offset;
	} else if (option.equals("-wpp")) {
	  wpp = true;
	  return offset;
	} else if (option.equals("-useDefpred")) {
	    useDefpred = true;
	    return offset;
	} else if (option.equals("-nodsa")) {
	    dsa = false;
	    spvc = false;  // spvc requires dsa for soundness
	    return offset;
	} else if (option.equals("-pdsa")) {
	    pdsa = true;
	    return offset;
	} else if (option.equals("-wpnxw")) {
	    if (offset >= args.length) {
              badOptionUsage("-wpnxw expects an integer parameter");
	    }
	    wpnxw = new Integer(args[offset]).intValue();
	    return offset+1;

	} else if (option.equals("-namePCsize")) {
	    if (offset >= args.length) {
              badOptionUsage("-namePCsize expects an integer parameter");
	    }
	    namePCsize = new Integer(args[offset]).intValue();
	    return offset+1;
	} else if (option.equals("-stats")) {
	    stats = true;
	    return offset;
	} else if (option.equals("-pjt")) {
	    pjt = true;
	    return offset;
	} else if (option.equals("-nvu")) {
	    nvu = true;
	    return offset;
	} else if (option.equals("-assertContinue")) {
	    assertContinue = true;
	    return offset;
	} else if (option.equals("-noTrackReadChars")) {
	    trackReadChars = false;
	    return offset;
	} else if (option.equals("-filterMethodSpecs")) {
	    filterMethodSpecs = true;
	    return offset;
	} else if (option.equals("-noPeepOptGCAssertFalse")) {
	    noPeepOptGCAssertFalse = true;
	    return offset;
	} else if (option.equals("-quiet")) {
	    quiet = true;
	    return offset;
	} else if (option.equals("-noEpeepopt")) {
	    peepOptE = false;
	    return offset;
	} else if (option.equals("-noGCpeepopt")) {
	    peepOptGC = false;
	    return offset;
	} else if (option.equals("-lazySubst")) {
	    lazySubst = true;
	    return offset;
	} else if (option.equals("-mergeinv")) {
	    mergeInv = true;
	    return offset;
	} else if (option.equals("-noAllocUseOpt")) {
	    allocUseOpt = false;
	    return offset;
	} else if (option.equals("-useAllInvPostCall")) {
	    useAllInvPostCall = true;
	    return offset;
	} else if (option.equals("-useAllInvPostBody")) {
	    useAllInvPostBody = true;
	    return offset;
	} else if (option.equals("-useAllInvPreBody")) {
	    useAllInvPreBody = true;
	    return offset;
	} else if (option.equals("-filterInvariants")) {
	    filterInvariants = true;
	    return offset;
	} else if (option.equals("-printAssumers")) {
	    printAssumers = true;
	    return offset;
	} else if (option.equals("-printCompilationUnitsOnLoad")) {
	    printCompilationUnitsOnLoad = true;
	    return offset;
	} else if (option.equals("-mt")) {
	    multiThreaded = true;
	    return offset;
	} else if (option.equals("-inferOTI")) {
	    otiInferOpt = true;
	    return offset;
	} else if (option.equals("-reduce")) {
	    reduce = true;
	    return offset;
	} else if (option.equals("-showReduce")) {
	    showReduce = true;
	    return offset;
	} else if (option.equals("-yieldAfter")) {
	    yieldAfter = true;
	    return offset;
	} else if (option.equals("-havocFinal")) {
	    havocFinal = true;
	    return offset;
	} else if (option.equals("-restrictScope")) {
	    restrictScope = true;
	    return offset;
	} else if (option.equals("-guardedVC")) {
	    if (offset >= args.length) {
              badOptionUsage("-guardedVC expects a directory parameter");
	    }
	    guardedVC = true;
            guardedVCDir = args[offset];
	    return offset+1;
	} else if (option.equals("-ignoreAnnFile")) {
	    if (offset >= args.length) {
              badOptionUsage("-ignoreAnnFile expects a file parameter");
	    }
	    ignoreAnnSet = new Set(readFile(args[offset]).elements());
	    // System.out.println("Ignore set: "+ignoreAnnSet+"\n");
	    return offset+1;
	} else if (option.equals("-routine")) {
	  // the argument to "-routine" is either a simple routine name or a fully
	  // qualified routine name with signature, but we won't ever parse these
	  if (offset == args.length) {
	    badOptionUsage("missing argument to -routine");
	  }
	  String routine = args[offset].intern();
	  if (routinesToCheck == null) {
	    routinesToCheck = new Set();
	  }
	  routinesToCheck.add(routine);
	  return offset+1;
	} else if (option.equals("-routineIndirect")) {
	  if (offset == args.length) {
	    badOptionUsage("missing argument to -routineIndirect");
	  }
	  if (routinesToCheck == null) {
	    routinesToCheck = new Set();
	  }
	  String routineIndirectionFilename = args[offset];
	  try {
	    BufferedReader in = new BufferedReader(
				  new FileReader(routineIndirectionFilename));
	    while (true) {
	      String routine = in.readLine();
	      if (routine == null) {
		break;
	      }
	      routine = routine.intern();
	      routinesToCheck.add(routine);
	    }
	  } catch (IOException e) {
	    badOptionUsage("error reading routine indirection file '" +
			   routineIndirectionFilename + "': " +
			   e.toString());
	  }
	  return offset+1;
	} else if (option.equals("-loopSafe")) {
  	    loopTranslation = LOOP_SAFE;
	    return offset;
	} else if (option.equals("-loop")) {
	  // syntax:  -loop <N>[.0|.5]
	  if (offset == args.length) {
	    badOptionUsage("missing argument to -loop");
	  }
	  String number = args[offset];
	  if (number.length() == 0 || option.charAt(0) == '.') {
	    badOptionUsage("illegal argument to -loop: " + number);
	  }
	  // any other parsing error will be caught in the following loop
	  int cnt = 0;
	  boolean andAHalf = false;
	  for (int i = 0; i < number.length(); i++) {
	    char ch = number.charAt(i);
	    if ('0' <= ch && ch <= '9') {
	      if (10000 <= cnt) {
		badOptionUsage("-loop specifies too many iterations: " +
			       number);
	      }
	      cnt = 10*cnt + ch - '0';
	      continue;
	    } else if (ch == '.') {
	      if (number.length() == i+2) {
	        if (number.charAt(i+1) == '5') {
		  andAHalf = true;
		  break;
		} else if (number.charAt(i+1) == '0') {
		  andAHalf = false;
		  break;
		}
	      }
	    }
	    badOptionUsage("illegal argument to -loop: " + number);
	  }
	  loopUnrollCount = cnt;
	  loopUnrollHalf = andAHalf;
	  return offset+1;
	} else if (option.equals("-loopFallThru")) {
	    loopTranslation = LOOP_FALL_THRU;
	    return offset;
	} else if (option.equals("-predAbstract")) {
	    predAbstract = true;
	    loopTranslation = LOOP_SAFE;
	    lastVarUseOpt = false;
	    return offset;
	} else if (option.equals("-inferPredicates")) {
	    inferPredicates = true;
	    predAbstract = true;
	    loopTranslation = LOOP_SAFE;
	    lastVarUseOpt = false;
	    return offset;
	} else if (option.equals("-noDirectTargetsOpt")) {
	    noDirectTargetsOpt = true;
	    return offset;
	} else if (option.equals("-nestQuantifiers")) {
	    nestQuantifiers = true;
	    return offset;
	} else if (option.equals("-useIntQuantAntecedents")) {
	    useIntQuantAntecedents = true;
	    return offset;
	} else if (option.equals("-excuseNullInitializers")) {
	    excuseNullInitializers = true;
	    return offset;
	} else if (option.equals("-strongAssertPostNever")) {
	    strongAssertPost = 0;
	    return offset;
	} else if (option.equals("-strongAssertPostAtomic")) {
	    strongAssertPost = 1;
	    return offset;
	} else if (option.equals("-strongAssertPostAlways")) {
	    strongAssertPost = 2;
	    return offset;
	} else if (option.equals("-noLastVarUseOpt")) {
	    lastVarUseOpt = false;
	    return offset;
	} else if (option.equals("-hidecall")) {
	    showCallDetails = false;
	    return offset;
	} else if (option.equals("-showloop")) {
	    showLoopDetails = true;
	    return offset;
	} else if (option.equals("-showyield")) {
	    showYieldDetails = true;
	    return offset;
	} else if (option.equals("-plainwarning")) {
	    plainWarning= true;
	    return offset;
	} else if (option.equals("-notrace")) {
	    traceInfo = 0;
	    return offset;
	} else if (option.equals("-verboseTrace")) {
	    traceInfo = 2;
	    return offset;
	} else if (option.equals("-counterexample")) {
	    counterexample = true;
	    return offset;
	} else if (option.equals("-bubbleNotDown")) {
	  bubbleNotDown = true;
	  return offset;
	} else if (option.equals("-backpredfile")) {
	    if (offset>=args.length) {
		usage();
		System.exit(usageError);
	    }
	    univBackPredFile = args[offset];
	    return offset+1;
	} else if (option.equals("-sxLog")) {
	    if (offset>=args.length) {
		usage();
		System.exit(usageError);
	    }
	    sxLog = args[offset];
	    return offset+1;
	} else if (option.equals("-nowarn")) {
	    if (offset>=args.length) {
		usage();
		System.exit(usageError);
	    }
	    if (args[offset].equals("All")) {
		// Commented by Shaz 2/27/2002 
		//		NoWarn.setAllChkStatus(TagConstants.CHK_AS_ASSUME);
		NoWarn.setAllChkStatus(TagConstants.CHK_AS_SKIP);
	    } else {
	      int tag = NoWarn.toNoWarnTag(args[offset]);
	      if (tag==0) {
		badOptionUsage(name() + ": '" + args[offset]
			       + "' is not a legal warning category");
	      }
	      // Commented by Shaz 2/27/2002 
	      //	      NoWarn.setChkStatus(tag, TagConstants.CHK_AS_ASSUME);
	      NoWarn.setChkStatus(tag, TagConstants.CHK_AS_SKIP);
	    }
	    return offset+1;
	} else if (option.equals("-warn")) {
	    if (offset>=args.length) {
		usage();
		System.exit(usageError);
	    }
	    // Note:  There's an intentional lack of symmetry with -nowarn
	    // here, in that "-warn All" is not supported.  This design
	    // allows ESC/Java to include special checks, perhaps like the
	    // Stale checks of the Higher-Level Races checker, that won't
	    // be publicly advertised.
	    int tag = NoWarn.toNoWarnTag(args[offset]);
	    if (tag==0) {
		badOptionUsage(name() + ": '" + args[offset]
			       + "' is not a legal warning category");
	    }
	    NoWarn.setChkStatus(tag, TagConstants.CHK_AS_ASSERT);
	    return offset+1;
        }

	// Pass on unrecognized options:
	return super.processOption(option, args, offset);
    }

    private Vector readFile(String filename) {
	Vector r = new Vector();
	StringBuffer s = new StringBuffer();
	try {
	    Reader R = new BufferedReader(new InputStreamReader(new FileInputStream(filename)));
	    int c;
	    do {
		while( (c=R.read())!= -1 && c != '\n' ) {
		    s.append( (char)c );
		}
		// Delete from 3rd space on
		String st = s.toString();
		int i=st.indexOf(' ');
		if( i != -1 ) i=st.indexOf(' ',i+1);
		if( i != -1 ) i=st.indexOf(' ',i+1);
		if( i != -1 ) st=st.substring(0,i);
		r.addElement( st );
		s.setLength(0);
	    } while( c != -1 );
	    return r;
        } catch(IOException e) {
	  throw new RuntimeException("IOException: " + e);
	}
    }

    //@ ensures false;
    private void badOptionUsage(String s) {
      System.err.println(s);
      System.err.println();
      usage();
      System.exit(usageError);
    }

    /***************************************************
     *                                                 *
     *  Front-end setup:		               *
     *                                                 *
     ***************************************************/

    /**
     ** Returns the Esc StandardTypeReader, EscTypeReader.
     **/
    public StandardTypeReader makeStandardTypeReader(String path,
						     PragmaParser P) {
        return EscTypeReader.make(path, P);
    }

    /**
     ** Returns the EscPragmaParser.
     **/
    public javafe.parser.PragmaParser makePragmaParser() {
      return new escjava.parser.EscPragmaParser();
    }

    /**
     ** Returns the pretty printer to set
     ** <code>PrettyPrint.inst</code> to.
     **/
    public PrettyPrint makePrettyPrint() {
      DelegatingPrettyPrint p = new EscPrettyPrint();
      p.del = new StandardPrettyPrint(p);
      return p;
    }

    /**
     ** Called to obtain an instance of the javafe.tc.TypeCheck class
     ** (or a subclass thereof). May not return <code>null</code>.  By
     ** default, returns <code>javafe.tc.TypeCheck</code>.
     **/
    public javafe.tc.TypeCheck makeTypeCheck() {
	return new escjava.tc.TypeCheck();
    }


    /**
     ** Override SrcTool.notify to ensure all lexicalPragmas get
     ** registered as they are loaded.
     **/
    public void notify(CompilationUnit justLoaded) {
	super.notify(justLoaded);

	NoWarn.registerNowarns(justLoaded.lexicalPragmas);

	if (printCompilationUnitsOnLoad) {
	  String pkgName = justLoaded.pkgName.printName();
	  String filename = Location.toFileName(justLoaded.loc);
	  System.out.println("LOADED: " + pkgName + " " + filename);
	}
    }


    /***************************************************
     *                                                 *
     * Main processing code:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Start up an instance of this tool using command-line arguments
     ** <code>args</code>. <p>
     **
     ** This is the main entry point for the <code>escjava</code>
     ** command.<p>
     **/
    //@ requires \nonnullelements(args)
    public static void main(String[] args) {
	//	javafe.util.Info.on = true;
      try {
	javafe.SrcTool t = new Main();

	// Disallow the -avoidSpec option:
	t.allowAvoidSpec = false;

	t.run(args);
      } catch (OutOfMemoryError oom) {
        Runtime rt = Runtime.getRuntime();
        long memUsedBytes = rt.totalMemory() - rt.freeMemory();
	System.out.println("java.lang.OutOfMemoryError (" + memUsedBytes +
			   " bytes used)");
	oom.printStackTrace(System.out);
	System.exit(1);
      }
    }


    /***************************************************
     *                                                 *
     * SrcTool-instance specific processing:	       *
     *                                                 *
     ***************************************************/

    /**
     ** Our Simplify instance or null iff stages<6.
     **/
    public static Simplify prover = null;
    //@ invariant (prover == null) == (stages<6)


    /** An instance of the GC->VC translator **/
    public static Translate gctranslator = new Translate();


    /**
     ** Override setup so can issue version # as soon as possible
     ** (aka, just after decode options so know if -quiet issued or
     ** not).
     **/
    public void setup() {
	super.setup();
	
	javafe.tc.Types.inst = new escjava.tc.Types();

	if (!quiet)
	    System.out.println("ESC/Java version " + version);
    }


    /**
     ** Hook for any work needed before <code>handleCU</code> is
     ** called on each <code>CompilationUnit</code> to process them.
     **/
    public void preprocess() {

	// call our routines to run the constructor inlining experiment
	if (inlineConstructors)
	    InlineConstructor.inlineConstructorsEverywhere(loaded);
	

	if (6 <= stages || predAbstract) {
	    long startTime = java.lang.System.currentTimeMillis();
	    prover = new Simplify();

	    // if (!quiet) System.out.println("  Prover started:" + timeUsed(startTime));
	}

	if (prover != null) {
	  escjava.backpred.BackPred.genUnivBackPred(prover.subProcessToStream());
	  prover.sendCommands("");
	}

    }

    /**
     ** A wrapper for opening output files for printing.
     **/
    private PrintStream fileToPrintStream(String dir, String fname) {
      File f = new File(dir, fname);
      try {
        return new PrintStream(new FileOutputStream(f));
      } catch (IOException e) {
        javafe.util.ErrorSet.fatal(e.getMessage());
        return null;  // unreachable
      }
    }

    /**
     ** Hook for any work needed after <code>handleCU</code> has been called
     ** on each <code>CompilationUnit</code> to process them.
     **/
    public void postprocess() {

      // If we are in the Houdini context (guardedVC is true), output
      // the association of file numbers to their names.
      // Also, dump out a list of guard variable names.
      if (guardedVC) {
        PrintStream o = fileToPrintStream(guardedVCDir, guardedVCFileNumbers);
        Vector v = LocationManagerCorrelatedReader.fileNumbersToNames();
        for(int i=0; i<v.size(); i++) o.println(i + " " + v.elementAt(i));
        o.close();

        o = fileToPrintStream(guardedVCDir, guardedVCGuardFile);
        Enumeration e = guardVars.elements();
        while (e.hasMoreElements()) {
          o.println((String)e.nextElement());
        }
        o.close();
      }

      if (prover != null) {
	prover.close();
	prover = null;
      }
    }

    /**
     ** This method is called on each <code>CompilationUnit</code>
     ** that this tool processes.  This method overrides the implementation
     ** given in the superclass, adding a couple of lines before the
     ** superclass implementation is called.
     **/
    public void handleCU(CompilationUnit cu) {
	NoWarn.setStartLine(startLine, cu);

        UniqName.setDefaultSuffixFile(cu.getStartLoc());
	super.handleCU(cu);

	startLine = -1;		// StartLine applies only to first CU
    }


    /**
     ** This method is called by SrcTool on the TypeDecl of each
     ** outside type that SrcTool is to process. <p>
     **
     ** In addition, it calls itself recursively to handle types
     ** nested within outside types.<p>
     **/
    public void handleTD(TypeDecl td) {
	TypeSig sig = TypeCheck.inst.getSig(td);
	if (sig.getTypeDecl().specOnly)	// do not process specs
	    return;

	if (Location.toLineNumber(td.getEndLoc()) < startLine)
	     return;

	long startTime = java.lang.System.currentTimeMillis();
	if (!quiet)
	     System.out.println("\n" + sig.toString() + " ...");

	// Do actual work:
	boolean aborted = processTD(td);

	if (!quiet)
	    System.out.println("  [" + timeUsed(startTime)
		+ " total]"
		+ (aborted ? " (aborted)" : "") );

	/*
	 * Handled any nested types:  [1.1]
	 */
	TypeDecl decl = sig.getTypeDecl();
	for (int i=0; i<decl.elems.size(); i++) {
	    if (decl.elems.elementAt(i) instanceof TypeDecl)
		handleTD((TypeDecl)decl.elems.elementAt(i));
	}
    }


    /**
     ** Run all the requested stages on a given TypeDecl;
     ** return true if we had to abort. <p>
     **
     ** Precondition: td is not from a binary file.<p>
     **/
    private boolean processTD(TypeDecl td) {
	// ==== Start stage 1 ====

	/*
	 * Do Java type checking then print Java types if we've been
	 * asked to:
	 */
	int errorCount = ErrorSet.errors;
	TypeSig sig = TypeCheck.inst.getSig(td);
	sig.typecheck();
	NoWarn.typecheckRegisteredNowarns();


	if (pjt) {
	    // Create a pretty-printer that shows types
	    DelegatingPrettyPrint p = new javafe.tc.TypePrint();
	    p.del = new EscPrettyPrint(p, new StandardPrettyPrint(p));

	    System.out.println("\n**** Source code with types:");
	    p.print(System.out, 0, td);
	}

	// Turn off extended static checking and abort if any errors
	// occured while type checking *this* TypeDecl:
	if (errorCount < ErrorSet.errors) {
	    if (stages>1) {
		stages = 1;
		ErrorSet.caution(
	         "Turning off extended static checking due to type error(s)");
	    }
	    return true;
	}

	// ==== Start stage 2 ====
	if (stages<2)
	    return false;
	

	// Generate the type-specific background predicate
	errorCount = ErrorSet.errors;
	FindContributors scope = new FindContributors(sig);
	VcToString.resetTypeSpecific();

        if (guardedVC) {
          String locStr = UniqName.locToSuffix(td.locId);
          String fn = locStr + ".class." + guardedVCFileExt;
          File f = new File(guardedVCDir, fn);
          PrintStream o = fileToPrintStream(guardedVCDir, fn);
          o.println(ClassVCPrefix);
          o.println(td.id + "@" + locStr);
          o.print("\n(BG_PUSH ");
          escjava.backpred.BackPred.genTypeBackPred(scope, o);
          o.println(")");
          o.close();
        }

	if (prover != null) {
	    PrintStream ps = prover.subProcessToStream();
	    ps.print("\n(BG_PUSH ");
	    escjava.backpred.BackPred.genTypeBackPred(scope, ps);
	    ps.println(")");
	    prover.sendCommands("");
	}

	// Turn off extended static checking and abort if any type errors
	// occured while generating the type-specific background predicate:
	if (errorCount < ErrorSet.errors) {
	    stages = 1;
	    ErrorSet.caution(
		"Turning off extended static checking due to type error(s)");
	    return true;
	}

	// ==== Start stage 3 ====
	if (3 <= stages) {

	    LabelInfoToString.reset();
	    InitialState initState = new InitialState(scope);
	    LabelInfoToString.mark();

	    // Process the elements of "td"; stage 3 continues into stages 4
	    // and 5 inside processTypeDeclElem:
	    if (inlineConstructors && !Modifiers.isAbstract(td.modifiers)) {
		// only process inlined versions of methods
		for (int i = 0; i < td.elems.size(); i++) {
		    TypeDeclElem tde = td.elems.elementAt(i);
		    if (!InlineConstructor.isConstructorInlinable(tde) ||
			InlineConstructor.isConstructorInlinedMethod((MethodDecl) tde))
			processTypeDeclElem(tde, sig, initState);
		}
	    }
	    else {
		for (int i = 0; i < td.elems.size(); i++)
		    processTypeDeclElem(td.elems.elementAt(i),
					sig, initState);
	    }
	}

	// ==== all done; clean up ====
	if (prover!=null)
	    prover.sendCommand("(BG_POP)");

	return false;
    }


    /**
     ** Run stages 3+..6 as requested on a TypeDeclElem. <p>
     **
     ** Precondition: te is not from a binary file,
     **		      sig is the TypeSig for te's parent, and
     **		      initState != null.<p>
     **/
    private void processTypeDeclElem(TypeDeclElem te, TypeSig sig,
				     InitialState initState) {
	// Only handle methods and constructors here:
	if (!(te instanceof RoutineDecl))
	    return;
	RoutineDecl r = (RoutineDecl)te;


	long startTime = java.lang.System.currentTimeMillis();
	if (!quiet) {
	    String name = TypeCheck.inst.getRoutineName(r) +
	                  TypeCheck.inst.getSignature(r);
	    System.out.println("\n" + sig.toString() + ": " +
			       name + " ...");
	}

	// Do the actual work, handling not implemented exceptions:
	String status = "error";
	try {
	    status = processRoutineDecl(r, sig, initState);
	} catch (javafe.util.NotImplementedException e) {
	    ErrorSet.error(e.getMessage());
	}

	if (!quiet)
	     System.out.println("    [" + timeUsed(startTime) + "]  "
		+ status);

	/*************************
	System.out.println("Lines " +
			   (Location.toLineNumber(r.getEndLoc())
			    -Location.toLineNumber(r.getStartLoc()))
			   +" time "+timeUsed(startTime));
	*******************/


    }


    /**
     ** Run stages 3+..6 as requested on a RoutineDeclElem; returns a
     ** short (~ 1 word) status message. <p>
     **
     ** Precondition: r is not from a binary file,
     **		      sig is the TypeSig for r's parent, and
     **		      initState != null.<p>
     **/
    //@ ensures \result != null;
    private String processRoutineDecl(/*@ non_null */ RoutineDecl r,
				      /*@ non_null */ TypeSig sig,
				      /*@ non_null */ InitialState initState) {


	if ( Location.toLineNumber(r.getEndLoc()) < startLine )
	     return "skipped";
	String simpleName = TypeCheck.inst.getRoutineName(r).intern();
	String fullName = sig.toString() + "." + simpleName +
	  TypeCheck.inst.getSignature(r);
	fullName = removeSpaces(fullName).intern();
	if (routinesToCheck != null &&
	    !routinesToCheck.contains(simpleName) &&
	    !routinesToCheck.contains(fullName)) {
	  return "skipped";
	}

	// ==== Stage 3 continues here ====
	/*
	 * Translate body into a GC:
	 */
	long startTime = java.lang.System.currentTimeMillis();
	long routineStartTime = startTime;

	// don't check the body if we're checking some other inlining depth
	Translate.globallyTurnOffChecks(gctranslator.inlineCheckDepth > 0);

	LabelInfoToString.resetToMark();
	GuardedCmd gc = computeBody(r, initState);
	GuardedCmd origgc = gc;

	// restore ordinary checking of assertions
	NoWarn.useGlobalStatus = false;

	String gcTime = timeUsed(startTime);
	startTime = java.lang.System.currentTimeMillis();
	
	UniqName.resetUnique();
	if (gc==null)
	    return "passed immediately";
	if (pgc) {
	    System.out.println("\n**** Guarded Command:");
	    ((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, gc);
	    System.out.println("");
	}
	Set directTargets = Targets.direct(gc);


	// ==== Start stage 4 ====
	if (stages<4)
	    return "ok";

	// Convert GC to DSA:

	String dsaTime = "";
	if (dsa) {
	  /*
	   * From experiements from POPL01 (Cormac)
	   gc = passify ? Passify.compute(gc) : DSA.dsa(gc);
	  */
	  gc = DSA.dsa(gc);
	  dsaTime = timeUsed(startTime);
	  startTime = java.lang.System.currentTimeMillis();
	
	  if (pdsa) {
	    System.out.println("\n**** Dynamic Single Assignment:");
	    ((EscPrettyPrint)PrettyPrint.inst).print(System.out, 0, gc);
	    System.out.println("");
	  }
	}
	
	// ==== Start stage 5 ====
	if (stages<5)
	    return "ok";

	// Generate the VC for GC:
	Expr vcBody;
	/*
         * From experiements from POPL01 (Cormac)
	if(wpnxw != 0 ) {
	    vcBody = WpName.compute( gc, wpnxw );
	} else 
	*/
	if (spvc) {
	    /*  
	     * From experiements from POPL01 (Cormac)
	        vcBody = wpp ? Wpp.compute(gc, GC.truelit, GC.truelit) : 
  	            SPVC.compute(gc);
	    */ 
	    vcBody = SPVC.compute(gc);
	} else {
	  vcBody = Ejp.compute(gc, GC.truelit, GC.truelit);
	}
	Expr vc = GC.implies(initState.getInitialState(), vcBody);
        // Attach a label for use in the logfile generated (if any):
	String label = "vc." + sig.toString() + ".";
	if (r instanceof MethodDecl)
	    label += ((MethodDecl)r).id;
	else
	    label += "<constructor>";
	label += "." + UniqName.locToSuffix(r.getStartLoc());
	vc = LabelExpr.make(r.getStartLoc(), r.getEndLoc(),
			    false, Identifier.intern(label), vc);

	// Check for VC too big:
	int usize = Util.size(vc, vclimit);
	if (usize == -1) {
	    ErrorSet.caution("Unable to check "
		+ TypeCheck.inst.getName(r)
		+ " of type " + TypeSig.getSig(r.parent)
		+ " because its VC is too large");
	    return "VC too big";
	}

	if (printAssumers) {
	  System.out.print("ASSUMERS: ");
	  System.out.print(Location.toFileName(r.getStartLoc()));
	  System.out.print('|');
	  System.out.print(fullName);
	  System.out.println(LabelInfoToString.get());
	}

	String ejpTime = timeUsed(startTime);
	startTime = java.lang.System.currentTimeMillis();
	// Translate VC to a string
	Info.out("[converting VC to a string]");
	// String vcsexpr = VcToString.compute(vc);  -- modified to use stream

        if (guardedVC) {
          String fn = UniqName.locToSuffix(r.locId) + ".method." + 
            guardedVCFileExt;
          PrintStream o = fileToPrintStream(guardedVCDir, fn);
          o.println(MethodVCPrefix);
          o.println(r.parent.id + "@" + UniqName.locToSuffix(r.parent.locId));
          VcToString.compute(vc, o);
          o.close();
          return "guarded VC generation finished";
        }

	String vcTime = timeUsed(startTime);
	startTime = java.lang.System.currentTimeMillis();

	// ==== Start stage 6 ====
	if (stages<6)
	    return "ok";

	// Check the VC:
        prover.startProve();

	VcToString.compute(vc, prover.subProcessToStream());
//,,,
//  FindContributors scope = new FindContributors(sig);
//  System.out.println("+++");
//  escjava.backpred.BackPred.genTypeBackPred(scope, System.out);
//  VcToString.compute(vc, System.out);
//  System.out.println("+++");
	Enumeration results = prover.streamProve();

	// Process Simplify's output
	String status = "unexpectedly missing Simplify output";
	boolean nextWarningNeedsPrecedingLine = true;
	while (results.hasMoreElements()) {
	  SimplifyOutput so = (SimplifyOutput)results.nextElement();
	  switch (so.getKind()) {
	    case SimplifyOutput.VALID:
	      status = "passed";
	      break;
	    case SimplifyOutput.INVALID:
	      status = "failed";
	      break;
	    case SimplifyOutput.UNKNOWN:
	      status = "timed out";
	      break;
	    case SimplifyOutput.COMMENT: {
	      SimplifyComment sc = (SimplifyComment)so;
	      System.out.println("SIMPLIFY: " + sc.getMsg());
	      break;
	    }
	    case SimplifyOutput.COUNTEREXAMPLE: {
	      if (nextWarningNeedsPrecedingLine) {
		escjava.translate.ErrorMsg.printSeparatorLine(System.out);
		nextWarningNeedsPrecedingLine = false;
	      }
	      SimplifyResult sr = (SimplifyResult)so;
	      escjava.translate.ErrorMsg.print(TypeCheck.inst.getName(r),
					       sr.getLabels(), sr.getContext(),
					       r, directTargets, System.out);
	      break;
	    }
	    case SimplifyOutput.EXCEEDED_PROVER_KILL_TIME: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check " +
			       TypeCheck.inst.getName(r) +
			       " of type " + TypeSig.getSig(r.parent) +
			       " completely because too much time required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.EXCEEDED_PROVER_KILL_ITER: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check " +
			       TypeCheck.inst.getName(r) +
			       " of type " + TypeSig.getSig(r.parent) +
			       " completely because" +
			       " too many iterations required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.REACHED_CC_LIMIT:
	      ErrorSet.caution("Not checking " +
			       TypeCheck.inst.getName(r) +
			       " of type " + TypeSig.getSig(r.parent) +
			       " completely because" +
			       " warning limit (PROVER_CC_LIMIT) reached");
	      break;
	    case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_TIME: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check subgoal of " +
			       TypeCheck.inst.getName(r) +
			       " of type " + TypeSig.getSig(r.parent) +
			       " completely because too much time required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_ITER: {
	      SimplifyResult sr = (SimplifyResult)so;
	      ErrorSet.caution("Unable to check subgoal of " +
			       TypeCheck.inst.getName(r) +
			       " of type " + TypeSig.getSig(r.parent) +
			       " completely because" +
			       " too many iterations required");
	      if (Info.on && sr.getLabels() != null) {
		Info.out("Current labels: " + sr.getLabels());
	      }
	      nextWarningNeedsPrecedingLine = true;
	      break;
	    }
	    case SimplifyOutput.WARNING_TRIGGERLESS_QUANT: {
	      TriggerlessQuantWarning tqw = (TriggerlessQuantWarning)so;
	      int loc = tqw.getLocation();
	      String msg = "Unable to use quantification because " +
                           "no trigger found: " + tqw.e1;
	      if (loc != Location.NULL) {
		ErrorSet.caution(loc, msg);
	      } else {
		ErrorSet.caution(msg);
	      }
	      if (Info.on && tqw.getLabels() != null) {
		Info.out("Current labels: " + tqw.getLabels());
	      }
	      break;
	    }
	    default:
	      Assert.fail("unexpected type of Simplify output");
	      break;
	  }
	}

	String proofTime = timeUsed(startTime);
	if (stats) {
	  System.out.println("    [Time: "+timeUsed(routineStartTime)
			     +" GC: "+gcTime
			     +" DSA: "+dsaTime
			     +" Ejp: "+ejpTime
			     +" VC: "+vcTime
			     +" Proof(s): "+proofTime
			     +"]");
	  System.out.println("    [Size: "
			     +" src: "+Util.size(r)
			     +" GC: "+Util.size(origgc)
			     +" DSA: "+Util.size(gc)
			     +" VC: "+Util.size(vc)
			     +"]");
	}
	return status;
    }


  /**
    * This method computes the guarded command (including assuming
    * the precondition, the translated body, the checked postcondition,
    * and the modifies constraints) for the method or constructor
    * <code>r</code> in scope <code>scope</code>. <p>
    *
    * Returns <code>null</code> if <code>r</code> doesn't have a body.
    **/

  private GuardedCmd computeBody(RoutineDecl r, InitialState initState) {
    if (r.getTag() == TagConstants.METHODDECL &&
	((MethodDecl)r).body == null) {
      // no body
      return null;
    }

    // don't check the routine if it's a helper
    if (Helper.isHelper(r)) {
	return null;
    }

    FindContributors scope = new FindContributors(r);

    /* Compute the desugaring of yield commands and store it in a new
     * TrYield object To do this, we need access to the stuff declared
     * in the "performs" pragmas. To do this, compute a partial spec,
     * and store it in partialSpec.  */
    Spec partialSpec = null;
    TrYield yieldObj = null;
    if (multiThreaded) {
	partialSpec = GetSpec.getCommonSpecMT(r, scope, initState.getPreMap());
	/* now create a new TrYield object */
	gctranslator.setYieldObj(scope, partialSpec);	
	/* and get it */
	yieldObj = gctranslator.getYieldObj();
    }

    /*
     * Compute an upper bound for synTargs if -O7 given.
     *
     * For now, do this via the kludge of calling trBody...  !!!!
     */
    Set predictedSynTargs = null;
    if (!useAllInvPreBody) {
	long T = java.lang.System.currentTimeMillis();
	GuardedCmd tmpBody = gctranslator.trBody(r, scope,
					         initState.getPreMap(),
					         /*predictedSynTargs*/ null,
						 null,
                                                 /* issueCautions */ false);

	if (Main.multiThreaded) {
	    yieldObj.desugarYields(tmpBody, r.getEndLoc());
	}

	if (noDirectTargetsOpt)
	  predictedSynTargs = Targets.normal(tmpBody);
	else
	  predictedSynTargs = Targets.direct(tmpBody);
	if (stats)
	    System.out.println("      [prediction time: " + timeUsed(T) + "]");

	if (Main.multiThreaded) {
	    Assert.notFalse(yieldObj != null);
	}

    }

    /*
     * Translate the body:
     */
    GuardedCmd body = gctranslator.trBody(r, scope,
					  initState.getPreMap(),
					  predictedSynTargs, null,
                                          /* issueCautions */ true);

    if (Main.multiThreaded) {
	yieldObj.desugarYields(body, r.getEndLoc());
    }

    Set fullSynTargs = Targets.normal(body);
    Set synTargs;
    if (noDirectTargetsOpt)
      synTargs = fullSynTargs;
    else
      synTargs = Targets.direct(body);


    /*
     * Verify predictedSynTargs if present that
     * synTargs is a subset of predictedSynTargs.
     */
    if (predictedSynTargs!=null) {
	Enumeration e = synTargs.elements();
	while (e.hasMoreElements()) {
	    GenericVarDecl target = (GenericVarDecl)(e.nextElement());
	    Assert.notFalse(predictedSynTargs.contains(target));
	}
    }

    Spec spec;
    GuardedCmd fullCmd;

    if (multiThreaded) {
	Assert.notFalse(yieldObj != null);

	spec = GetSpec.getSpecForBody(partialSpec, scope, synTargs);	

	// if the current RoutineDecl corresponds to one of our
	// constructor-inlined methods, then zero out its postconditions
	if (r instanceof MethodDecl &&
	    InlineConstructor.isConstructorInlinedMethod((MethodDecl) r))
	    spec.post = ConditionVec.make();
	
	fullCmd = 
	    GetSpec.surroundBodyBySpec(gctranslator, 
				       body, spec, scope, fullSynTargs,
				       initState.getPreMap(),
				       r.locOpenBrace,
				       r.getEndLoc());

	Translate.addTraceLabelSequenceNumbers(fullCmd);

	if (Main.reduce) {
	    // perform reduction to get rid of yields 
	    Reduction.reduce(r, initState, yieldObj, fullCmd);
	}

	fullCmd = yieldObj.desugarYields(fullCmd, r.getEndLoc());
	
	GuardedCmdVec gcVec = GuardedCmdVec.make();

	// check transitivity & reflexivity of OTIs 
	gcVec.addElement(ChkReflTrans.getOtiReflTransChecks(scope, r.locOpenBrace));
 	gcVec.addElement(yieldObj.getStateTrueAssignCode());
	gcVec.addElement(yieldObj.getHavocAssumes(GenericVarDeclVec.make()));
	gcVec.addElement(yieldObj.getAssignCode());
	gcVec.addElement(fullCmd);
	gcVec.addElement(yieldObj.getPerformsEndAssertCode(r.getEndLoc()));

	fullCmd = GC.seq(gcVec);

	/*
	spec = GetSpec.getSpecForBody(partialSpec, scope, 
				      synTargs);	
	*/
    } else {
	spec = GetSpec.getSpecForBody(r, scope, synTargs,
				       initState.getPreMap());

	// if the current RoutineDecl corresponds to one of our
	// constructor-inlined methods, then zero out its postconditions
	if (r instanceof MethodDecl &&
	    InlineConstructor.isConstructorInlinedMethod((MethodDecl) r))
	    spec.post = ConditionVec.make();

	fullCmd = 
	    GetSpec.surroundBodyBySpec(body, spec, scope, fullSynTargs,
				       initState.getPreMap(),
				       r.getEndLoc());
	Translate.addTraceLabelSequenceNumbers(fullCmd);
    }

    // commented out temporarily to avoid reference to predicate abstraction and BDDs
    /*
    if (Main.loopTranslation == LOOP_SAFE && Main.predAbstract) {
	long T = java.lang.System.currentTimeMillis();
	Traverse.compute(fullCmd, initState, gctranslator);
	if (stats) {
	    System.out.println("      [predicate abstraction time: " + timeUsed(T) + "]");
	}
    }
    */

    yieldObj = null; // can be garbage collected

    return fullCmd;

  }


    /***************************************************
     *                                                 *
     * Misc. Utility routines:			       *
     *                                                 *
     ***************************************************/

    /**
     ** Compute the time used from a start time to now, then return it
     ** in a user readable form.
     **/
    /*@ensures \result != null*/
    public static String timeUsed(long startTime) {
	long delta = java.lang.System.currentTimeMillis() - startTime;

	return (delta/1000.0) + " s";
    }

  private static String removeSpaces(/*@ non_null */ String s) {
    while (true) {
      int k = s.indexOf(' ');
      if (k == -1) {
	return s;
      }
      s = s.substring(0, k) + s.substring(k+1);
    }
  }

}
