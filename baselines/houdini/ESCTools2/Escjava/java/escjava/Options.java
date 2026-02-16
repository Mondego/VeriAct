/* Copyright 2000, 2001, Compaq Computer Corporation */
/* Copyright 2007, 2008, Systems Research Group, University College Dublin, 
 * Ireland */

package escjava;

import java.util.Enumeration;
import java.util.Hashtable;
import java.util.Properties;
import java.util.Vector;
import java.io.*;

import javafe.ast.*;
import javafe.util.Set;
import javafe.util.UsageError;
import escjava.ast.*;
import escjava.ast.TagConstants;
import escjava.translate.NoWarn;

import javafe.util.Assert;

/**
 * This class parses the options on the command-line and is a structure for holding
 * the values of options.
 */

public class Options extends javafe.SrcToolOptions {
    final String[][] escPublicOptionData = {
            { "-CounterExample",
                    "If a warning is discovered, generate a counter-example if possible." },
            { "-JavaAssertions, -eaJava",
                    "Treat Java assert statements as normal Java asserts." },
            {
                    "-JMLAssertions, -eaJML",
                    "Treat Java assert statements like JML assert statements.  -eaJML and -eaJava are mutually exclusive switches.  -eaJML is the default setting." },
            {
                    "-Loop <iteration_count>[.0|.5]",
                    "Consider <iteration_count> iterations of all loops (i.e., unroll all loops <iteration_count> times); if <iteration_count>.5, evaluate loop guard one extra time. The default is -Loop 1.5." },
            {
                    "-NoCheck",
                    "Do all steps, including verification condition generation, but perform no checking with the prover." },
            {
                    "-NoSemicolonWarnings",
                    "Suppress warnings about semicolons missing at the end of annotations (to support old ESC/Java)." },
            { "-Simplify <path_to_executable>",
                    "The path to the SIMPLIFY executable." },
            {
                    "-Specs <path_to_specs_directory_or_jarfile>",
                    "The jar file or directory path of the set of system specs to use; these are appended to the sourcepath, if specified, or else the classpath.  Multiple uses of -Specs are ignored; only final use of -Specs is recognized, as in javac.  The default version of specs depends on the platform" },
            {
                    "-Typecheck",
                    "Perform only parsing and typechecking; no generation of verification conditions or execution of prover takes place." },
            {
                    "-LoopSafe",
                    "Use an alternate form of loop unrolling for VC generation that is more rigorous but more expensive." },
            {
                    "-NoRedundancy",
                    "Do not check specs in redundant specification (e.g., requires_redundantly, etc.)." },
            {
                    "-NoTrace",
                    "Do not print the execution trace that leads to the potential error being warned about." },
            {
                    "-NoWarn <category>",
                    "Do not warn about the specified warning category.  The special category 'All' can be used to ignore all warnings.  The full list of warnings is found in the User's Manual." },
            {
                    "-PlainWarning",
                    "Suppress the output of the partial counterexample in the case of invariant warnings." },
            //$$
            { "-Prover <prover_list>",
                    "Specify which prover(s) should be used; <prover_list> is a comma-separated list of names of the provers to be enabled. Usage example : '-Prover simplify,harvey'. Simplify is the default prover if this option is not used." },
            //$$
            {
                    "-Routine [<routine_identifier> | <fully_quality_routine_signature>]",
                    "Check\n\tonly the specified routine in all specified classes." },
            {
                    "-RoutineIndirect <routine_file>",
                    "The file <routine_file> should contain a list of all routines that are to be checked, similar to the -list\n\toption." },
            {
                    "-Start <line_number>",
                    "Start checking the first class specified at line number\n"
                            + "\t<line_number>; all lines before <line_number> have warnings disabled." },
            {
                    "-Stats",
                    "Display a more detailed breakdown of some information. You can supply some keyword indicating what you want :  time space complexity termComplexity variableComplexity quantifierComplexity. Usage example : -Stats time,space,variableComplexity. Default behavior is -Stats time,space. The complexity parameter displays all *Complexity flags." },
            { "-Suggest",
                    "After each warning, suggest an annotation that will suppress the\n\twarning." },
            { "-VCLimit <number>",
                    "Set the maximum size of the VC to <number> bytes;\n\tdefaults to 500,000." },
            { "-Warn <category>",
                    "Turns on all warnings of category <category> if they are currently turned off." },
            { "-nonNullByDefault", "Declarations of reference types are assumed to be non-null by default." }
    };

    final String[][] escPrivateOptionData = {
            {
                    "-ShowDesugardedSpecs, -sds",
                    "Show the parsed specs after being desugared from heavyweight to lightweight specs" },
            { "-PrintGuardedCommands, -pgc", "Print the guarded commands" },
            { "-PrettyPrintVC, -ppvc", "Pretty-print verification conditions" },
            {
                    "-pxLog <filename>",
                    "Pretty-print the commands (S-expressions) sent to Simplify in the file named <filename>" },
            {
                    "-sxLog <filename>",
                    "Print the commands (S-expressions) sent to Simplify in the file named <filename>" },
            {
                    "-Stages <number>",
                    "Run at most <number> stages.  The current stages are (1) loading, parsing, and type checking; (2) generating the type-specific background predicate; (3) translating from Java to guarded commands; (4) optionally converting to dynamic-single-assignment form; (5) generating the verification condition (VC); (6) and calling the theorem prover to verify the VC." },
            {
                    "-InlineFromConstructors",
                    "When checking a constructor of a class 'T', inline every call 'this.m(...)' transitively from within the constructor, whenever 'm' statically resolves to a non-overridable instance method defined in 'T'." },
            {
                    "-InlineCheckDepth <depth>",
                    "When a method body is translated, at least <depth> levels of inlining of calls are performed.  The <depth> level of inlining is checked, while all previous levels are turned off.  By default, <depth> is zero.   This flag cannot be used with -inlineConstructors.  See also -inlineDepthPastCheck." },
            {
                    "-InlineDepthPastCheck <depth>",
                    "When a method body is translated, 'n' levels of inlining of calls are performed beyond the inlining level that is checked (see the -inlineCheckDepth option).  Checks at all 'n' levels are turned off.  By default 'n' is 0.  This flag cannot be used in combination with the -inlineConstructors flag." },
            {
                    "-InlineConstructors",
                    "For each non-static method 'm' and constructor 'c', generate and check a new method consisting of an inline call to 'c' followed immediately by 'm'.  This flag cannot be used in combination with -inlineCheckDepth or -inlineDepthPastCheck." },
            {
                    "-PrintCounterExampleContext, -pcc",
                    "Causes the full counterexample context to be displayed after each unsuccessful verification attempt." },
            {
                    "-NoStrongestPostconditionVC, -nospvc",
                    "Generate verification conditions using weakest preconditions, not strongest postconditions." },
            { "-Passify", "TBD" },
            { "-UseDefPred", "TBD" },
            {
                    "-NoDynamicSingleAssignment, -nodsa",
                    "Do not convert guarded command into dynamic single-assignment form before generating a VC.  This flag implies the -nospvc flag." },
            {
                    "-PrintDynamicSingleAssignment, -pdsa",
                    "For each method and constructor to be verified, the guarded-command translation of its body in dynamic single-assignment form is printed if available." },
            { "-PrintVC, -pvc", "Print the generated VC.  See also -ppvc." },
            { "-wpnxw <number>", "TBD" },
            { "-NamePCSize <number>", "TBD" },
            { "-CheckSpecs", "TBD" },
            {
                    "-PrintJavaWithTypes, -pjt",
                    "Prints out the Java source plus annotations, with a comment before each expression containing its Java static type." },
            {
                    "-NoVariableUniquification, -nvu",
                    "Print all name resolutions performed when using -pgc without location information." },
            {
                    "-AssertContinue, -ac",
                    "Experimental feature to try to make Houdini invocations refute more than one annotation in one pass." },
            {
                    "-NoTrackReadChars",
                    "Turns off recording of characters that come back from the prover, which makes unexpected Simplify output messages be less informative." },
            {
                    "-FilterMethodSpecs",
                    "Ignore routine preconditions, frame axioms, and postconditions that mention fields that are not spec-accessible." },
            {
                    "-NoPeepOptGCAssertFalse",
                    "Experimental patch for the benefit of Houdini.  Disengages the guarded command peephole optimization that removes what is sequentially composed after an 'assert false'." },
            { "-NoEPeepOpt", "Turns off peephold optimization for expressions." },
            { "-NoGCPeepOpt",
                    "Turns off peephole optimizations for guarded commands." },
            { "-LazySubst",
                    "Enable lazy substitutions.  Possibly uses less memory and more time." },
            {
                    "-MergeInv",
                    "Merge all invariants into a single predicate.  This improves checking speed at the cost of incorrect error report locations for invariant warnings." },
            {
                    "-NoAllocUseOpt",
                    "Causes the variable 'alloc' in the guarded-command encoding to be treated as possibly being modified by every routine call." },
            {
                    "-UseAllInvPostCall",
                    "Check all postconditions resulting from invariants even if there is no overlap between the free variables of the invariant and the modifies clause of the call." },
            {
                    "-UseAllInvPostBody",
                    "Check all postconditions resulting from invariants even if there is no overlap between the free variables of the invariant and the syntactic targets of the body being checked.  See also -UseAllInvPreBody." },
            {
                    "-UseAllInvPreBody",
                    "Check all preconditions resulting from invariants even if there is no overlap between the free variables of the invariant and the syntactic targets of the body being checked.  See also -UseAllInvPostBody." },
            {
                    "-FilterInvariants",
                    "Ignore invariants and axioms that mention variables that are not spec-accessible." },
            {
                    "-PrintAssumers",
                    "Prints the list of annotations that are in some way related to each routine.  This switch is provided for the benefit of Houdini." },
            { "-PrintCompilationUnitsOnLoad",
                    "Prints the name of each file as it is loaded." },
            {
                    "-GuardedVC <directory>",
                    "Write all guarded verification conditions to the specified directory, one file per VC." },
            { "-IgnoreAnnFile <filename>", "TBD" },
            { "-Skip", "TBD" },
            { "-PreciseTargets",
                    "; using this switch implies -loopSafe is enabled also" },
            {
                    "-LoopFallThru",
                    "Insert a break at the end of all loop unrolling, rather than the guarded command 'fail'.  (Probably undesirable.)" },
            { "-MapsUnrollCount <count>", "TBD" },
            { "-PredAbstract", "TBD" },
            { "-InferPredicates",
                    "; using this switch implies -loopSafe is enabled also" },
            {
                    "-NoDirectTargetsOpt",
                    "Uses normal targets, instead of direct targets, when intersecting the free variables on invariants to see if the invariant needs to be considered." },
            {
                    "-NestQuantifiers",
                    "Causes all user-supplied quantifiers to be nested in the translation, with one bound variable per quantification." },
            {
                    "-UseIntQuantAntecedents",
                    "Generates type antecedent for user-supplied quantified variables of type 'int' and 'long' (rather than just 'true')." },
            { "-ExcuseNullInitializers, -eni",
                    "Suppress the non-null check for explicit null initialization in constructors." },
            {
                    "-StrongAssertPostNever",
                    "Causes the strongest-postcondition based verification condition generation to never assume a condition after it has been asserted.  (Note, depending on Simplify's subsumption settings, a proved condition may still be assumed by Simplify.)  See also -StrongAssertPostAtomic and -StrongAssertPostAlways." },
            {
                    "-StrongAssertPostAtomic",
                    "Causes the strongest-postcondition based verification condition generation to, in essence, assume a condition after it has been asserted, provided the condition is 'simple', meaning a conjunction of atomic formulas.  This is the default behavior.  See also -StrongAssertPostNever and -StrongAssertPostAlways." },
            {
                    "-StrongAssertPostAlways",
                    "Causes the strongest-postcondition based verification condition generation to, in essence, always assume a condition after it has been asserted.  See also -StrongAssertPostAtomic and -StrongAssertPostNever." },
            {
                    "-NoLastVarUseOpt",
                    "Disables the dead variable analysis and optimization that are otherwise applied in the generation of the DSA form of guarded commands." },
            {
                    "-NoVarCheckDeclsAndUses",
                    "Dispense with the check that there are no multiply declared local variables, no free uses of variables outside their local-declaration blocks, and no duplicate dynamic-instance inflections, assumptions that DSA uses when making all local-declaration blocks implicit." },
            { "-Hidecall",
                    "Omit the details of calls when printing guarded commands." },
            { "-ShowLoop",
                    "Show all details of desugaring of loops when printing guarded commands." },
            { "-VerboseTrace",
                    "Print additional trace information.  See also -notrace." },
            {
                    "-BubbleNotDown, -bnd",
                    "Bubble down all logical NOTs in specification formulas to literals whenever possible." },
            {
                    "-BackPredFile <filename>, -bpf <filename>",
                    "Specifies an alternate file that should be used as the universal background predicate." },
            {
                    "-NoOutCalls",
                    "By default, we allow calls out from routines even when one or more objects have their invariants broken so long as those objects are not passed as (implicit) parameters or via static fields (in scope) to the callee.  This switch outlaws even those calls unless the only object broken is the object being constructed by the caller (a constructor)." },
            {
                    "-ParsePlus",
                    "Parse pragmas that begin with /*+@ which are normally only parsed by the core JML tools." },
            { "-NoNotCheckedWarnings",
                    "Do not print any warnings about constructs that are not checked." },
            {
                    "-TestRef",
                    "Pretty-print each compilation unit on the command-line; used primarily to test refinement synthesis." },
            {
                    "-CheckPurity",
                    "Enable checking of pure methods; currently disabled by default until issues with pure inheritance semantics are resolved." },
            {
                    "-RewriteDepth <depth>",
                    "Set the limit to the rewriting depth of non-functional method calls to <depth> calls." },
            { "-UseFcns", "TBD" },
            { "-UseVars", "TBD" },
            { "-UseFcnsForModelVars", "TBD" },
            { "-UseVarsForModelVars", "TBD" },
            { "-UseFcnsForMethods", "TBD" },
            { "-UseVarsForMethods", "TBD" },
            { "-ShowFields", "TBD" },
            { "-EscProjectFileV0", "TBD" },
            {
                    "-NonNullElements, -nne",
                    "Enable support for generating type-predicates for the \nonnullelements keyword.  Disabled by default." },
            {
                    "-ConsistencyCheck, -inChk",//conor Jan05 #testme
                    "Check for JML specification inconsistencies. Removes each specification one-at-a-time and rechecks. Currently requires the manual insertion of a false assert(assert false; will work). Incomplete implementation. The removed specifications are currently listed with a toString(). Needs to be made more presentable" },
            //$$
            {
                    "-nvcg <prover>[,<prover>]*",
                    "Use the new verification conditions generator and write the proof for a comma separated list of <prover>'s (eg. simplify, pvs, coq, etc.), as generated by the new verification condition generator, to a collection of files. Should <prover> have the form \"xml.<name>\", then we assume that the stylesheet escjava/vcGeneration/xml/<name>.xslt is to be used to target our prover." },
            { "-pErr", "Print errors generated by the vcg" },
            { "-pWarn", "Print warnings generated by the vcg" },
            { "-pInfo", "Print informations generated by the vcg" },
            {
                    "-pColors",
                    "Use colors to display messages generated by the verification conditions/proofs generator (should work on most Unix terminals w bash) [experimental]" },
            { "-vc2dot", "Output the gc tree in dot format" },
            { "-pToDot", "Output the translation of the gc tree in dot format" },
            {   "-svcg <prover>",
                "Use the yet-newer, sorted logic, verification condition generator " +
                "and invoke the <prover> (e.g., simplify, fx7, cvc etc.). " +
                "Prover resource flags can be added after colon, (e.g., fx7:TimeLimit=30,MaxQuantIters=100)."},
            //$$
            { "-ReachabilityAnalysis, -era", "Enable reachability analysis." },
            { "-ReachabilitySpecTest, -erst", "Enable reachability analysis spec testing." },
            { "-ReachabilitySpecTestAll, -ersta", "Enable reachability analysis spec testing, run on methods that have implementation and skip the implementation." },
	    { "-idc", "Check that assertions are defined (i.e. not undefined) in the sense of the new JML semantics."},
	    { "-debug", "Turn on for selected modules."},
	    { "-warnUnsoundIncomplete", "Turn on warnings about locations where ESC/Java2 reasons unsoundly or incompletely." }
    //$$
    };

    // Global escjava flags

    /*
     * These are set by the command-line switches of the same name.
     * They default to false (unset).
     */

    public String simplify = System.getProperty("simplify");

    //$$
    public static final String simplifyName = "Simplify";
    public static final String sammyName ="Sammy";
    public static final String harveyName = "haRVey";
    public static final String cvc3Name = "CVC3";
    private final String[] supportedProvers = { simplifyName, sammyName, 
        harveyName, cvc3Name };
    private final String[] defaultProvers =  { simplifyName };
    private Set enabledProvers;

    // use the new verification conditions generator
    public boolean nvcg = false;
    
    // use the even newer typed one 
    public boolean svcg = false;
    public Properties svcgProverResourceFlags = new Properties(); 

    // check "is-defined conditions" (IDCs) rather than normal specification correctness.
    public boolean idc = false;
    public boolean debug = false;

    // Non-null by default and non-null types ...
    public boolean nonNullByDefault = false;

    // print information about the vcg
    public boolean pErr = false;

    public boolean pWarn = false;

    public boolean pInfo = false;

    // use colors to add sense to output of vc generator (should work
    // on most unix terminals w bash) 
    public boolean pColors = false;

    // output vc generated by the new verification condition generator
    // to a file.
    public String[] pProver = null;

    // output gc tree in dot format
    public boolean vc2dot = false;

    // create dot tree of proof
    public boolean pToDot = false;

    //$$

    public boolean suggest = false;

    public boolean pjt = false; // print java w/ types

    public boolean nvu = false; // unknown functionality

    public boolean pgc = false; // print the guarded commands

    public boolean pdsa = false; // prints the single-assignment GCs

    public boolean pvc = false; // pretty-print the verification conditions

    public boolean pcc = false; // a pruned, pretty-printed version of pcc:

    public boolean counterexample = false;

    public boolean statsTime = false;

    public boolean statsSpace = false;

    public boolean statsTermComplexity = false;

    public boolean statsVariableComplexity = false;

    public boolean statsQuantifierComplexity = false;

    public boolean plainWarning = false;

    //conor Jan05 #testme
    public boolean consistencyCheck = false; //Check for JML inconsistencies when true

    public boolean useOldStringHandling = false; // Just for backwards compatibility for Esc/Java tests
    
    public boolean enableReachabilityAnalysis = false;

    /** JML allows only subtypes of Exception in signals clauses.  Thus signals
     clauses cannot be written about Errors.  Set this option to true to have
     annotations allow any Throwable.
     */
    public boolean useThrowable = false;

    /** The dirpath or jar file of system specs to use. */
    public String specspath = null;

    /** Statically check against redundant specs?  Default is true. */
    public boolean checkRedundantSpecs = true;

    // trace info: (0) no, (1) default, (2) verbose
    public int traceInfo = 1;

    //@ invariant 0 <= traceInfo && traceInfo < 3;

    /** When set, pretty-prints the VCs that are obtained with verbose output
     or in the log (-sxLog) */
    public boolean prettyPrintVC = false;

    /** When set, prints out the desugared specs for debugging purposes. */
    public boolean desugaredSpecs = false;

    /** When true, pretty prints each compilation unit on the command-line;
     this is only used for testing, to test the combining of refinements.
     */
    public boolean testRef = false;

    /** Temporary option to turn on purity checking, since it is off by
     default until purity issues with inheritance are resolved.
     */
    public boolean checkPurity = false;

    public boolean strictExceptions = false;

    /** When true, parses pragmas that begin with /*+@, which are normally
     parsed only by JML; this allows test runs in which everything JML
     parses is parsed by escjava, to see if we have full coverage of all
     of JML.
     */
    public boolean parsePlus = false;

    /** When true, does not print any warnings about things not checked. */
    public boolean noNotCheckedWarnings = false;

    /** JML requires semicolons to terminate annotations.  ESC/Java2 will
     warn about such missing semicolons; these were not required in
     ESC/Java.  When the following option is true, such warnings are
     suppressed to support old ESC/Java annotations.
     **/
    public boolean noSemicolonWarnings = false;

    //** The limit to the rewriting depth when handling non-functional method calls. */
    public int rewriteDepth = 2;

    public boolean spvc = true;

    public boolean dsa = true;

    //@ invariant spvc ==> dsa;  // spvc requires dsa for soundness
    public boolean passify = false;

    public boolean wpp = false;

    public boolean useDefpred = false;

    public int wpnxw = 0;

    public int namePCsize = 0;

    public boolean peepOptE = true;

    public boolean peepOptGC = true;

    public boolean lazySubst = false;

    public boolean mergeInv = false;

    public static final int JAVA_ASSERTIONS = 0;

    public static final int JML_ASSERTIONS = 1;

    public int assertionMode = JAVA_ASSERTIONS;

    public boolean useAllInvPostCall = false;

    public boolean useAllInvPostBody = false;

    public boolean useAllInvPreBody = false;

    public boolean allocUseOpt = true;

    public static final int LOOP_FAST = 0;

    public static final int LOOP_SAFE = 1;

    public static final int LOOP_FALL_THRU = 2;

    public int loopTranslation = LOOP_FAST;

    //@ invariant LOOP_FAST <= loopTranslation && loopTranslation <= LOOP_FALL_THRU;

    // Documentation above needs to changed if these values are changed.
    // The default loop unrolling is: -loop 1.5 
    public int loopUnrollCount = 1;

    public boolean loopUnrollHalf = true;

    public int mapsUnrollCount = 2;

    public boolean preciseTargets = false;

    public boolean predAbstract = false;

    public boolean inferPredicates = false;

    public boolean noDirectTargetsOpt = false;

    public boolean nestQuantifiers = false;

    public boolean lastVarUseOpt = true;

    public boolean noVarCheckDeclsAndUses = false;

    public boolean useIntQuantAntecedents = false;

    public boolean excuseNullInitializers = false;

    /** The following values are used: <br>
     *   0-never, 1-atomic (default), 2-always
     */
    public int strongAssertPost = 1;

    public boolean showCallDetails = true;

    public boolean showLoopDetails = false;

    public boolean bubbleNotDown = false;

    public int startLine = -1; // starting line # for processing

    public int vclimit = 400000;

    public boolean checkSpecs = false;

    public boolean noOutCalls = false;

    // flags primarily used by Houdini
    public boolean printAssumers = false;

    public boolean noPeepOptGCAssertFalse = false;

    public boolean assertContinue = false;

    public boolean trackReadChars = true;

    public boolean guardedVC = false;

    public String guardedVCDir;

    public String guardedVCFileNumbers = "files.txt";

    public String guardedVCGuardFile = "guards.txt";

    public String guardedVCPrefix = "G_";

    public String guardedVCFileExt = "sx";

    public Set guardVars = new Set();

    public String ClassVCPrefix = "*Class*";

    public String MethodVCPrefix = "*Method*";

    public Set ignoreAnnSet = null;

    public boolean printCompilationUnitsOnLoad = false;

    // Flags to control the treatment of model variables and routines calls
    public boolean useFcnsForModelVars = true;

    public boolean useFcnsForMethods = true;

    public boolean useFcnsForAllocations = true;

    // filter invariants and axioms
    public boolean filterInvariants = false;

    // filter method specifications
    public boolean filterMethodSpecs = false;

    public Set routinesToCheck = null; // null means "check everything"

    public Set routinesToSkip = null; // null means "skip nothing"

    // do the inlined constructor experiment?
    public boolean inlineConstructors = false;

    // do some other inlining experiment, using at least one of the two
    // inline depth flags?
    public boolean inlineDepthFlags = false;

    // when checking a constructor of a class T, inline every call
    // this.m(...) transitively from within the constructor, whenever m
    // statically resolves to a non-overridable instance method defined in T.
    public boolean inlineFromConstructors = false;

    public String sxLog = null;

    // The following switch hardcodes whether or not also_requires is to
    // be part of the annotation language
    public boolean allowAlsoRequires = true;

    // Debug flag that dumps the fields of each class
    public boolean showFields = false;
    
    public boolean warnUnsoundIncomplete = false;

    /**
     * Number of stages to run.  The stages currently in order are:
     * <pre>
     *     1. loading, parsing, and type checking
     *     2. generating the type-specific background predicate
     *        3. translating from Java to GC
     *        4. translating from GC to DSA
     *     5. generating the VC from the GC
     *     6. calling the theorem prover to verify the VC
     * </pre>
     *
     * <p> Defaults to running all stages (6); must be positive.
     *
     * <p> -nocheck sets <code>stages</code> to run all but the last
     * stage (5).  The -stages option can be used to set
     * <code>stages</code> to a particular value.
     */
    public int stages = 6;

    //* Which file to obtain the universal background predicate from.
    public String univBackPredFile = null;

    /**
     * Enable support for generating type-predicates for the
     * \nonnullelements keyword.  Disabled by default.
     */
    public boolean nne = false;

    // Generating an options message

    /**
     * Generate all command-line option information.
     *
     * @param all if true show all non-public (experimental) switches as well.
     * @return a String containing all command-line option information.
     */
    public String showOptions(boolean all) {
        String s = super.showOptions(all);
        // All public options.
        s += showOptionArray(escPublicOptionData);
        // All private options only if requested.
        if (all)
            s += showOptionArray(escPrivateOptionData);
        return s;
    }

    /**
     * @return non-option usage information in a string.
     */
    public String showNonOptions() {
        return super.showNonOptions();
    }


    // Prover settings

    /*@ pure */ public boolean isSupportedProver(/*@ non_null*/String prover) {
        return isMember(prover, supportedProvers);
    }
    /*@ pure */ public boolean isDefaultProver(/*@ non_null*/String prover) {
        return isMember(prover, defaultProvers);
    }

    //@ requires \nonnullelements(ps);
    /*@ pure*/ boolean isMember(/*@ non_null */String p, /*@ non_null*/String[] ps) {
        for (int i = 0; i < ps.length; i++) {
            if (p.compareToIgnoreCase(ps[i]) == 0) {
                return true;
            }
        }
        return false;
    }

    /** Determines whether the specified prover is enabled. 
     * @param proverName String identifying the prover, it has to be in the list of supported provers.
     */
    /*@ pure */boolean isProverEnabled(/*@ non_null */String proverName) {
        proverName = proverName.toLowerCase();
        Assert.notFalse(isSupportedProver(proverName), "Prover name " + proverName + " is not in the list of supported provers.");
        return enabledProvers.contains(proverName);
    }


    private void enableProver(/*@ non_null */String proverName) {
        enabledProvers.add(proverName.toLowerCase());
    }

    private void disableProver(String proverName) {
        enabledProvers.remove(proverName.toLowerCase());
    }

    /** Initialized the  <code>proverEnabled</code> table.  */
    private void initializeProverSetting() {
        enabledProvers = new Set();
        disableAllProvers();
        enableProvers(defaultProvers);
    }


    /** Disables all supported provers. */
    private void disableAllProvers() {
        enabledProvers.clear();
    }

    /** Enables the specified provers. */
    //@ requires \nonnullelements(ps);
    private void enableProvers(/*@ non_null */String[] ps) {
        for (int i = 0; i < ps.length; i++) {
            enableProver(ps[i]);
        }
    }

    // Option processing
    public static final Option optLoop            = Option.registerOption(new String[] {"-Loop"});
    public static final Option optLoopSafe        = Option.registerOption(new String[] {"-LoopSafe"});
    public static final Option optLoopFallThru    = Option.registerOption(new String[] {"-LoopFallThru"});
    public static final Option optPredAbstract    = Option.registerOption(new String[] {"-PredAbstract"});
    public static final Option optInferPredicates = Option.registerOption(new String[] {"-InferPredicates"});

    public static final Option optEaJava = Option.registerOption(new String[] {"-JavaAssertions", "-eaJava"});
    public static final Option optEaJML  = Option.registerOption(new String[] {"-JMLAssertions", "-eaJML"});

    public static final Option optProver = Option.registerOption(new String[] {"-Prover"});
    
    public static final Option optPlatform = Option.registerOption(new String[] {"-Platform"});


    public static final Option optERST = Option.registerOption(new String[] { "-ReachabilitySpecTest", "-erst"});
    public static final Option optERSTA = Option.registerOption(new String[] { "-ReachabilitySpecTestAll", "-ersta"});
    // TODO: exclude erst with idc, at the moment optERST doesn't let idc to operate --mikolas


    private static final Option[][] excludes = { 
        { optLoopSafe, optLoop, optLoopFallThru }, // options optLoopSafe, optLoop, optLoopFallThru are mutually excluded
        { optEaJava, optEaJML }
    };

    private OptionsConsistency consistency = new OptionsConsistency(excludes);

    //

    /**
     *  Query whether a certain option is on.
     *
     * @param o an option previously registered with <code>Option.registerOption</code>
     */
    public boolean isOptionOn(/*@non_null*/Option o) {
        //@ assert Option.isRegistered(o);
        return consistency.isOn(o);
    }

     public Options() {
         consistency.addInduces(optPredAbstract, new Option[] { optLoopSafe }); // options optPredAbstract implie loopSafe on
         consistency.addInduces(optInferPredicates, new Option[] { optPredAbstract });

         initializeProverSetting();
    }

    /**
     * Process next tool option.
     *
     * <p> See {@link javafe.Options#processOptions(String [])} for the complete
     * specification of this routine.
     */
    public int processOption(String option, String[] args, int offset) throws UsageError {
        Option regOption = Option.findOption(option);
        if (regOption != null) {
            // check if this option is not excluded by some other option
            OptionsConsistency.Conflict conflict = consistency.isConflict(regOption);
            if (conflict != null) 
                throw new UsageError("Option " + option + " cannnot be set, because " + conflict.getOpt2() + " conflicts with " + conflict.getOpt1() +  ".");

            // add the option to the on set
            consistency.set(regOption);
        }

        // First, change option to lowercase for case-less comparison.
        option = option.toLowerCase();

        if (option.equals("-nocheck")) {
            stages = 5; // run all but last stage
            return offset;
        } else if (option.equals("-typecheck")) {
            stages = 1;
            return offset;
        } else if (option.equals("-stages")) {
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            try {
                stages = new Integer(args[offset]).intValue();
                if (stages < 1)
                    throw new NumberFormatException();
            } catch (NumberFormatException e) {
                throw new UsageError("Option " + option
                        + " requires one positive integer argument: "
                        + e.toString());
            }
            return offset + 1;
        } else if (option.equals("-start")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            try {
                startLine = new Integer(args[offset]).intValue();
            } catch (NumberFormatException e) {
                throw new UsageError("Option " + option
                        + " requires one positive integer argument: "
                        + e.toString());
            }
            return offset + 1;
        } else if (option.equals("-specs")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one String argument");
            }
            specspath = args[offset];
            return offset + 1;
        } else if (option.equals("-vclimit")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            try {
                vclimit = new Integer(args[offset]).intValue();
            } catch (NumberFormatException e) {
                throw new UsageError("Option " + option
                        + " requires one positive integer argument: "
                        + e.toString());
            }
            return offset + 1;
        } else if (option.equals("-inlinefromconstructors")) {
            inlineFromConstructors = true;
            return offset;
        } else if (option.equals("-inlinecheckdepth")) {
            inlineDepthFlags = true;
            // can't use this along with the -inlineConstructors flag
            if (inlineConstructors)
                throw new UsageError(
                        "Can't use -inlineConstructors in combination with either -inlineCheckDepth or -inlineDepthPastCheck");
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            try {
                Main.gctranslator.inlineCheckDepth = new Integer(args[offset])
                        .intValue();
                if (Main.gctranslator.inlineCheckDepth < 0)
                    throw new NumberFormatException();
            } catch (NumberFormatException e) {
                throw new UsageError("Option " + option
                        + " requires one positive integer argument: "
                        + e.toString());
            }
            return offset + 1;
        } else if (option.equals("-inlinedepthpastcheck")) {
            inlineDepthFlags = true;
            // can't use this along with the -inlineConstructors flag
            if (inlineConstructors)
                throw new UsageError(
                        "can't use -inlineConstructors in combination with either -inlineCheckDepth or -inlineDepthPastCheck");
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            try {
                Main.gctranslator.inlineDepthPastCheck = new Integer(
                        args[offset]).intValue();
                if (Main.gctranslator.inlineDepthPastCheck < 0)
                    throw new NumberFormatException();
            } catch (NumberFormatException e) {
                throw new UsageError("Option " + option
                        + " requires one positive integer argument: "
                        + e.toString());
            }
            return offset + 1;
        } else if (option.equals("-inlineconstructors")) {
            // can't use this along with either of the inline depth flags
            if (inlineDepthFlags)
                throw new UsageError(
                        "can't use -inlineConstructors in combination with either -inlineCheckDepth or -inlineDepthPastCheck");
            inlineConstructors = true;
            return offset;
        } else if (option.equals("-suggest")) {
            suggest = true;
            return offset;
        } else if (option.equals("-pgc")
                || option.equals("-printguardedcommands")) {
            pgc = true;
            return offset;
        } else if (option.equals("-printcounterexamplecontext")
                || option.equals("-pcc")) {
            pcc = true;
            return offset;
        } else if (option.equals("-noStrongestPostconditionVC")
                || option.equals("-nospvc")) {
            spvc = false;
            return offset;
        } else if (option.equals("-passify")) {
            passify = true;
            return offset;
        } else if (option.equals("-wpp")) {
            wpp = true;
            return offset;
        } else if (option.equals("-usedefpred")) {
            useDefpred = true;
            return offset;
        } else if (option.equals("-noDynamicSingleAssignment")
                || option.equals("-nodsa")) {
            dsa = false;
            spvc = false; // spvc requires dsa for soundness
            return offset;
        } else if (option.equals("-printDynamicSingleAssignment")
                || option.equals("-pdsa")) {
            pdsa = true;
            return offset;
        } else if (option.equals("-printVC") || option.equals("-pvc")) {
            pvc = true;
            prettyPrintVC = true;
            return offset;
        }
        //$$
        else if (optProver.isMe(option)) {
            if ((offset >= args.length) || (args[offset].charAt(0) == '-')) {
                throw new UsageError(
                        "Option "
                                + option
                                + " requires one argument or more indicating which provers you want to use.\n"
                                + "(e.g., '-Prover simplify,sammy')");
            }
            // split the provided argument by commas into individual provers
            String[] provers = args[offset].split(",");

            // test if the given provers are supported 
            for (int i = 0; i < provers.length; i++) {
                if (!isSupportedProver(provers[i])) {
                    throw new UsageError("Unknown prover: '" + provers[i] + "', when processing the option " + option + ".");
                }
            }

            // override default settings
            disableAllProvers();
            enableProvers(provers);

            return offset + 1;
        } else if (option.equals("-nvcg")) {
            nvcg = true;

            if ((offset >= args.length) || (args[offset].charAt(0) == '-')) {
                throw new UsageError(
                        "Option "
                                + option
                                + " requires a comma separated argument indicating which prover(s) you want to use.\n"
                                + "(e.g. \"" + option + " pvs\" or \"" + option + " simplify,coq\")");
            }

            pProver = new String(args[offset]).split(",");
            
            return offset+1;

        } else if (option.equals("-svcg")) {
            svcg = true;  
            
            if ((offset >= args.length) || (args[offset].charAt(0) == '-')) {
                throw new UsageError(
                        "Option " + option
                                + " requires an argument indicating which prover you want to use.\n"
                                + "(e.g. \"" + option + " simplify\" or \"" + option + " fx7:TimeLimit=10\")");
            }
            
            String prover =args[offset];
        	int idx = prover.indexOf(':');
        	
            if (idx > 0) {
            	String options = prover.substring(idx + 1);
            	prover = prover.substring(0, idx);
            	String[] opts = options.split(",");
            	for (int i = 0; i < opts.length; ++i) {
            		idx = opts[i].indexOf('='); 
            		if (idx > 0) {
            			svcgProverResourceFlags.setProperty(
            					opts[i].substring(0, idx), 
            					opts[i].substring(idx + 1));
            		} else {
                        throw new UsageError(
                                "Prover parameter in -svcg is supposed to have name=value format, got: " + opts[i]);            			
            		}            		
            	}            		
            }

            pProver = new String[] { prover };
            
            return offset+1;
            
        } else if (option.equals("-perr")) {
            pErr = true;

            return offset;
        } else if (option.equals("-pcolors")) {
            pColors = true;

            return offset;
        } else if (option.equals("-pwarn")) {
            pWarn = true;

            return offset;
        } else if (option.equals("-pinfo")) {
            pInfo = true;

            return offset;
        } else if (option.equals("-vc2dot")) {
            vc2dot = true;

            return offset;
        } else if (option.equals("-ptodot")) {
            pToDot = true;

            return offset;
        } else if (option.equals("-idc")) {
            idc = true;

            return offset;
        } else if (option.equals("-debug")) {
            debug = true;

            return offset;
        }
        //$$
        else if (option.equals("-wpnxw")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            wpnxw = new Integer(args[offset]).intValue();
            return offset + 1;

        } else if (option.equals("-namepcsize")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            namePCsize = new Integer(args[offset]).intValue();
            return offset + 1;
        } else if (option.equals("-nonnullbydefault")) {
        	nonNullByDefault = true;
        	return offset;
        } else if (option.equals("-stats")) {

            /* no more parameter or last parameter 
             * , default behavior is to consider
             * keyword 'time' & 'space' as activated.
             */
            if ((offset >= args.length) || (args[offset].charAt(0) == '-')) {
                statsTime = true;
                statsSpace = true;

                return offset;
            } else { /* parse the string after the stats keyword 
             which shouldn't contain any space
             the usual way to handle it is to store the string in a field
             (look at guardedVCDir for example)
             and use it in the main, but this way is nicer I think. [Clement]
             */
                if (args[offset].charAt(args[offset].length() - 1) == ',')
                    throw new UsageError("Argument to option '" + option
                            + "' is finished by ',' which is not correct.");

                String[] s = args[offset].split(",");
                String sTemp = null;
                int i = 0;

                for (; i < s.length; i++) {

                    sTemp = s[i];

                    //@ assert sTemp.length() != 0;

                    if (sTemp.equals("time"))
                        statsTime = true;
                    else if (sTemp.equals("space"))
                        statsSpace = true;
                    else if (sTemp.equals("complexity")) {
                        // default behavior for complexity implemented here
                        statsTermComplexity = true;
                        statsVariableComplexity = true;
                        statsQuantifierComplexity = true;
                    } else if (sTemp.equals("termComplexity"))
                        statsTermComplexity = true;
                    else if (sTemp.equals("variableComplexity"))
                        statsVariableComplexity = true;
                    else if (sTemp.equals("quantifierComplexity"))
                        statsQuantifierComplexity = true;
                    else
                        throw new UsageError("Argument to the option '"
                                + option + "' not recognized : '" + s
                                + "', skipping it");

                }

                return offset + 1;

            } // </else> 

        } else if (option.equals("-checkspecs")) {
            checkSpecs = true;
            return offset;
        } else if (option.equals("-printjavawithtypes")
                || option.equals("-pjt")) {
            pjt = true;
            return offset;
        } else if (option.equals("-novariableuniquification")
                || option.equals("-nvu")) {
            nvu = true;
            return offset;
        } else if (option.equals("-ac") || option.equals("-assertcontinue")) {
            assertContinue = true;
            return offset;
        } else if (option.equals("-notrackreadchars")) {
            trackReadChars = false;
            return offset;
        } else if (option.equals("-filtermethodspecs")) {
            filterMethodSpecs = true;
            return offset;
        } else if (option.equals("-nopeepoptgcassertfalse")) {
            noPeepOptGCAssertFalse = true;
            return offset;
        } else if (option.equals("-noepeepopt")) {
            peepOptE = false;
            return offset;
        } else if (option.equals("-nogcpeepopt")) {
            peepOptGC = false;
            return offset;
        } else if (option.equals("-lazysubst")) {
            lazySubst = true;
            return offset;
        } else if (option.equals("-mergeinv")) {
            mergeInv = true;
            return offset;
        } else if (option.equals("-noallocuseopt")) {
            allocUseOpt = false;
            return offset;
        } else if (option.equals("-useallinvpostcall")) {
            useAllInvPostCall = true;
            return offset;
        } else if (option.equals("-useallinvpostbody")) {
            useAllInvPostBody = true;
            return offset;
        } else if (option.equals("-useallinvprebody")) {
            useAllInvPreBody = true;
            return offset;
        } else if (option.equals("-filterinvariants")) {
            filterInvariants = true;
            return offset;
        } else if (option.equals("-printassumers")) {
            printAssumers = true;
            return offset;
        } else if (option.equals("-printcompilationunitsonload")) {
            printCompilationUnitsOnLoad = true;
            return offset;
        } else if (option.equals("-guardedvc")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one directory argument");
            }
            guardedVC = true;
            guardedVCDir = args[offset];
            return offset + 1;
        } else if (option.equals("-ignoreannfile")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one filename argument");
            }
            ignoreAnnSet = new Set(readFile(args[offset]).elements());
            // System.out.println("Ignore set: "+ignoreAnnSet+"\n");
            return offset + 1;
        } else if (option.equals("-routine")) {
            // the argument to "-routine" is either a simple routine name or a fully
            // qualified routine name with signature, but we won't ever parse these
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            String routine = args[offset].intern();
            if (routinesToCheck == null) {
                routinesToCheck = new Set();
            }
            routinesToCheck.add(routine);
            return offset + 1;
        } else if (option.equals("-skip")) {
            // the argument to "-skip" is either a simple routine name or a fully
            // qualified routine name with signature, but we won't ever parse these
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            String routine = args[offset].intern();
            if (routinesToSkip == null) {
                routinesToSkip = new Set();
            }
            routinesToSkip.add(routine);
            return offset + 1;
        } else if (option.equals("-routineindirect")) {
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            if (routinesToCheck == null) {
                routinesToCheck = new Set();
            }
            String routineIndirectionFilename = args[offset];
            BufferedReader in = null;
            try {
                in = new BufferedReader(new FileReader(
                        routineIndirectionFilename));
                while (true) {
                    String routine = in.readLine();
                    if (routine == null) {
                        break;
                    }
                    routine = routine.intern();
                    routinesToCheck.add(routine);
                }
            } catch (IOException e) {
                throw new UsageError("error reading routine indirection file '"
                        + routineIndirectionFilename + "': " + e.toString());
            } finally {
                try {
                    if (in != null)
                        in.close();
                } catch (IOException e) {
                    throw new UsageError(
                            "error closing routine indirection file '"
                                    + routineIndirectionFilename + "': "
                                    + e.toString());
                }
            }
            return offset + 1;
        } else if (optLoopSafe.isMe(option)) {
            loopTranslation = LOOP_SAFE;
            return offset;
        } else if (option.equals("-precisetargets")) {
            preciseTargets = true;
            loopTranslation = LOOP_SAFE;
            return offset;
        } else if (optLoop.isMe(option)) {
            // syntax:  -loop <N>[.0|.5]
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            String number = args[offset];
            if (number.length() == 0 || option.charAt(0) == '.') {
                throw new UsageError("illegal argument to -loop: " + number);
            }
            // any other parsing error will be caught in the following loop
            int cnt = 0;
            boolean andAHalf = false;
            for (int i = 0; i < number.length(); i++) {
                char ch = number.charAt(i);
                if ('0' <= ch && ch <= '9') {
                    if (10000 <= cnt) {
                        throw new UsageError(
                                "-loop specifies too many iterations: "
                                        + number);
                    }
                    cnt = 10 * cnt + ch - '0';
                    continue;
                } else if (ch == '.') {
                    if (number.length() == i + 2) {
                        if (number.charAt(i + 1) == '5') {
                            andAHalf = true;
                            break;
                        } else if (number.charAt(i + 1) == '0') {
                            andAHalf = false;
                            break;
                        }
                    }
                }
                throw new UsageError("illegal argument to -loop: " + number);
            }
            loopUnrollCount = cnt;
            loopUnrollHalf = andAHalf;
            return offset + 1;
        } else if (optLoopFallThru.isMe(option)) {
            loopTranslation = LOOP_FALL_THRU;
            return offset;
        } else if (option.equals("-mapsunrollcount")) {
            if (offset == args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            mapsUnrollCount = Integer.parseInt(args[offset]);
            return offset + 1;
        } else if (optPredAbstract.isMe(option)) {
            predAbstract = true;
            loopTranslation = LOOP_SAFE;
            lastVarUseOpt = false;
            return offset;
        } else if (optInferPredicates.isMe(option)) {
            inferPredicates = true;
            predAbstract = true;
            loopTranslation = LOOP_SAFE;
            lastVarUseOpt = false;
            return offset;
        } else if (option.equals("-nodirecttargetsopt")) {
            noDirectTargetsOpt = true;
            return offset;
        } else if (option.equals("-nestquantifiers")) {
            nestQuantifiers = true;
            return offset;
        } else if (option.equals("-useintquantantecedents")) {
            useIntQuantAntecedents = true;
            return offset;
        } else if (option.equals("-excusenullinitializers")
                || option.equals("-eni")) {
            excuseNullInitializers = true;
            return offset;
        } else if (option.equals("-strongassertpostnever")) {
            strongAssertPost = 0;
            return offset;
        } else if (option.equals("-strongassertpostatomic")) {
            strongAssertPost = 1;
            return offset;
        } else if (option.equals("-strongassertpostalways")) {
            strongAssertPost = 2;
            return offset;
        } else if (option.equals("-nolastvaruseopt")) {
            lastVarUseOpt = false;
            return offset;
        } else if (option.equals("-novarcheckdeclsanduses")) {
            noVarCheckDeclsAndUses = true;
            return offset;
        } else if (option.equals("-hidecall")) {
            showCallDetails = false;
            return offset;
        } else if (option.equals("-showloop")) {
            showLoopDetails = true;
            return offset;
        } else if (option.equals("-plainwarning")) {
            plainWarning = true;
            return offset;
        } else if (option.equals("-noredundancy")) {
            checkRedundantSpecs = false;
            return offset;
        } else if (option.equals("-notrace")) {
            traceInfo = 0;
            return offset;
        } else if (option.equals("-verbosetrace")) {
            traceInfo = 2;
            return offset;
        } else if (option.equals("-consistencycheck")
                || option.equals("-inchk")) {
            consistencyCheck = true;//conor Jan05 #testme
            return offset;
        } else if (option.equals("-counterexample")) {
            counterexample = true;
            return offset;
        } else if (option.equals("-bubblenotdown") || option.equals("-bnd")) {
            bubbleNotDown = true;
            return offset;
        } else if (option.equals("-nooutcalls")) {
            noOutCalls = true;
            return offset;
        } else if (option.equals("-backpredfile") || option.equals("-bpf")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            univBackPredFile = args[offset];
            return offset + 1;
        } else if (option.equals("-sxlog")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            sxLog = args[offset];
            return offset + 1;
        } else if (option.equals("-pxlog")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            sxLog = args[offset];
            prettyPrintVC = true;
            return offset + 1;
        } else if (option.equals("-nowarn")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            if (args[offset].equals("All")) {
                NoWarn.setAllChkStatus(TagConstants.CHK_AS_ASSUME);
            } else {
                int tag = NoWarn.toNoWarnTag(args[offset]);
                if (tag == 0) {
                    throw new UsageError("'" + args[offset]
                            + "' is not a legal warning category");
                }
                NoWarn.setChkStatus(tag, TagConstants.CHK_AS_ASSUME);
            }
            return offset + 1;
        } else if (option.equals("-warn")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one argument");
            }
            // Note:  There's an intentional lack of symmetry with -nowarn
            // here, in that "-warn All" is not supported.  This design
            // allows ESC/Java to include special checks, perhaps like the
            // Stale checks of the Higher-Level Races checker, that won't
            // be publicly advertised.
            int tag = NoWarn.toNoWarnTag(args[offset]);
            if (tag == 0) {
                throw new UsageError("'" + args[offset]
                        + "' is not a legal warning category");
            }
            NoWarn.setChkStatus(tag, TagConstants.CHK_AS_ASSERT);
            return offset + 1;
        } else if (option.equals("-parseplus")) {
            parsePlus = true;
            return offset;
        } else if (option.equals("-nonotcheckedwarnings")) {
            noNotCheckedWarnings = true;
            return offset;
        } else if (option.equals("-testref")) {
            testRef = true;
            return offset;
        } else if (option.equals("-strictexceptions")) {
            strictExceptions = true;
            return offset;
        } else if (option.equals("-checkpurity")) {
            checkPurity = true;
            return offset;
        } else if (option.equals("-nocheckpurity")) {
            checkPurity = false;
            return offset;
        } else if (option.equals("-ppvc") || option.equals("-prettyprintvc")) {
            prettyPrintVC = true;
            return offset;
        } else if (option.equals("-sds")
                || option.equals("-showdesugaredspecs")) {
            desugaredSpecs = true;
            return offset;
        } else if (optEaJava.isMe(option)) {
            assertionMode = JAVA_ASSERTIONS;
            assertionsEnabled = true;
            assertIsKeyword = true;
            return offset;
        } else if (optEaJML.isMe(option)) {
            assertionMode = JML_ASSERTIONS;
            assertIsKeyword = true;
            assertionsEnabled = true;
            return offset;
        } else if (option.equals("-rewritedepth")) {
            if (offset >= args.length) {
                throw new UsageError("Option " + option
                        + " requires one integer argument");
            }
            rewriteDepth = Integer.parseInt(args[offset]);
            return offset + 1;
        } else if (option.equals("-nosemicolonwarnings")) {
            noSemicolonWarnings = true;
            return offset;
        } else if (option.equals("-usefcns")) {
            useFcnsForModelVars = true;
            useFcnsForMethods = true;
            useFcnsForAllocations = true;
            return offset;
        } else if (option.equals("-usevars")) {
            useFcnsForModelVars = false;
            useFcnsForMethods = false;
            useFcnsForAllocations = false;
            return offset;
        } else if (option.equals("-usefcnsformodelvars")) {
            useFcnsForModelVars = true;
            return offset;
        } else if (option.equals("-usevarsformodelvars")) {
            useFcnsForModelVars = false;
            return offset;
        } else if (option.equals("-usefcnsformethods")) {
            useFcnsForMethods = true;
            return offset;
        } else if (option.equals("-usevarsformethods")) {
            useFcnsForMethods = false;
            return offset;
        } else if (option.equals("-showfields")) {
            showFields = true;
            return offset;
        } else if (option.equals("-simplify")) {
            // FIXME - shcek for additional argument
            simplify = args[offset];
            //System.setProperty("simplify",args[offset]);
            return offset + 1;
        } else if (option.equals("-escprojectfilev0")) {
            // Ignored, but also used to mark a project file.
            return offset;
        } else if (option.equals("-nonnullelements") || option.equals("-nne")) {
            nne = true;
            return offset + 1;
        } else if (option.equals("-useoldstringhandling")) {
            useOldStringHandling = true;
            return offset;
        } else if (option.equals("-usethrowable")) {
            useThrowable = true;
            return offset;
        } else if (option.equals("-warnunsoundincomplete")){
        	warnUnsoundIncomplete = true;
        	return offset;
        } else if (option.equals("-reachabilityanalysis") || option.equals("-era")) {
            enableReachabilityAnalysis = true;
            noPeepOptGCAssertFalse = true;
            return offset;
        } else if (optERST.isMe(option)) {
            // TODO: do I need to do anything here? --mikolas
            return offset;
        }  else if (optERSTA.isMe(option)) {
            // TODO: do I need to do anything here? --mikolas
            return offset;
        } 



        // Pass on unrecognized options:
        return super.processOption(option, args, offset);
    }

    private Vector readFile(String filename) {
        Vector r = new Vector();
        StringBuffer s = new StringBuffer();
        Reader R = null;
        try {
            R = new BufferedReader(new InputStreamReader(new FileInputStream(
                    filename)));
            int c = 0;
            do {
                while ((c = R.read()) != -1 && c != '\n') {
                    s.append((char) c);
                }
                // Delete from 3rd space on
                String st = s.toString();
                int i = st.indexOf(' ');
                if (i != -1)
                    i = st.indexOf(' ', i + 1);
                if (i != -1)
                    i = st.indexOf(' ', i + 1);
                if (i != -1)
                    st = st.substring(0, i);
                r.addElement(st);
                s.setLength(0);
            } while (c != -1);
            return r;
        } catch (IOException e) {
            throw new RuntimeException("IOException: " + e);
        } finally {
            try {
                if (R != null)
                    R.close();
            } catch (IOException e) {
                throw new RuntimeException("IOException: " + e);
            }
        }
    }

    public String nowarnOptionString() {
        StringBuffer sb = new StringBuffer(200);
        for (int i = escjava.ast.TagConstants.FIRSTESCCHECKTAG; i < escjava.ast.TagConstants.CHKQUIET; ++i) {
            if (NoWarn.getChkStatus(i) == escjava.ast.TagConstants.CHK_AS_ASSUME) {
                sb.append(" -nowarn ");
                sb.append(escjava.ast.TagConstants.toString(i));
            }
        }
        return sb.toString();
    }
} // end of class Options

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */

