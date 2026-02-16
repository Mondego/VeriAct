/* Copyright 2000, 2001, Compaq Computer Corporation */
// Modified 2006-2007, Systems Research Group, University College Dublin

package escjava;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.Vector;
import java.io.*;

import javafe.InputEntry;
import javafe.ast.*;
import escjava.ast.*;
import javafe.tc.OutsideEnv;
import escjava.ast.ASTVisitor;
import escjava.ast.EscPrettyPrint;
import escjava.ast.TagConstants;
import escjava.ast.Modifiers;

import escjava.backpred.FindContributors;
import escjava.completeness.FloatingPointCompletenessVisitor;
import escjava.dfa.daganalysis.ReachabilityAnalysis;
import escjava.dfa.daganalysis.SpecTester;
import escjava.gui.GUI;
import escjava.AnnotationHandler;

import javafe.reader.StandardTypeReader;
import escjava.reader.EscTypeReader;

import javafe.parser.PragmaParser;

import escjava.sp.*;
import escjava.translate.*;
import escjava.pa.*;

import javafe.tc.TypeSig;
import escjava.tc.TypeCheck;

import escjava.prover.*;

import escjava.vcGeneration.*;

import javafe.util.*;

import escjava.sortedProver.Lifter;
import escjava.soundness.*;

/**
 * Top level control module for ESC for Java.
 *
 * <p>This class (and its superclasses) handles parsing
 * <code>escjava</code>'s command-line arguments and orchestrating the
 * other pieces of the front end and escjava to perform the requested
 * operations.<p>
 *
 * @see javafe.Tool
 * @see javafe.SrcTool
 */

public class Main extends javafe.SrcTool {
  static public final/*@ nullable */String jarlocation; // can be null

  static {
    ClassLoader loader = GUI.class.getClassLoader();
    //@ assert loader != null;
    java.net.URL urlJar = loader.getResource("escjava/Main.class");
    //@ assert urlJar != null;
    String urlStr = urlJar.toString();
    //@ assert urlStr != null;
    int from = "jar:file:".length();
    int to = urlStr.indexOf("!/");
    if (to != -1) {
      String j = urlStr.substring(from, to);
      int k = j.lastIndexOf('/');
      if (k != -1)
        j = j.substring(0, k);
      jarlocation = j;
    } else {
      jarlocation = null;
    }
    // This does not produce a good guess for the distribution
    // root when running within the CVS distribution - just
    // when running from a jar file.
    //System.out.println("LOCATION " + urlStr + " " + jarlocation);
  }

  {
    // Makes sure that the escjava.tc.Types factory instance is loaded
    escjava.tc.Types.init();
  }

  /** Our version number */
  //public final static String version = "(Nijmegen/Kodak) 1.3, 2003";
  public final static/*@ non_null @*/String version = Version.VERSION;

  private/*@ non_null @*/AnnotationHandler annotationHandler = new AnnotationHandler();

  // Convenience copy of options().stages
  public int stages;

  /**
   * Return the name of this tool.  E.g., "ls" or "cp".<p>
   *
   * Used in usage and error messages.<p>
   */
  public/*@ non_null @*/String name() {
    return "escjava";
  }

  public/*@ non_null @*/javafe.Options makeOptions() {
    return new Options();
  }

  //@ requires javafe.Tool.options != null;
  public /*@pure*/ static /*@non_null*/ Options options() {
    return (/*+@non_null*/Options) options;
  }

  // Front-end setup

  /**
   * Returns the Esc StandardTypeReader, EscTypeReader.
   */
  // All three arguments can be null.
  public/*@ non_null @*/StandardTypeReader makeStandardTypeReader(
                                                                   String path,
                                                                   String sourcePath,
                                                                   PragmaParser P) {
    return EscTypeReader.make(path, sourcePath, P, annotationHandler);
  }

  /**
   * Returns the EscPragmaParser.
   */
  public/*@ non_null @*/javafe.parser.PragmaParser makePragmaParser() {
    return new escjava.parser.EscPragmaParser();
  }

  /**
   * Returns the pretty printer to set
   * <code>PrettyPrint.inst</code> to.
   */
  public/*@ non_null @*/PrettyPrint makePrettyPrint() {
    DelegatingPrettyPrint p = new EscPrettyPrint();
    p.setDel(new StandardPrettyPrint(p));
    return p;
  }

  /**
   * Called to obtain an instance of the javafe.tc.TypeCheck class
   * (or a subclass thereof). May not return <code>null</code>.  By
   * default, returns <code>javafe.tc.TypeCheck</code>.
   */
  public/*@ non_null @*/javafe.tc.TypeCheck makeTypeCheck() {
    return new escjava.tc.TypeCheck();
  }

  /**
   * Override SrcTool.notify to ensure all lexicalPragmas get
   * registered as they are loaded.
   */
  //@ also
  //@   requires justLoaded != null;
  public void notify(CompilationUnit justLoaded) {
    super.notify(justLoaded);

    NoWarn.registerNowarns(justLoaded.lexicalPragmas);

    if (options().printCompilationUnitsOnLoad) {
      String pkgName = justLoaded.pkgName == null ? "" : justLoaded.pkgName
          .printName();
      String filename = Location.toFileName(justLoaded.loc);
      System.out.println("LOADED: " + pkgName + " " + filename);
    }
  }

  // Main processing code

  //Store Instance of class for use with Consistency checking (Soundness Package) Conor 05

  private static Main instance = null;

  public static Main getInstance() {
    return instance;
  }

  /**
   * Start up an instance of this tool using command-line arguments
   * <code>args</code>. <p>
   *
   * This is the main entry point for the <code>escjava</code>
   * command.<p>
   */
  //@ requires \nonnullelements(args);
  public static void main(/*@ non_null @*/String[] args) {

    int exitcode = compile(args);
    if (exitcode != 0)
      System.exit(exitcode);
  }

  public Main() {
    // resets any static variables left from a previous instantiation
    clear(true);
  }

  boolean keepProver = false;

  public void clear(boolean complete) {
    // restore ordinary checking of assertions
    super.clear(complete);
    if (complete)
      NoWarn.init();
    gctranslator = new Translate();
    if (!keepProver)
      ProverManager.kill();
    // Disallow the -avoidSpec option:
    javafe.SrcToolOptions.allowAvoidSpec = false;
    javafe.util.LocationManagerCorrelatedReader.clear();
    escjava.translate.NoWarn.clear();
  }

  /**
   * An entry point for the tool useful for executing tests,
   * since it returns the exit code.
   * 
   * @param args The command-line arguments the program was invoked with
   * @return The exit code for the program, indicating either a successful 
   * 		exit or an exit with errors or an exit because of an out of
   * 		memory condition
   * @see javafe.Tool#run(java.lang.String[])
   */
  /*@ ensures \result == okExitCode || \result == badUsageExitCode
    @      || \result == errorExitCode || \result == outOfMemoryExitCode;
  */

  public static int compile(/*@non_null*/String[] args) {
    try {
      Main t = new Main();
      instance = t;
      int result = t.run(args);
      return result;
    } catch (OutOfMemoryError oom) {
      Runtime rt = Runtime.getRuntime();
      long memUsedBytes = rt.totalMemory() - rt.freeMemory();
      System.out.println("java.lang.OutOfMemoryError (" + memUsedBytes
                         + " bytes used)");
      //oom.printStackTrace(System.out);
      return outOfMemoryExitCode;
    }
  }

  // SrcTool-instance specific processing

  /** An instance of the GC->VC translator */
  public static Translate gctranslator = new Translate();

  /**
   * Override setup so can issue version # as soon as possible (aka,
   * just after decode options so know if -quiet or -testMode issued or not).
   */
  public void setup() {
    stages = options().stages;
    if (options().simplify == null)
      setDefaultSimplify();
    if (options().simplify != null)
      System.setProperty("simplify", options().simplify);
    super.setup();

    //$$
    ProverManager.useSimplify = options().isProverEnabled(Options.simplifyName);
    ProverManager.useSammy = options().isProverEnabled(Options.sammyName);
    ProverManager.useHarvey = options().isProverEnabled(Options.harveyName);
    ProverManager.useCvc3 = options().isProverEnabled(Options.cvc3Name);
    ;
    ProverManager.useSorted = options().svcg;
    ProverManager.sortedProvers = options().pProver;
    //$$

    if (!options().quiet) {
      System.out.print("ESC/Java version "
                       + (options().testMode ? "VERSION" : version));

      System.out.print("\n");
    }

  }

  public void setDefaultSimplify() {
    String os = System.getProperty("os.name");
    String root = null;
    String name = "Simplify-1.5.";
    if (os.startsWith("Windows")) {
      root = name + "4.exe";
    } else if (os.startsWith("Mac")) {
      root = name + "5.macosx"; // Use universal binary for Mac OS X
    } else if (os.startsWith("Linux")) {
      root = name + "4.linux";
    } else if (os.startsWith("Solaris")) {
      root = name + "4.solaris";
    } else {
      ErrorSet.warning("Unknown OS - could not find Simplify: " + os);
    }
    if (root == null)
      return;

    File f;
    if (jarlocation == null)
      f = new File(root);
    else
      f = new File(jarlocation, root);
      
    // Make it easier to find Simplify when running tests from Maven
    if (!f.exists()) {
     f = new File("Escjava/release/master/bin/",root);
    }

    if (!f.exists()) {
      ErrorSet
          .warning("Could not find a default SIMPLIFY executable - specify the path to SIMPLIFY for this operating system using the -simplify option[ "
                   + os + " " + root + "]");
      return;
    }
    try {
      options().simplify = f.getCanonicalPath();
    } catch (IOException e) {
      ErrorSet
          .warning("Could not find a default SIMPLIFY executable - specify the path to SIMPLIFY for this operating system using the -simplify option[ "
                   + os + " " + root + "]");
      return;
    }
  }

  public void setupPaths() {
    super.setupPaths();
    if (options().specspath == null)
      return;
    if (compositeSourcePath == null) {
      compositeClassPath = options().specspath + java.io.File.pathSeparator
                           + compositeClassPath;
    } else {
      compositeSourcePath = options().specspath + java.io.File.pathSeparator
                            + compositeSourcePath;
    }
  }

  /**
   * Hook for any work needed before <code>handleCU</code> is called
   * on each <code>CompilationUnit</code> to process them.
   */
  public void preprocess() {

    if (ErrorSet.fatals > 0) {
      ErrorSet.fatal(null);
    }

    // call our routines to run the constructor inlining experiment
    if (options().inlineConstructors)
      InlineConstructor.inlineConstructorsEverywhere(loaded);

    //if (6 <= stages || options().predAbstract) {
    //ProverManager.start();
    //}

  }

  /**
   * A wrapper for opening output files for printing.
   *
   * dir can be null.
   */
  //@ ensures \result != null;
  private PrintStream fileToPrintStream(String dir, /*@ non_null @*/
                                        String fname) {
    File f = new File(dir, fname);
    try {
      return new PrintStream(new FileOutputStream(f));
    } catch (IOException e) {
      javafe.util.ErrorSet.fatal(e.getMessage());
      return null; // unreachable
    }
  }

  public void postload() {
    super.postload();
    if (OutsideEnv.filesRead() == 0) {
      ErrorSet.error("No files read.");
    }
  }

  /**
   * Hook for any work needed after <code>handleCU</code> has been
   * called on each <code>CompilationUnit</code> to process them.
   */
  public void postprocess() {

    // If we are in the Houdini context (guardedVC is true), output
    // the association of file numbers to their names.
    // Also, dump out a list of guard variable names.
    if (options().guardedVC) {
      PrintStream o = fileToPrintStream(options().guardedVCDir,
                                        options().guardedVCFileNumbers);
      Vector v = LocationManagerCorrelatedReader.fileNumbersToNames();
      for (int i = 0; i < v.size(); i++)
        o.println(i + " " + v.elementAt(i));
      o.close();

      o = fileToPrintStream(options().guardedVCDir,
                            options().guardedVCGuardFile);
      Enumeration e = options().guardVars.elements();
      while (e.hasMoreElements()) {
        o.println((String) e.nextElement());
      }
      o.close();
    }

    if (!keepProver)
      ProverManager.kill();
  }

  /**
   * Check each Java class specification for soundness and completeness
   * 
   * @param classunit
   *
   */
protected void checkSoundnessAndCompleteness(CompilationUnit classunit) {
    
   // Soundness and Completeness Visitors
   ASTVisitor[] visitors = registerVisitors();
   
   // Apply the soundness and completeness visitors
   for (int visitorNumber = 0; visitorNumber < visitors.length; visitorNumber++) {
     final ASTVisitor visitor = visitors[visitorNumber];
     if (visitor != null) visitor.visitASTNode(classunit);
   }
}

/**
 * Determine which Soundness and Completeness warnings to show
 * 
 * @return Array of Soundness and Completeness Warning Visitors
 */
protected /*@ non_null */ ASTVisitor[] registerVisitors() {
   ASTVisitor[] visitors = new ASTVisitor[7];
   int index = 0;
   visitors[index++] = new RShiftVisitor();
   visitors[index++] = new LShiftVisitor();
   visitors[index++] = new LoopSoundnessVisitor();
   visitors[index++] = new FloatingPointCompletenessVisitor();
   visitors[index++] = new ModificationSoundnessVisitor();
   visitors[index++] = new ObjectInvariantSoundnessVisitor();
   visitors[index++] = new TrustingPragmaSoundnessVisitor();
   return visitors;
}

  /**
   * This method is called on each <code>CompilationUnit</code> that
   * this tool processes.  This method overrides the implementation
   * given in the superclass, adding a couple of lines before the
   * superclass implementation is called.
   */
  public void handleCU(CompilationUnit cu) {
    if (options().testRef)
      makePrettyPrint().print(System.out, cu);

    NoWarn.setStartLine(options().startLine, cu);

    UniqName.setDefaultSuffixFile(cu.getStartLoc());
    try {
      super.handleCU(cu);
    } catch (FatalError e) {
      // Errors are already reported
      //ErrorSet.report("Aborted processing " + cu.sourceFile().getHumanName() + " because of fatal errors");
    }

    options().startLine = -1; // StartLine applies only to first CU
    
    if (options().warnUnsoundIncomplete) {
       // Calls the visitors that detect specification unsoundness or
       // incompleteness in the compilation unit
       
       checkSoundnessAndCompleteness(cu);
     }
  }

/**
   * This method is called by SrcTool on the TypeDecl of each
   * outside type that SrcTool is to process.
   *
   * <p> In addition, it calls itself recursively to handle types
   * nested within outside types.
   */
  //@ also
  //@ requires td != null;
  public void handleTD(TypeDecl td) {
    long startTime = currentTime();
    TypeSig sig = TypeCheck.inst.getSig(td);

    if (!options().quiet)
      System.out.println("\n" + sig.toString() + " ...");

    /* If something is on the command-line, presume we want to check it
       as thoroughly as possible.
       if (sig.getTypeDecl().specOnly &&
       !options().checkSpecs) {    // do not process specs
       // No bodies to process
       if (!options().quiet) System.out.println("Skipping " + 
       sig.toString() + " - specification only");
       return;
       }
    */

    if (Location.toLineNumber(td.getEndLoc()) < options().startLine)
      return;

    // Do actual work:
    boolean aborted = processTD(td);

    if (!options().quiet)
      System.out.println("  [" + timeUsed(startTime) + " total]"
                         + (aborted ? " (aborted)" : ""));

    /*
     * Handled any nested types:  [1.1]
     */
    TypeDecl decl = sig.getTypeDecl();
    for (int i = 0; i < decl.elems.size(); i++) {
      if (decl.elems.elementAt(i) instanceof TypeDecl)
        handleTD((TypeDecl) decl.elems.elementAt(i));
    }
  }

  /**
   * Run all the requested stages on a given TypeDecl; return true
   * if we had to abort.
   *
   */
  //@ requires td != null;
  //@ requires (* td is not from a binary file. *);
  private boolean processTD(TypeDecl td) {
    try {

      // ==== Start stage 1 ====

      /*
       * Do Java type checking then print Java types if we've been
       * asked to:
       */
      long startTime = currentTime();
      int errorCount = ErrorSet.errors;
      TypeSig sig = TypeCheck.inst.getSig(td);
      sig.typecheck();
      NoWarn.typecheckRegisteredNowarns();

      if (options().pjt) {
        // Create a pretty-printer that shows types
        DelegatingPrettyPrint p = new javafe.tc.TypePrint();
        p.setDel(new EscPrettyPrint(p, new StandardPrettyPrint(p)));

        System.out.println("\n**** Source code with types:");
        p.print(System.out, 0, td);
      }

      // Turn off extended static checking and abort if any errors
      // occured while type checking *this* TypeDecl:
      if (errorCount < ErrorSet.errors) {
        if (stages > 1) {
          stages = 1;
          ErrorSet.caution("Turning off extended static checking "
                           + "due to type error(s)");
        }
        return true;
      }

      // ==== Start stage 2 ====
      if (stages < 2)
        return false;

      // Generate the type-specific background predicate
      errorCount = ErrorSet.errors;
      if (Info.on)
        Info.out("[ Finding contributors for " + sig + "]");
      FindContributors scope = new FindContributors(sig);
      VcToString.resetTypeSpecific();

      if (Info.on)
        Info.out("[ Found contributors for " + sig + "]");

      if (options().guardedVC) {
        String locStr = UniqName.locToSuffix(td.locId);
        String fn = locStr + ".class." + options().guardedVCFileExt;
        File f = new File(options().guardedVCDir, fn);
        PrintStream o = fileToPrintStream(options().guardedVCDir, fn);
        o.println(options().ClassVCPrefix);
        o.println(td.id + "@" + locStr);
        o.print("\n(BG_PUSH ");
        escjava.backpred.BackPred bp = new escjava.backpred.BackPred();
        bp.genTypeBackPred(scope, o);
        o.println(")");
        o.close();
      }

      // Turn off extended static checking and abort if any type errors
      // occured while generating the type-specific background predicate:
      if (errorCount < ErrorSet.errors) {
        stages = 1;
        ErrorSet.caution("Turning off extended static checking "
                         + "due to type error(s)");
        return true;
      }

      if (options().testRef)
        makePrettyPrint().print(System.out, 0, td);

      // ==== Start stage 3 ====
      if (3 <= stages) {

        if (6 <= stages || options().predAbstract) {
          if (options().svcg)
            ProverManager.kill();
          ProverManager.push(scope);
        }

        LabelInfoToString.reset();
        InitialState initState = new InitialState(scope);
        LabelInfoToString.mark();

        if (!options().quiet)
          System.out.println("    [" + timeUsed(startTime) + "]");

        // Process the elements of "td"; stage 3 continues into stages 4
        // and 5 inside processTypeDeclElem:
        if (options().inlineConstructors && !Modifiers.isAbstract(td.modifiers)) {
          // only process inlined versions of methods
          for (int i = 0; i < td.elems.size(); i++) {
            TypeDeclElem tde = td.elems.elementAt(i);
            if (!InlineConstructor.isConstructorInlinable(tde)
                || InlineConstructor
                    .isConstructorInlinedMethod((MethodDecl) tde))
              processTypeDeclElem(tde, sig, initState);
          }
        } else {
          for (int i = 0; i < td.elems.size(); i++)
            processTypeDeclElem(td.elems.elementAt(i), sig, initState);
        }
      }
    } catch (FatalError e) {
      // Error already reported
      throw e;
    } catch (Throwable e) {
      System.out.println("Exception " + e + " thrown while processing "
                         + TypeSig.getSig(td));
      e.printStackTrace(System.out);
      return true;
    } finally {
      // ==== all done; clean up ====
      ProverManager.pop();
    }
    return false;
  }

  /**
   * Run stages 3+..6 as requested on a TypeDeclElem.
   *
   * requires te is not from a binary file, sig is the
   * TypeSig for te's parent, and initState != null.
   */
  //@ requires sig != null && initState != null;
  private void processTypeDeclElem(TypeDeclElem te, TypeSig sig,
                                   InitialState initState) {
    // Only handle methods and constructors here:
    if (!(te instanceof RoutineDecl))
      return;
    RoutineDecl r = (RoutineDecl) te;

    long startTime = java.lang.System.currentTimeMillis();
    if (!options().quiet) {
      String name = TypeCheck.inst.getRoutineName(r)
                    + javafe.tc.TypeCheck.getSignature(r);
      System.out.println("\n" + sig.toString() + ": " + name + " ...");
    }

    // Do the actual work, handling not implemented exceptions:
    String status = "error";

    ///////////////////////////////////////////////////////
    ///     Remove one of this RoutineDecl 's           ///
    ///     annotations and continue,                   ///
    ///     each time returning results                 ///
    ///     (and annotation removed)        ##Incomplete///
    ///////////////////////////////////////////////////////

    if (options().consistencyCheck) {

      Consistency c = new Consistency();
      c.consistency(r, sig, initState, startTime);
    } else {

      try {
        status = processRoutineDecl(r, sig, initState);
      } catch (javafe.util.NotImplementedException e) {
        // continue - problem already reported
        status = "not-implemented";
      } catch (FatalError e) {
        // continue;
      }

      if (!options().quiet)
        System.out.println("    [" + timeUsed(startTime) + "]  " + status);

      /*************************
         System.out.println("Lines " +
             (Location.toLineNumber(r.getEndLoc())
                 -Location.toLineNumber(r.getStartLoc()))
                 +" time "+timeUsed(startTime));
      *******************/

    }

  }


  private void specTest(RoutineDecl r, TypeSig sig, InitialState initState) {
      // method fabrication
      MethodDecl testMethod =  
          SpecTester.fabricateTest(r, sig, initState); // create the testing method

      // get GC for the fabricated method
      TrAnExpr.initForRoutine();
      FindContributors scope = new FindContributors(testMethod);
      GuardedCmd testBody = gctranslator.trBody(testMethod, scope, initState.getPreMap(), 
                                                predictSynTargs(testMethod, initState, scope),
                                                null,
                                                /* issueCautions */true);
      Spec spec = GetSpec.getSpecForBody(testMethod, scope, Targets.normal(testBody), initState.getPreMap());
      //TODO: mimic target computation in computeBody--mikolas
      GetSpec.addAxioms(Translate.axsToAdd, spec.preAssumptions);

      // prepending the GC with assumptions (for invariants)
      StackVector code = new StackVector();
      code.push();
      GetSpec.addAssumptions(spec.preAssumptions, code);
      GetSpec.assumeConditions(spec.pre, code);
      code.addElement(testBody);

      GuardedCmd testGC =  GC.seq(GuardedCmdVec.popFromStackVector(code));
      //      GuardedCmd testGC = computeBody(testMethod, initState);
      if (options().pgc) {
          System.out.println("\n**** Fabricated Guarded Command:");
          ((EscPrettyPrint) PrettyPrint.inst).print(System.out, 0, testGC);
          System.out.println();
      }

      // run reachability on the fabricated GC
      SpecTester.runReachability(testGC, r.getStartLoc()); 
  }


  /**
   * Run stages 3+..6 as requested on a RoutineDeclElem; returns a
   * short (~ 1 word) status message.
   *
   * requires - r is not from a binary file, sig is the TypeSig
   * for r's parent.
   */
  public /*@non_null*/String processRoutineDecl(/*@ non_null */RoutineDecl r,
  /*@ non_null */TypeSig sig,
  /*@ non_null */InitialState initState) {
    // ==== skip method according to option setting ====

    boolean checkBodyless = false; // should pass immediately routines with no body

    // or analyses that process specs only, by default bodyless are skipped
    checkBodyless |= Main.options().idc;
    checkBodyless |= options().isOptionOn(Options.optERST) && SpecTester.knowHowToCheck(r);
    checkBodyless |= options().isOptionOn(Options.optERSTA) && SpecTester.knowHowToCheck(r);

    if (r.body == null && !checkBodyless)
        return "passed immediately";

    if (r.parent.specOnly && !checkBodyless)
        return "passed immediately";

    if (Location.toLineNumber(r.getEndLoc()) < options().startLine)
        return "skipped";


    String simpleName = TypeCheck.inst.getRoutineName(r).intern();
    String fullName = sig.toString() + "." + simpleName
                      + javafe.tc.TypeCheck.getSignature(r);
    fullName = removeSpaces(fullName).intern();
    if (options().routinesToSkip != null
        && (options().routinesToSkip.contains(simpleName) || options().routinesToSkip
            .contains(fullName))) {
      return "skipped";
    }
    if (options().routinesToCheck != null
        && !options().routinesToCheck.contains(simpleName)
        && !options().routinesToCheck.contains(fullName)) {
      return "skipped";
    }

    // ==== actual analysis ====

    // === experimental for SpecTester, blame mikolas (Nov 2007)
    if ( (options().isOptionOn(Options.optERSTA) || options().isOptionOn(Options.optERST) ) &&  SpecTester.knowHowToCheck(r) ) {
        if (r.body==null || options().isOptionOn(Options.optERSTA)) { // TODO: what exactly should we skip?--mikolas
              if (!options().quiet)
                  System.out.println("          running reachability-based spec-checker"); // TODO: use standard logging mechanism--mikolas

              specTest(r, sig, initState);
              return "processed by reachability-based spec-checker (only); a bug has been found something  unreachable";
          }
    }
    // === end of experimental for SpecTester



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
    /*-@ uninitialized @*//* readable_if stats; */int origgcSize = 0;
    if (options().statsTime || options().statsSpace) {
      origgcSize = Util.size(gc);
    }

    String gcTime = timeUsed(startTime);
    startTime = java.lang.System.currentTimeMillis();

    UniqName.resetUnique();

    if (gc == null)
      return "passed immediately";
    if (options().pgc) {
      System.out.println("\n**** Guarded Command:");
      ((EscPrettyPrint) PrettyPrint.inst).print(System.out, 0, gc);
      System.out.println("");
    }

    Set directTargets = Targets.direct(gc);
    GCSanity.check(gc);

    // ==== Start stage 4 ====
    if (stages < 4)
      return "ok";

    // Convert GC to DSA:

    String dsaTime = "";
    if (options().dsa) { // always true
      /*
       * From experiements from POPL01 (Cormac)
       gc = passify ? Passify.compute(gc) : DSA.dsa(gc);
      */
      gc = DSA.dsa(gc);
      dsaTime = timeUsed(startTime);
      startTime = java.lang.System.currentTimeMillis();

      if (options().pdsa) {
        System.out.println("\n**** Dynamic Single Assignment:");
        ((EscPrettyPrint) PrettyPrint.inst).print(System.out, 0, gc);
        System.out.println("");
      }
    }

    // ==== Start stage 5 ====
    if (stages < 5)
      return "ok";

    // Generate the VC for GC:
    Expr vcBody;
    /* From experiements from POPL01 (Cormac)
     if(wpnxw != 0 ) {
     vcBody = WpName.compute( gc, wpnxw );
     } else 
    */
    if (options().spvc) {
      /* From experiements from POPL01 (Cormac)
      vcBody = wpp ? Wpp.compute(gc, GC.truelit, GC.truelit) : 
      SPVC.compute(gc);
      */
      vcBody = SPVC.compute(gc);
    } else {
      vcBody = Ejp.compute(gc, GC.truelit, GC.truelit);
    }

    String label = "vc." + sig.toString() + ".";
    if (r instanceof MethodDecl)
      label += ((MethodDecl) r).id;
    else
      label += "<constructor>";
    label += "." + UniqName.locToSuffix(r.getStartLoc());

    /* Use the new vc generator (= nvcg) */
    if (options().nvcg)
      useNewVCgen(r, sig, initState, vcBody, label);

    Expr vc = GC.implies(initState.getInitialState(), vcBody);

    vc = LabelExpr.make(r.getStartLoc(), r.getEndLoc(), false, Identifier
        .intern(label), vc);

    // Check for VC too big:
    int usize = Util.size(vc, options().vclimit);
    if (usize == -1) {
      ErrorSet.caution("Unable to check " + TypeCheck.inst.getName(r)
                       + " of type " + TypeSig.getSig(r.parent)
                       + " because its VC is too large");
      return "VC too big";
    }
    //System.err.println("main_vc_size " + Util.size(vc));
    //System.err.println("initial_vc_size " + Util.size(initState.getInitialState()));

    if (options().printAssumers) {
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

    if (!options().svcg
        && (options().pvc || (Info.on && options().traceInfo > 0))) {
      VcToString.compute(vc, System.out);
    }

    if (options().guardedVC) {

      String fn = UniqName.locToSuffix(r.locId) + ".method."
                  + options().guardedVCFileExt;
      PrintStream o = fileToPrintStream(options().guardedVCDir, fn);
      o.println(options().MethodVCPrefix);
      o.println(r.parent.id + "@" + UniqName.locToSuffix(r.parent.locId));
      VcToString.compute(vc, o);
      o.close();
      return "guarded VC generation finished";
    }

    String vcTime = timeUsed(startTime);
    startTime = java.lang.System.currentTimeMillis();

    // ==== Start stage 6 ====
    if (stages < 6)
      return "ok";

    // Process Simplify's output
    String status = "unexpectedly missing Simplify output";
    try {
      int stat = doProving(vc, r, directTargets, null);
      switch (stat) {
      case Status.STATICCHECKED_OK:
        status = "passed";
        break;
      case Status.STATICCHECKED_ERROR:
        status = "failed";
        break;
      case Status.STATICCHECKED_TIMEOUT:
        status = "timed out";
        break;
      default:
        status = "unexpectedly missing Simplify output";
      }

    } catch (escjava.prover.SubProcess.Died e) {
      // System.out.println("DIED");
      ProverManager.died();
    } catch (FatalError e) {
      // System.out.println("DIED");
      ProverManager.died();
    }

    if (options().enableReachabilityAnalysis) {
      // Gives warnings for unreached code (assertions for now)
      if (options().dsa) {
        ReachabilityAnalysis.analyze(gc);
      } else {
        ErrorSet
            .caution("Skipping reachability analysis because DSA is turned off.");
      }
    }

    String proofTime = timeUsed(startTime);
    if (options().statsTime) {
      System.out.println("    [Time: " + timeUsed(routineStartTime) + " GC: "
                         + gcTime + " DSA: " + dsaTime + " Ejp: " + ejpTime
                         + " VC: " + vcTime + " Proof(s): " + proofTime + "]");
    }
    if (options().statsSpace) {
      System.out.println("    [Size: " + " src: " + Util.size(r) + " GC: "
                         + origgcSize + " DSA: " + Util.size(gc) + " VC: "
                         + Util.size(vc) + "]");
    }
    if (options().statsTermComplexity)
      System.out
          .println("    [Number of terms: " + VcToString.termNumber + "]");
    if (options().statsVariableComplexity)
      System.out.println("    [Number of variables: "
                         + VcToString.variableNumber + "]");
    if (options().statsQuantifierComplexity)
      System.out.println("    [Number of quantifiers: "
                         + VcToString.quantifierNumber + "]");

    return status;
  }

private void useNewVCgen(RoutineDecl r, TypeSig sig, InitialState initState,
		Expr vcBody, String label) {
	VcGenerator vcg = null;
      String[] subpackage = options().pProver;
      for (int spindex = 0; spindex < subpackage.length; spindex++) {
        String className = "";
        String proverName = "";
        String[] spname = subpackage[spindex].split("\\.");
        try {
          for (int index = 0; index < spname.length; index++) {
            className = className + spname[index].substring(0, 1).toUpperCase()
                        + spname[index].substring(1);
          }
          proverName = className;
          if (className.startsWith("Xml")) {
            proverName = "Xml";
          }
          ProverType prover = null;
          if (subpackage[spindex].startsWith("xml.")) {
            String stylesheet = subpackage[spindex].substring(4);
            prover = (ProverType) (Class
                .forName("escjava.vcGeneration.xml.XmlProver").newInstance());
            ((escjava.vcGeneration.xml.XmlProver) prover)
                .setStyleSheet(stylesheet);
          } else {
            prover = (ProverType) (Class.forName("escjava.vcGeneration."
                                                 + subpackage[spindex] + "."
                                                 + className + "Prover")
                .newInstance());
          }
          String method = sig.toString() + ".";
          if (r instanceof MethodDecl)
            method += ((MethodDecl) r).id;
          else
            method += "<constructor>";
          System.out.println("[" + proverName + "Prover: generating VC for "
                             + method + "]");
          Expr vc = prover.addTypeInfo(initState, vcBody);
          vcg = new VcGenerator(prover, vc, options().pErr, options().pWarn,
                                options().pInfo, options().pColors);

          // write the proof generated by the new vcg to a file
          String fn = UniqName.locToSuffix(r.locId);
          fn = fn + ".p" + className;

          FileWriter fw = new FileWriter(fn);

          vcg.getProof(fw, prover.labelRename(label));

          fw.close();

          System.out.println("[" + proverName + "Prover: "
                             + subpackage[spindex] + " VC has been written to "
                             + fn + "]");
        } catch (escjava.vcGeneration.xml.XmlProverException exn) {
          System.out.println("[XmlProver: can not locate '"
                             + exn.stylesheet
                             + ".xslt' within ESC/Java or within the "
                             + (System.getProperty("XMLPROVERPATH") == null
                                || System.getProperty("XMLPROVERPATH")
                                    .equals("") ? "current working directory"
                                 : "system property XMLPROVERPATH (ie. "
                                   + System.getProperty("XMLPROVERPATH") + ")")
                             + "]");

        } catch (ClassNotFoundException exn) {
          System.out
              .println("["
                       + proverName
                       + "Prover: \""
                       + subpackage[spindex]
                       + "\" not recognised - ensure that you have specified the correct prover]");
        } catch (Exception e) {
          if (e.getMessage() != null) {
            System.out.println("[" + proverName + "Prover: " + e.getMessage()
                               + "]");
          }
          e.printStackTrace();
        }
      }
      // generate the dot file for the original vc tree
      if (options().vc2dot) {
        try {
          String fn = UniqName.locToSuffix(r.locId);
          fn = fn + ".vc.dot";

          FileWriter fw = new FileWriter(fn);

          /* initialization of dot format */
          fw.write("digraph G {\n");

          fw.write(vcg.old2Dot());

          /* end of dot file */
          fw.write("\n}\n");
          fw.close();

          /* run the appropriate commad to generate the graph */
          Runtime run = Runtime.getRuntime();

          run.exec("dot -Tps " + fn + " -o " + fn + ".ps");

          System.out.println("[Graph of the original vc tree for method "
                             + UniqName.locToSuffix(r.locId)
                             + " have been written to " + fn + ".ps]");

        } catch (Exception e) {
          System.out.println(e.getMessage());
        }
      }

      /* 
       * generate the tree of the proof
       */
      if (options().pToDot) {
        try {
          String fn = UniqName.locToSuffix(r.locId);
          fn = fn + ".proof.dot";

          FileWriter fw = new FileWriter(fn);

          /* initialization of dot format */
          fw.write("digraph G {\n");

          /* generate the graph by visiting the tree */
          fw.write(vcg.toDot());

          /* end of dot file */
          fw.write("\n}\n");
          fw.close();

          /* run the appropriate command to generate the graph */
          Runtime run = Runtime.getRuntime();

          run.exec("dot -Tps " + fn + " -o " + fn + ".ps");

          // if(options().pColors){
          // 			if(
          // 		    }

          System.out.println("[Graph of generic proof have been written to "
                             + fn + ".ps]");

        } catch (Exception e) {
          System.out.println(e.getMessage());
        }
      }
}

  public int doProving(/*@non_null*/Expr vc, RoutineDecl r, Set directTargets,
                       /*@nullable*/FindContributors scope) {
	  try {
		  String simpleName = TypeCheck.inst.getRoutineName(r).intern();
		  String fullName = TypeCheck.inst.getSig(r.parent).toString() + "."
		  					+ simpleName + javafe.tc.TypeCheck.getSignature(r);
		  fullName = removeSpaces(fullName).intern();

		  Enumeration results = ProverManager.prove(vc, scope, fullName);

		  // $$
		  if (!(ProverManager.useSimplify || ProverManager.useSorted))
			  return 0; // PC: should use one of the statuses, not a hard-coded value.

		  // Process Simplify's output
		  String status = "unexpectedly missing Simplify output";
		  int stat = Status.NOTPROCESSED;

		  boolean nextWarningNeedsPrecedingLine = true;
		  if (results != null) {
			  while (results.hasMoreElements()) {
				  SimplifyOutput so = (SimplifyOutput) results.nextElement();
				  switch (so.getKind()) {
				  case SimplifyOutput.VALID:
					  status = "passed";
					  stat = Status.STATICCHECKED_OK;
					  break;
				  case SimplifyOutput.INVALID:
					  status = "failed";
					  stat = Status.STATICCHECKED_ERROR;
					  break;
				  case SimplifyOutput.UNKNOWN:
					  status = "timed out";
					  stat = Status.STATICCHECKED_TIMEOUT;
					  break;
				  case SimplifyOutput.COMMENT: {
					  SimplifyComment sc = (SimplifyComment) so;
					  System.out.println("SIMPLIFY: " + sc.getMsg());
					  break;
				  }
				  case SimplifyOutput.COUNTEREXAMPLE: {
					  if (nextWarningNeedsPrecedingLine) {
						  escjava.translate.ErrorMsg
						  .printSeparatorLine(System.out);
						  nextWarningNeedsPrecedingLine = false;
					  }
					  SimplifyResult sr = (SimplifyResult) so;
					  escjava.translate.ErrorMsg.print(TypeCheck.inst
							  .getName(r), sr.getLabels(), sr.getContext(),
							  r, directTargets, System.out);
					  break;
				  }
				  case SimplifyOutput.EXCEEDED_PROVER_KILL_TIME: {
					  SimplifyResult sr = (SimplifyResult) so;
					  ErrorSet.caution("Unable to check "
							  + TypeCheck.inst.getName(r) + " of type "
							  + TypeSig.getSig(r.parent)
							  + " completely because too much time required");
					  if (Info.on && sr.getLabels() != null) {
						  Info.out("Current labels: " + sr.getLabels());
					  }
					  nextWarningNeedsPrecedingLine = true;
					  if (stat == Status.NOTPROCESSED) {
						  status = "timed out";
						  stat = Status.STATICCHECKED_TIMEOUT;
					  }
					  break;
				  }
				  case SimplifyOutput.EXCEEDED_PROVER_KILL_ITER: {
					  SimplifyResult sr = (SimplifyResult) so;
					  ErrorSet.caution("Unable to check "
							  + TypeCheck.inst.getName(r) + " of type "
							  + TypeSig.getSig(r.parent)
							  + " completely because"
							  + " too many iterations required");
					  if (Info.on && sr.getLabels() != null) {
						  Info.out("Current labels: " + sr.getLabels());
					  }
					  nextWarningNeedsPrecedingLine = true;
					  break;
				  }
				  case SimplifyOutput.REACHED_CC_LIMIT:
					  ErrorSet.caution("Not checking "
							  + TypeCheck.inst.getName(r) + " of type "
							  + TypeSig.getSig(r.parent)
							  + " completely because"
							  + " warning limit (PROVER_CC_LIMIT) reached");
					  break;
				  case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_TIME: {
					  SimplifyResult sr = (SimplifyResult) so;
					  ErrorSet.caution("Unable to check subgoal of "
							  + TypeCheck.inst.getName(r) + " of type "
							  + TypeSig.getSig(r.parent)
							  + " completely because too much time required");
					  if (Info.on && sr.getLabels() != null) {
						  Info.out("Current labels: " + sr.getLabels());
					  }
					  if (stat == Status.NOTPROCESSED) {
						  status = "timed out";
						  stat = Status.STATICCHECKED_TIMEOUT;
					  }
					  nextWarningNeedsPrecedingLine = true;
					  break;
				  }
				  case SimplifyOutput.EXCEEDED_PROVER_SUBGOAL_KILL_ITER: {
					  SimplifyResult sr = (SimplifyResult) so;
					  ErrorSet.caution("Unable to check subgoal of "
							  + TypeCheck.inst.getName(r) + " of type "
							  + TypeSig.getSig(r.parent)
							  + " completely because"
							  + " too many iterations required");
					  if (Info.on && sr.getLabels() != null) {
						  Info.out("Current labels: " + sr.getLabels());
					  }
					  nextWarningNeedsPrecedingLine = true;
					  break;
				  }
				  case SimplifyOutput.WARNING_TRIGGERLESS_QUANT: {
					  TriggerlessQuantWarning tqw = (TriggerlessQuantWarning) so;
					  int loc = tqw.getLocation();
					  /*
					   * Turn off this warning for now. FIXME Some generated
					   * axioms require using the Simplify heuristic to work
					   * correctly, while others generate this warning if
					   * there is no explict quantifier. String msg =
					   * "Unable to use quantification because " +
					   * "no trigger found: " + tqw.e1; if (loc !=
					   * Location.NULL) { ErrorSet.caution(loc, msg); } else {
					   * ErrorSet.caution(msg); } if (Info.on &&
					   * tqw.getLabels() != null) {
					   * Info.out("Current labels: " + tqw.getLabels()); }
					   */
					  break;
				  }
				  default:
					  Assert.fail("unexpected type of Simplify output");
				  break;
				  }
			  }
		  }
		  return stat;
	  } catch (escjava.prover.SubProcess.Died e) {
		  // status = "died";
		  return Status.STATICCHECKED_ERROR;
	  }
  }

  Set predictSynTargs(RoutineDecl r, InitialState initState, FindContributors scope) {
    /*
     * Compute an upper bound for synTargs if -O7 given.
     *
     * For now, do this via the kludge of calling trBody...  !!!!
     */
    Set predictedSynTargs = null;
    if (!options().useAllInvPreBody) {
      long T = java.lang.System.currentTimeMillis();
      /*
       * Compute translation assuming synTargs is empty:
       * (gives same set of targets faster than using null)
       */
      GuardedCmd tmpBody;
      if (r.body == null && Main.options().idc) {
        tmpBody = null;
        predictedSynTargs = new Set();
      } else {
        tmpBody = gctranslator.trBody(r, scope, initState.getPreMap(),
        /*predictedSynTargs*/new Set(), null,
        /* issueCautions */false);
        if (options().noDirectTargetsOpt)
          predictedSynTargs = Targets.normal(tmpBody);
        else
          predictedSynTargs = Targets.direct(tmpBody);
      }
      if (options().statsTime)
        System.out.println("      [prediction time: " + timeUsed(T) + "]");
    }

    return predictedSynTargs;
  }

  /**
   * This method computes the guarded command (including assuming
   * the precondition, the translated body, the checked
   * postcondition, and the modifies constraints) for the method or
   * constructor <code>r</code> in scope <code>scope</code>.
   *
   * @return <code>null</code> if <code>r</code> doesn't have a body.
   */

  protected /*@nullable*/ GuardedCmd computeBody(/*@non_null*/RoutineDecl r, /*@non_null*/InitialState initState) {
    if (r.getTag() == TagConstants.METHODDECL && ((MethodDecl) r).body == null
        && !Main.options().idc) {
      // no body
      return null;
    }

    // don't check the routine if it's a helper
    if (Helper.isHelper(r)) {
      return null;
    }

    FindContributors scope = new FindContributors(r);
    TrAnExpr.initForRoutine();

    Set predictedSynTargs = predictSynTargs(r, initState, scope);

    /*
     * Translate the body:
     */
    /* Note: initState.preMap is the same for all declarations.
       This may be overkill (FIXME).
       It might be better to use information from scope directly
       since it is generated from the routine decl.
       However, I don't know for sure what would go missing.  DRCok
    */
    GuardedCmd body;
    Set fullSynTargs;
    Set synTargs;
    // Denotes whether the method has body or not
    // used in GetSpec.surroundBodyBySpec()
    boolean nobody = false;
    if (r.body == null && Main.options().idc) {
      GuardedCmd gc3 = GC.skip();
      nobody = true;
      //GuardedCmd gc2=GC.assume(GC.falselit);
      //GuardedCmd gc3=GC.seq(gc1,gc2);
      body = gc3;
      if (r.getTag() == TagConstants.CONSTRUCTORDECL) {
        // get java.lang.Object
        TypeSig obj = escjava.tc.Types.javaLangObject();
        FieldDecl owner = null; // make the compiler happy
        boolean found = true;
        boolean save = escjava.tc.FlowInsensitiveChecks.inAnnotation;
        try {
          escjava.tc.FlowInsensitiveChecks.inAnnotation = true;
          owner = escjava.tc.Types.lookupField(obj, Identifier.intern("owner"),
                                               obj);
        } catch (javafe.tc.LookupException e) {
          found = false;
        } finally {
          escjava.tc.FlowInsensitiveChecks.inAnnotation = save;
        }
        // if we couldn't find the owner ghost field, there's nothing to do
        if (found) {
          VariableAccess ownerVA = TrAnExpr.makeVarAccess(owner, Location.NULL);
          Expr ownerNull = GC.nary(TagConstants.REFEQ, GC.select(ownerVA,
                                                                 GC.resultvar),
                                   GC.nulllit);
          GuardedCmd gcOwner = GC.assume(ownerNull);
          body = GC.seq(gc3, gcOwner);
        }
      }
      fullSynTargs = new Set();
      synTargs = new Set();
    } else {

      body = gctranslator.trBody(r, scope, initState.getPreMap(),
                                 predictedSynTargs, null,
                                 /* issueCautions */true);
      fullSynTargs = Targets.normal(body);
      if (options().noDirectTargetsOpt)
        synTargs = fullSynTargs;
      else
        synTargs = Targets.direct(body);

    }

    /*
     * Verify predictedSynTargs if present that
     * synTargs is a subset of predictedSynTargs.
     */
    if (predictedSynTargs != null) {
      Enumeration e = synTargs.elements();
      while (e.hasMoreElements()) {
        GenericVarDecl target = (GenericVarDecl) (e.nextElement());
        Assert.notFalse(predictedSynTargs.contains(target));
      }
    }

    TrAnExpr.translate = gctranslator;
    Spec spec = GetSpec.getSpecForBody(r, scope, synTargs, initState
        .getPreMap());
    GetSpec.addAxioms(Translate.axsToAdd, spec.preAssumptions);
    gctranslator.addMoreLocations(spec.postconditionLocations);

    // if the current RoutineDecl corresponds to one of our
    // constructor-inlined methods, then zero out its postconditions
    if (r instanceof MethodDecl
        && InlineConstructor.isConstructorInlinedMethod((MethodDecl) r))
      spec.post = ConditionVec.make();

    GuardedCmd fullCmd = GetSpec.surroundBodyBySpec(body, spec, scope,
                                                    fullSynTargs, initState
                                                        .getPreMap(), r
                                                        .getEndLoc(), nobody);

    // loop invariant guessing, based on assertions inside a loop
    /*
    if (Main.options().loopTranslation == Options.LOOP_SAFE) {
       LoopInvariantGuessing.traverse(fullCmd, gctranslator);
    }
    */

    if (Main.options().loopTranslation == Options.LOOP_SAFE
        && Main.options().predAbstract) {
      long T = java.lang.System.currentTimeMillis();
      Traverse.compute(fullCmd, initState, gctranslator);
      if (options().statsTime) {
        System.out.println("      [predicate abstraction time: " + timeUsed(T)
                           + "]");
      }
    }
    Translate.addTraceLabelSequenceNumbers(fullCmd);

    return fullCmd;

  }

  // Misc. Utility routines

  protected void virtualMachineVersionCheck() {
    super.virtualMachineVersionCheck();
    // Check to see that we are using a legitimate Java VM version.
    // ESC/Java2 does not fully support Java 1.5 at this time.
    // In particular, there are bugs wrt handling nested (anonymous) 
    // classes from 1.5 bytecode.  See bug#574 for more information.

    final String recommendedVersion = "1.4";
    int checkVersion = compareVersion(recommendedVersion);
    if (checkVersion < 0)
      System.out.println("Your VM is older than " + recommendedVersion);
    else if (checkVersion > 0) {

      final String java_VM_version = System
          .getProperty("java.specification.version");
      // Implementation of bug#575.
      // Issue a warning while in verbose mode about this problem with Java 1.5.
      final String byteCodeVersion = "1.5";
      int checkByteCodeVersion = compareVersion(byteCodeVersion);
      if (checkByteCodeVersion == 0 && javafe.util.Info.on)
        ErrorSet
            .caution("Java " + byteCodeVersion + " source parsing is not supported at all.\n"
                     + "Java " + byteCodeVersion + " bytecode and VMs are partially supported at this time.\n"
                     + "Please use at least a Java " + recommendedVersion + " VM and only process Java " + recommendedVersion + " (or earlier) source code\n"
                     + "and Java " + byteCodeVersion + " (or earlier) bytecode.");

      // Issue a warning if we are running in a Java 1.6 or later VM.
      if (checkByteCodeVersion > 0 && javafe.util.Info.on) {
        final String futureVersion = "1.6";
        ErrorSet
            .caution("Java " + futureVersion + " or later VMs are not fully supported at this time.\n"
            		+ " Also, the JML JDK specifications bundled with ESC/Java2 are for Java 1.4 which means that"
            		+ " false errors may be reported, due to a mismatch between the JDK and the JML specifications.");
      }
    }
  }

  private static String removeSpaces(/*@ non_null */String s) {
    while (true) {
      int k = s.indexOf(' ');
      if (k == -1) {
        return s;
      }
      s = s.substring(0, k) + s.substring(k + 1);
    }
  }

  // JVM version comparison

  private static int compareIntLex(String[] a, String[] b) {
    for (int i = 0; i < Math.min(a.length, b.length); ++i) {
      int an = Integer.parseInt(a[i]), bn = Integer.parseInt(b[i]);
      if (an < bn)
        return -1;
      if (an > bn)
        return +1;
    }
    if (a.length != b.length)
      return a.length - b.length;
    return 0;
  }

  private static int compareVersion(String other) {
    String tv = System.getProperty("java.specification.version");
    return compareIntLex(tv.split("[^0-9]"), other.split("[^0-9]"));
  }

}
