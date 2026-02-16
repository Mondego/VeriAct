/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;
import javafe.genericfile.*;

import javafe.reader.StandardTypeReader; // for debugging only
import javafe.reader.TypeReader;

import javafe.util.Location;
import javafe.util.Assert;
import javafe.util.ErrorSet;

import java.util.ArrayList;
import java.util.Iterator;
import java.io.File;

/**
 * <code>OutsideEnv</code> implements the top-level environment
 * consisting of only the package-member types.
 *
 * <p> This is the environment outside of any compilation unit (e.g.,
 * no import declarations are in effect).  It is used to lookup the
 * {@link TypeSig} for a given fully-qualified package-member name
 * (P.T).  Class-member types are obtained by using the lookup methods
 * of the {@link TypeSig} that contains them as members. </p>
 *
 * <h3> Initialization </h3>
 *
 * <p> In order to greatly simplify the front end, there can be at
 * most one such environment during front-end execution.  All of
 * <code>OutsideEnv</code>'s lookup methods are accordingly static
 * methods.  <code>OutsideEnv</code> must be initialized before any of
 * its lookup methods can be called. </p>
 *
 * <p> At initialization time <code>OutsideEnv</code> is passed a way
 * to determine which fully-qualified package-member-type names exist
 * and a means to read in and parse the files of those types into
 * {@link CompilationUnit}s.  This is done by passing
 * <code>OutsideEnv</code> a {@link TypeReader}, which contains
 * exactly this information. </p>
 *
 * <p> <code>OutsideEnv</code> uses this information to determine
 * which package-member types exist and to create
 * <code>TypeSig</code>s for them when needed by loading their
 * underlying {@link TypeDecl}s in from the filesystem.  (Each
 * java file contains a <code>CompilationUnit</code>, which is a set
 * of <code>TypeDecl</code>s.) </p>
 *
 * <h3> Loading <code>CompilationUnit</code>s </h3>
 *
 * <p> Loading is actually done lazily for efficiency reasons(*).
 * When a fully-qualified package-member-type name is looked up for
 * the first time, <code>OutsideEnv</code> first checks to see if it
 * exists.  If it exists, then a new unloaded <code>TypeSig</code> is
 * returned.  Otherwise, <code>null</code> is returned.  Future
 * lookups of the same name return the same result, except that for a
 * local package-member-type (see next section), a null result may
 * change to a non-null result. </p>
 *
 * <p> Only when the new <code>TypeSig</code>'s <code>TypeDecl</code>
 * is touched (via {@link TypeSig#getTypeDecl}) for the first time does
 * <code>OutsideEnv</code> load in the <code>CompilationUnit</code>
 * that should contain that type.  Errors may be reported via {@link
 * ErrorSet} at this time (e.g., I/O error, syntax error, file fails
 * to contain the type, etc.).  This loading is otherwise transparent
 * to the users of <code>TypeSig</code>.  An special version of lookup
 * is available that defers testing for type existence until loading
 * time; this is useful for dealing with types that are required to
 * exist by the Java language specification. </p>
 *
 * <p> (*) - Exception: if {@link #eagerRead} is set (not the
 * default), all loading is done non-lazily. </p>
 *
 * <p> The {@link #avoidSpec} flag is used when
 * <code>CompilationUnit</code>s are read in to determine if a spec or
 * a non-spec should be read.  (Note that non-specs are not always
 * available.) </p>
 *
 * <p> When <code>CompilationUnit</code>s are loaded in, TypeSigs are
 * automatically created for each of their <code>TypeDecl</code>s
 * (including recursively). </p>
 * 
 * <h3> Local package-member types </h3>
 *
 * <p> A package-member type named <i>T</i> that is contained in a
 * file <i>V</i><code>.java</code>, <i>T</i> != <i>V</i>, is called a
 * <em>local package-member type</em>.  Such types are accessible only
 * from within in the same file.  <code>OutsideEnv</code> handles such
 * types as follows:
 *
 * <ul>
 *   <li> Before the file containing local package-member type
 *        <i>P.T</i> is loaded in, looking up <i>P.T</i> returns
 *        <code>null</code>.  (Aka, it is considered not to exist.)
 *        </li>
 *
 *   <li> Afterwards, the lookup returns a <code>TypeSig</code> that
 *        has been preloaded with the correct <code>TypeDecl</code>
 *        from the file.  It is the caller's responsibility to check
 *        whether the returned type is accessible or not. </li>
 * </ul>
 *
 * <p> The existence of local package-member types opens up the
 * possibility of duplicate package-member-type definitions.  Should
 * <code>OutsideEnv</code> load two different package-member types
 * with the same name, a fatal error will be reported via
 * <code>ErrorSet</code>.  Because files are loaded lazily, some
 * duplicate type errors may not be detected. </p>
 *
 * <h3> Additional source files </h3>
 *
 * <p> A client of <code>OutsideEnv</code> may add additional
 * package-member types to those defined by the information provided
 * at initialization time by using the method <code>addSource</code>.
 * <code>addSource</code> is called with a source file; it attempts to
 * load the <code>CompilationUnit</code> contained in that file.  If
 * successful, it adds the package-member types contained in that file
 * to the package-member-type environment and returns the loaded
 * <code>CompilationUnit</code> to the caller. </p>
 *
 * <p> {@link #addSource(GenericFile)} is intended primarily for use in handling
 * source files given to a tool as command-line arguments.  It can be
 * called only before the first lookup is done.  The filenames of the
 * source files passed to <code>addSource</code> are ignored. </p>
 *
 * <h3> Notification </h3>
 *
 * <p> Whenever <code>OutsideEnv</code> successfully loads a
 * <code>CompilationUnit</code>, it notifies the current {@link
 * Listener}, if any.  Only one <code>Listener</code> at a time is
 * currently supported; {@link #setListener} is used to set the
 * current <code>Listener</code>. </p>
 *
 * <p> Because this notification is "asynchronous" (it can occur in
 * the middle of any code that touches a <code>TypeSig</code>'s
 * <code>TypeDecl</code>), it is strongly recommended that
 * <code>Listener</code>s take no action other then storing
 * information for later use. </p>
 *
 * <h3> Implementation </h3>
 *
 * <p> Note that the implementation of the functionality described
 * here is spread between this class and that of
 * <code>TypeSig</code>. </p>
 *
 * @see TypeSig
 * @see javafe.reader.TypeReader
 * @see javafe.ast.CompilationUnit
 * @see Listener
 */

public final class OutsideEnv {
  // Class Variables

  /**
   * The {@link TypeReader} for our underlying Java file space.
   */
  private static /*@nullable spec_public*/ TypeReader reader;
  //@ public static model boolean initialized;
  //@   private static represents initialized <- reader!=null;

  //+@ requires initialized;
  public static /*@non_null*/TypeReader reader() { //@ nowarn NonNullResult;
	return /*+@(non_null)*/reader;
  }

/**
   * When we load in types, do we prefer to read specs or non-specs?
   * Defaults to preferring non-specs.
   */
  public static boolean avoidSpec = true;

  /**
   * If true, files are read eagerly, as soon as we look them up.
   * Defaults to false.
   */
  public static boolean eagerRead = false;

  /** Count of files read so far. */
  //@ private invariant filesRead >= 0;
  //@ spec_public
  private static int filesRead = 0;

  /**
   * The {@link Listener} to notify when a {@link CompilationUnit}
   * is loaded.  May be <code>null</code> if there is no current
   * <code>Listener</code> (the initial state).
   */
  //@ spec_public
  private static /*@nullable*/ Listener listener = null;

  /** Return count of files read so far. */
  //@ ensures \result == filesRead;
  //@ ensures \result >= 0;
  public static int filesRead() {
    return filesRead;
  }

  // Initialization

  //* No constructors available:
  //@ requires false;
  private OutsideEnv() {
    Assert.fail("No instances!");
  }

  /**
   * Initialize ourselves to use <code>TypeReader</code>
   * <code>R</code> for our underlying Java file space.
   *
   * @requires <code>R</code> is not <code>null</code>, no
   * <code>init</code> method for this class has previously been
   * called.
   */
  //+@ requires !initialized;
  //@ ensures reader == R;
  //@ ensures_redundantly initialized;
  public static void init(/*@ non_null @*/TypeReader R) {
    reader = R;
  }

  //@ ensures !initialized;
  //@ ensures filesRead == 0;
  //@ ensures listener == null;
  //@ ensures !eagerRead;
  //@ ensures avoidSpec;
  public static void clear() {
	if (reader instanceof javafe.reader.StandardTypeReader)
		((javafe.reader.StandardTypeReader)reader).clear();
    reader = null;
    filesRead = 0;
    listener = null;
    eagerRead = false;
    avoidSpec = true;
    javafe.tc.Types.remakeTypes();
    TypeSig.clear();
  }

  // Looking up TypeSig's

  /**
   * Get the <code>TypeSig</code> for fully-qualified
   * package-member name <code>P.T</code>.  Returns null if no such
   * type exists.
   *
   * <p> This function never results in
   * <code>CompilationUnit</code>s being loaded unless eagerRead is
   * set. </p>
   *
   * <p> Calling this function twice with the same arguments is
   * guaranteed to give back the same answer, except that a
   * <code>null</code> answer may later change to a
   * non-<code>null</code> answer. </p>
   *
   * @requires an init method has already been called
   */
  //@ requires \nonnullelements(P);
  //+@ requires initialized;
  public static /*@nullable*/TypeSig lookup(String[] P, /*@ non_null @*/String T) {
    TypeSig result = TypeSig.lookup(P, T);
    if (result == null && reader().exists(P, T)) result = TypeSig.get(P, T);
    if (result != null && eagerRead) result.getTypeDecl();
    return result;
  }

  /**
   * Like <code>lookup</code> except that checking the existence of
   * the type is deferred until it's <code>TypeDecl</code> is touched
   * for the first time.  If eagerRead is set, existence is always
   * checked, with non-existance resulting in an error.
   *
   * <p> This routine never returns <code>null</code>: if
   * <code>P</code> does not exist in our Java file space, then an
   * unloaded <code>TypeSig</code> is returned; when its
   * <code>TypeDecl</code> is first referenced, an error will be
   * reported. </p>
   *
   * <p> This function is intended to be used only to load types
   * required to be present by the language specification (e.g.,
   * {@link java.lang.Object}). </p>
   *
   * @requires an init method has already been called.
   */
  //@ requires \nonnullelements(P);
  //@ requires T != null;
  //@ requires initialized;
  //@ ensures \result != null;
  public static TypeSig lookupDeferred(String[] P, String T) {
    TypeSig result = TypeSig.get(P, T);

    if (eagerRead) result.getTypeDecl();

    return result;
  }

  // Loading CompilationUnits

  /**
   * Attempt to add the package-member types contained in a source
   * file to the package-member-types environment, returning the
   * <code>CompilationUnit</code>, if any, found in that file.
   *
   * <p> If an error occurs, it will be reported via
   * <code>ErrorSet</code> and <code>null</code> will be returned. </p>
   *
   * <p> <code>null</code> may also be returned if a file is
   * repeated on the command line. </p>
   *
   * @requires no lookup has been done yet using this class.
   *
   * @note Calling <code>addSource</code> twice on the same file may
   * or may not produce a duplicate-type error.
   */
  //+@ requires initialized;
  public static CompilationUnit addSource(/*@non_null*/GenericFile source) {
    filesRead++;
    CompilationUnit cu = reader().read(source, avoidSpec);
    if (cu != null) {
      setSigs(cu);
      notify(cu);
    }
    return cu;
  }

  /**
   * Adds all relevant files from the given package; 'relevant' is
   * defined by the 'findFiles' method of the current reader.
   */
  //+@ requires sources.elementType <: \type(GenericFile);
  //+@ ensures \result.elementType <: \type(CompilationUnit);
  public static ArrayList addSources(ArrayList sources) {
    ArrayList out = new ArrayList(sources.size());
    Iterator i = sources.iterator();
    while (i.hasNext()) {
      GenericFile gf = (GenericFile)i.next(); //@nowarn Cast;
      out.add(addSource(gf));
    }
    return out;
  }

  //+@ requires initialized;
  //@ ensures \result.elementType <: \type(GenericFile);
  public static ArrayList resolveSources(String[] pname) {
    ArrayList a = reader().findFiles(pname);
    if (a == null) {
      ErrorSet.error("Could not locate package: "
          + javafe.parser.ParseUtil.arrayToString(pname, "."));
      return null;
    }
    if (a.isEmpty()) {
      ErrorSet.caution("Package has no files: "
          + javafe.parser.ParseUtil.arrayToString(pname, "."));
    }
    return a;
  } //@ nowarn Post;

  /**
   * Attempt to add the package-member types contained in a named
   * source file to the package-member-types environment, returning
   * the <code>CompilationUnit</code>, if any, found in that
   * file.
   *
   * <p> If an error occurs, it will be reported via
   * <code>ErrorSet</code> and <code>null</code> will be
   * returned. </p>
   *
   * @note Calling <code>addSource</code> twice on the same file may
   * or may not produce a duplicate-type error.
   *
   * @requires no lookup has been done yet using this class.
   */
  //@ requires sourceName != null;
  public static CompilationUnit addSource(String sourceName) {
    GenericFile source = new NormalGenericFile(sourceName);
    return addSource(source);
  }

  // Output is an ArrayList of GenericFiles.
  //+@ requires initialized;
  //@ ensures \result == null || \result.elementType <: \type(GenericFile);
  public static /*@nullable*/ArrayList resolveDirSources(String dirname) {
	  File f = new File(dirname);
	  if (!f.exists()) {
		  ErrorSet.caution("Directory does not exist: " + dirname);
		  return null;
	  }
	  File[] names = f.listFiles(reader().filter());
	  if (names.length == 0) {
		  ErrorSet.caution("Directory has no files: " + dirname);
	  }
	  ArrayList a = new ArrayList(names.length);
	  for (int i = 0; i < names.length; ++i) {
		  /*@non_null*/File name = names[i]; //@ nowarn NonNull;
		  a.add(new NormalGenericFile(name));
	  }
	  return a;
  }

  /**
   * This routine creates TypeSigs for each TypeDecl member of
   * <code>cu</code>.
   *
   * <p> As a side effect, this sets the sig fields of
   * <code>cu</code>'s direct TypeDecl members (aka, the TypeDecls
   * for the package-member types cu contains) to point to TypeSigs
   * that have been loaded with the TypeDecls that point to
   * them. </p>
   *
   * @requires <code>cu</code> must be non-null.
   */
  //@ requires cu != null;
  private static void setSigs(CompilationUnit cu) {
    // Get package name from cu (may be null):
    String[] P = new String[0];
    if (cu.pkgName != null) P = cu.pkgName.toStrings();

    // Iterate over all the TypeDecls representing package-member
    // types in cu:
    TypeDeclVec elems = cu.elems;
    for (int i = 0, sz = elems.size(); i < sz; i++) {
      TypeDecl decl = elems.elementAt(i);
      String T = decl.id.toString(); // decl's typename

      TypeSig sig = TypeSig.get(P, T);
      sig.load(decl, cu);
    }
  }

  /**
   * Attempt to load the TypeDecl of TypeSig sig.
   *
   * <p> This method should be called only from
   * TypeSig.preload. </p>
   *
   * <p> Tries to load the file that should contain sig.  Reports
   * any errors encountered to ErrorSet.  If successful, calls
   * sig.load with its TypeDecl. </p>
   *
   * <p> It is a fatal error if this routine cannot load sig.  Later
   * the error may be made non-fatal; in that case TypeSig.preload
   * will be responsible for substituting a wildcard TypeDecl. </p>
   *
   * @requires an init method has already been called.
   */
  //@ requires initialized;
  //@ ensures sig.myTypeDecl != null;
  /*package*/static void load(/*@ non_null @*/TypeSig sig) {
    // Do nothing if sig is already loaded:
    if (sig.isPreloaded()) return;

    filesRead++;
    // Read in the CompilationUnit that should have sig in it:
    CompilationUnit cu = reader()
        .read(sig.packageName, sig.simpleName, avoidSpec);
    if (cu == null) {
      ErrorSet.fatal("unable to load type " + sig.getExternalName());
      return;
    }

    /*
     * Get cu's package name in the same format as
     * TypeSig.getPackageName uses:
     */
    String actualPkg = (cu.pkgName == null) ? TypeSig.THE_UNNAMED_PACKAGE
        : cu.pkgName.printName();

    // Check that cu is in the correct package:
    if (sig.getPackageName().equals(actualPkg)) {
      /*
       * Only load the types in cu if it is in the correct
       * package:
       */
      setSigs(cu);
      notify(cu);
    } else {
      // Get the location of the package declaration in cu if
      // present, otherwise get a location for the entire cu:
      int pkgDeclLoc = (cu.pkgName != null) ? cu.pkgName.getStartLoc()
          : Location.createWholeFileLoc(Location.toFile(cu.loc));

      ErrorSet.error(pkgDeclLoc, "file declared to be in package " + actualPkg
          + " rather than in the correct package " + sig.getPackageName());
    }

    // Make sure the CompilationUnit actually contains sig:
    if (!sig.isPreloaded()) {
      int fileLoc = Location.createWholeFileLoc(Location.toFile(cu.loc));
      ErrorSet.fatal(fileLoc, "file does not contain the type "
          + sig.getExternalName() + " as expected");
    }
  }

  // Notification

  /**
   * Set the <code>Listener</code> to be notified about
   * <code>CompilationUnit</code> loading.
   *
   * <p> <code>l</code> may be <code>null</code> if no notification
   * is desired (the initial default).  The previous current
   * <code>Listener</code> is replaced.  (I.e., only 1
   * <code>Listener</code> may be in effect at a time.) </p>
   */
  public static void setListener(Listener l) {
    listener = l;
  }

  /**
   * Send a CompilationUnit-loaded notification event to the current
   * Listener (if any).
   *
   * @requires justLoaded != null, justLoaded must already have
   * the <code>sig</code> fields of its direct
   * <code>TypeDecl</code>s adjusted.
   */
  private static void notify(/*@non_null*/CompilationUnit justLoaded) {
	  Listener _listener = listener;
    if (_listener != null) _listener.notify(justLoaded);
  }

  // Test methods

  /**
   * A debugging harness that allows describing the results of
   * calling <code>lookup</code> on a series of package-member-type
   * names.
   */
  //@ requires \nonnullelements(args);
  public static void main(/*@non_null*/String[/*#@non_null*/] args) {
    // Check argument usage:
    if (args.length == 0) {
      System.err
          .println("OutsideEnv: <fully-qualified package-member-type name>...");
      System.exit(1);
    }

    init(StandardTypeReader.make()); // Use default classpath...

    // Test each package-member-type name:
    for (int i = 0; i < args.length; i++)
      describeLookup(args[i]);
  } //@ nowarn Post;

  /**
   * Call lookup on N then describe the results.
   *
   */
  //@ requires initialized; // that is, an init method has alreaady been called
  private static void describeLookup(/*@ non_null @*/String N) {
    // Convert N to a list of its components:
    String[/*#@non_null*/] components = javafe.filespace.StringUtil.parseList(N, '.');
    if (components.length == 0) {
      System.out.println("Error: `' is an illegal type name");
      return;
    }

    // Split components into P and T:
    String[/*#@non_null*/] P = new String[components.length - 1];
    for (int i = 0; i < P.length; i++)
      P[i] = components[i];
    String T = components[components.length - 1];

    TypeSig result = lookup(P, T);
    if (result == null) {
      System.out.println("no such type " + N);
      return;
    }

    System.out.println("Sig = " + result + "; "
        + (result.isPreloaded() ? " " : "not ") + "preloaded");

    System.out.println("  represents package-member-type "
        + result.getExternalName());

    System.out.println("  it's TypeDecl is:");
    PrettyPrint.inst.print(System.out, 0, result.getTypeDecl());
  }
}
