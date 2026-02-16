// $Id$
/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import java.util.Hashtable;

import javafe.ast.*;
import javafe.util.*;

/**

A <code>TypeSig</code> is a proxy and adaptor for <a
href="javafe.ast.TypeDecl.html"><code>TypeDecl</code></a>.<p>

They are proxies because, during name resolution, <a
href="javafe.ast.TypeName.html"><code>TypeName</code></a> objects are
resolved to point to <code>TypeSig</code> objects rather than directly
to <code>TypeDecl</code> objects. This gives us the flexibility of
deferring parsing of a type declaration until we know we need detailed
information about it.<p>

They are adaptors because they provide extra functionality not provided
by <code>TypeDecl</code> objects.  In particular, they provide lookup
methods that allow clients to find particular members of a type
declaration (including inherited members), and resolve methods that
perform name resolution inside the implementation of a type
declaration.<p>

<i>What about the fact that we are a subtype of <code>Type</code>?</i>


<h3> Interning </h3>

<code>TypeSig</code> objects are meant to be "interned"; that is, for
each <i>canonical</i> <code>TypeDecl</code>, we want there to be exactly
one associated <code>TypeSig</code> object.  The first
<code>TypeDecl</code> loaded by the system for a given typename is
considered to be the canonical <code>TypeDecl</code> for that typename;
all other <code>TypeDecl</code>s for that typename are considered
non-canonical.<p>

To achieve this interning, clients of the <code>javafe.tc</code> package
should not directly create <code>TypeSig</code> objects; rather, they
should rely on <code>javafe.tc</code> to create <code>TypeSig</code>
objects for them.  Similarly, clients of the <code>javafe.tc</code>
package should rely on <code>javafe.tc</code> to do the parsing
necessary to create instances of <code>TypeDecl</code>.  Not only will
the <code>javafe.tc</code> package intern <code>TypeSig</code> objects,
it will also ensure that, for <code>TypeSig s</code> and <code>TypeDecl
d,</code><ul>

<li> <code>s.getTypeDecl()</code> is always a canonical <code>TypeDecl</code>

<li> <code>s.getTypeDecl().sig == s</code>

<li> if <code>d</code> is canonical then <code>d.sig.getTypeDecl()==d</code>

<li> if <code>d</code> is not canonical then <code>d.sig</code> is the
<code>TypeSig</code> associated with <code>d</code>'s typename.</ul>

<i>What is the interface for clients within the <code>javafe.tc</code>
package?</i><p>



<h3> Staging the processing of type declarations </h3><p>

Resolving the names in a type declaration and checking its static
semantics usually involves looking at other declarations to which it
refers.  Finding, reading, and processing referred-to types makes
resolution and checking fairly complicated.  As a result, we have
decomposed it into smaller steps.  Type declarations move through a
number of states as the resolution and checking process proceeds.  In
addition to making the overall processing of type declarations
conceptually more manageable, this decomposition has two other 
benefits:<ul>


<li> <i>Handling cycles</i>. As mentioned above, processing one type may
involve processing types to which it refers.  However, two types may
refer to each other, making it impossible to process any one of them
"first."  Decomposing the processing into stages helps us handle such
cycles.

<li> <i>Improving performance</i>. Processing one type declaration does
not require fully processing the declarations to which it refers.  How
much processing is required of a referred-to type depends on the manner
in which it is referred (e.g., the superclass of a class requires more
processing than a type referred to by another type referred to by the
class).  Decomposing processing into stages allows us to be lazy in
processing referred-to types; that is, it allows us to process them only
to the extent that is necessary and no further.</ul>


The rest of this section details the states through which a
<code>TypeSig</code> object moves during its lifetime.  Most clients of
the <code>javafe.tc</code> package do not need to know about these
details.  However, clients interested in extending the processing done
by the <code>javafe.tc</code> package may need to add other states as
well.  Also, this documentation gives an overview of the
<code>javafe.tc</code> package for those interested in understanding its
implementation.<p>

<code>TypeSig</code>s start in the <i><code>TypeSig</code> created</i>
state then move sequentially through the following four states:
<i>parsed</i>, <i>supertype links resolved</i>, <i>prepped</i>, and
<i>checked</i>.  Each following state represents additional processing
completed on top of the processing required for the previous state.  A
description of these states in reverse order follows.  Included are the
properties that hold in each state; these properties are "additive":
properties of states later in the list are also properties of states
earlier in the list.<ul>

<li> <i>Checked.</i> Type declarations in this state have been fully
disambiguated, resolved and checked (see <a
href="javafe.tc.TypeCheck.html"> <code>TypeCheck</code> </a> for the
definitions of disambiguation, resolution, and checking).
Transitioning to this state can trigger parsing and/or preparation of
types referenced (<i>where?</i>) either explicitly or implicitly.
Implicitly referenced types are types used but not explicitly
mentioned in a <code>TypeName</code> node (<i>huh?</i>); for example,
the type of the result of <code>foo</code> in the expression
"<code>bar(x.foo())</code>" is implicitly referenced by this
expression.<p>


<li> <i>Prepped.</i> When checking a type declaration, we need to look
up members of other type declarations it uses.  Before looking up
members of these referred-to declarations, it is useful to do some work
on them first, such as building internal tables.  <code>TypeSig</code>s
in the prepped state have had such work done on their declarations,
preparing them for having their members queried.  As a convenience, the
methods of <code>TypeSig</code> that lookup the members of a type
declaration, such as <A href="#lookupField(javafe.ast.Identifier,
javafe.tc.TypeSig)"> <code>lookupField</code></A> and <A
href="#lookupMethod(javafe.ast.Identifier, javafe.ast.Type[],
javafe.tc.TypeSig)"> <code>lookupMethod</code></a>, automatically
transition the declaration (<i><code>TypeSig</code></i>?) to the prepped
state if it isn't there already.<p>

When preparing the declaration associated with a <code>TypeSig</code>,
it is also useful to report certain errors in it, such as two fields
with the same name.  This way, when checking a class, the user will also
see a report of errors in the interfaces (but not the implementations)
of the classes used by the class being checked.<p>

The prepped state is <i>supertype transitive</i>: a state <i>X</i> is
supertype transitive if a <code>TypeSig</code> can only be in state
<i>X</i> if all its supertypes are at least in state <i>X</i>.  Note
that there is no requirement that <i>X</i>'s type members be in any
particular state.<p>


<li> <i>Supertype links resolved.</i> When preparing or checking a type
declaration, we may need to ask if one type referred to by the
declaration is a supertype of another.  To perform this query, we need
to resolve supertype links.  After parsing, the supertypes of a type
declaration are represented as <code>TypeName</code>s, which are
symbolic references.  Resolution of a <code>TypeName</code> object
involves pointing its <code>sig</code> field to the <code>TypeSig</code>
object for the type declaration to which the <code>TypeName</code>
symbolically refers.  Resolving the supertype links of type declarations
separately from preparing them allows us to perform supertype checks
during the preparation process.<p>

As a convenience, the <code>isSubtypeOf</code> method of
<code>TypeSig</code>, which performs the supertype test, automatically
resolves supertype links if they have not been resolved already.<p>

Like the prepped state, the "supertype links resolved" state is
supertype transitive.<p>


<li> <i>Parsed</i>.  Although many <code>TypeSig</code>s come into
existence when their type declarations are parsed, some of them can be
created earlier.  Thus, we can distinguish <code>TypeSig</code>s that
have been parsed from those that have not.  However, it is difficult for
a client outside <code>javafe.tc</code> to know whether or not a
<code>TypeSig</code> has been parsed.  This is because, if a client
calls <a href="#getTypeDecl()"> <code>getTypeDecl</code></a> on a
<code>TypeSig</code> that has not been parsed, it will be parsed
automatically.<p>


<li> <i><code>TypeSig</code> created</i>.  This is the initial state
of <code>TypeSig</code> objects that come into existence before their
corresponding declarations have been parsed.  Again, it is it is
difficult for a client outside <code>javafe.tc</code> to to distinguish
between this state and the parsed state.</ul>




<h3> More internal details </h3>

<p> This section contains a few more details about the implementation
for those who wish to understand it.

<p> Inside the <code>javafe.tc</code> package, creation of
<code>TypeSig</code> and <code>TypeDecl</code> objects is managed by
the <a href="javafe.tc.OutsideEnv.html"> <code>OutsideEnv</code>
</a> class.  Methods of this class take the fully-qualified name of
an "package-member type" -- that is, a type declaration not nested
inside another declaration -- and return the <code>TypeSig</code>
instance associated with the type declaration denoted by that name.
<code>OutsideEnv</code> also coordinates the parsing of type
declarations, ensuring that only one <code>TypeDecl</code> is
created for a type declaration, and ensuring that
<code>TypeSig</code> objects point to the appropriate
<code>TypeDecl</code> objects.

<p> The <code>OutsideEnv</code> uses direct instances of
<code>TypeSig</CODE> for nested types only.  It uses a special subclass
of <code>TypeSig</code>, <code>TopTypeSig</code>, for externally
nameable types (what the Java spec calls "top-level types").  This
subclass has fields giving the types external name.  A further subclass
of <code>TopTypeSig</code>, <code>OutsideTypeSig</code>, handles outside
types.  Alone among <code>TypeSigs</code>, <code>OutsideTypeSig</code>s
have the the ability to load their <code>TypeDecl</code> objects lazily.
All other <code>TypeSig</code> classes must have their
<code>TypeDecl</code> provided at creation time.  However, this is an
implementation issue only: clients outside of <code>javafe.tc</code>
should not assume the existence of these subclasses.

<p> Transitioning a <code>TypeSig</code> from one state to another may
require parsing and processing of other types such as supertypes or the
types found in method signatures.  For example, checking that the
<code>throws</code> clause of a method override is legal may involve
checking a subtype relationship, which requires that types involved be
supertype-link resolved.  We call transitions that arise in the process
of another transitions "secondary transitions."  The code that
implements transitions does not have to explicitly invoke secondary
transitions; rather, they are performed by calling methods like
<code>lookupField</code> or <code>isSubtypeOf</code>, whose
implementations <em>do</em> invoke transitions if they are needed.  For
example, the method that implements the preparation transition does not
directly call the method that implements supertype link resolution, but
it <em>does</em> call <code>isSubtypeOf</code>, which in turn does
directly call the supertype link resolution method.  Thus, secondary
transitions are largely transparent to the code implementing each
transition, with one exception.  The code implementing the transition to
one state can<em>not</em> invoke methods that automatically transition
types to a higher state.  For example, the code implementing supertype
link resolution cannot call <code>lookupField</code>.  This discipline
helps us avoid looping.


@see OutsideEnv
@see TypeCheck
@see TypeDecl
@see TypeName

*/
public class TypeSig extends Type
{
    /***************************************************
     *                                                 *
     * Basic TypeSig instance variables:	       *
     *                                                 *
     **************************************************/

    /**
     * The name of the package we belong to.  Always non-null.
     */
    //@ invariant \nonnullelements(packageName);
    public /*@ non_null */ String[] packageName;

    /**
     * Our enclosing type or null iff we are a package-member type.
     */
    public /*@ nullable */ TypeSig enclosingType;

    /**
     * Our simple name or null if we are an anonymous type.
     */
    public /*@ nullable */ String simpleName;

    /**
     * Are we a direct member of a package or a type? <p>
     *
     * (Otherwise, we are a block-level type or an anonymous type.) <p>
     */
    public boolean member;


    /**
     * Our enclosing Env; may be null for member types because of
     * laziness.  Use TypeSig.getEnclosingEnv() to obtain an always
     * non-null version of this.
     */
    protected /*subclass-only*/ Env enclosingEnv;

    /**
     * Our TypeDecl; may be null only for package-member type
     * because of laziness.  Use getTypeDecl() to obtain an always
     * non-null version of this. <p>
     *
     * This variable should be set only using setDecl. <p>
     */
    //@ invariant state >= PARSED ==> myTypeDecl != null;
    //@ invariant state < PARSED ==> myTypeDecl == null;
    /*@spec_public*/ protected TypeDecl myTypeDecl;

    /**
     * The CompilationUnit we belong to; null iff myTypeDecl is null
     * because of laziness.  Use getCompilationUnit() to obtain an
     * always non-null version of this.
     */
    //@ invariant (myTypeDecl==null) == (CU==null);
    protected CompilationUnit CU;



    /*
     * Enforce the various invariants about what kind of type we represent
     * and the nullness of our basic instance variables.
     *
     *
     * You can tell what kind of type we are as follows:
     *
     *   package-member type              if enclosingType==null 
     *   type-member type                 if enclosingType != null && member
     *   block level type                 if simpleName != null && !member
     *   anonymous type                   if simpleName==null
     */
    //@ invariant (enclosingType==null) ==> member;   // package-member
    //@ invariant (simpleName==null) ==> !member;      // anonymous
    //@ invariant !member ==> enclosingEnv != null;
    //@ invariant (enclosingType != null) ==> myTypeDecl != null;


    /***************************************************
     *                                                 *
     * Associating TypeSigs and TypeDecls:             *
     *                                                 *
     **************************************************/

    /**
     * Decorates <code>TypeDecl</code> nodes to point to
     * <code>TypeSig</code> objects.
     */
    //@ invariant sigDecoration.decorationType == \type(TypeSig);
    public static final ASTDecoration sigDecoration
	= new ASTDecoration("sigDecoration");


    /**
     * The myTypeDecl field maps TypeSigs to TypeDecls.  The
     * getSig(TypeDecl) function provides the reverse mapping. <p>
     *
     * Precondition: d has already been associated with a TypeSig. <p>
     */
    public static /*@non_null*/TypeSig getSig(/*@non_null*/TypeDecl d) {
	TypeSig r = (TypeSig)sigDecoration.get(d);
	if (r == null) Assert.notNull(r,      //@ nowarn Pre;
	       "getSig called on a TypeDecl (" + d.id + ") not associated with a TypeSig");
	return r;
    }


    /**
     * Protected routine used to associate a TypeSig with a TypeDecl. <p>
     *
     * It automatically creates TypeSigs for any (indirect) members
     * of decl and associates them properly. <p>
     *
     * CU must be the CompilationUnit that decl belongs to.<p>
     */
    //@ ensures this.myTypeDecl != null;
    protected void setDecl(/*@ non_null @*/ TypeDecl decl,
			 /*@ non_null @*/ CompilationUnit CU) {
	this.myTypeDecl = decl;
	this.CU = CU;
	state = PARSED;
	sigDecoration.set(decl, this);

	// Create TypeSigs for any TypeDecl members of us:
	for (int i = 0; i < decl.elems.size(); i++) {
	    TypeDeclElem member = decl.elems.elementAt(i);
	    if (!(member instanceof TypeDecl))
		continue;

	    TypeDecl subDecl = (TypeDecl)member;
	    TypeSig subSig = Types.makeTypeSig(
					packageName, subDecl.id.toString(),
					 this, subDecl, CU);
	    // (The TypeSig constructor will call subSig.setDecl(subDecl, CU))
	}
    }

    /***************************************************
     *                                                 *
     * Creation:                       		       *
     *                                                 *
     **************************************************/

    /**
     * Create a TypeSig that represents a non-member type. <p>
     *
     * We represent a block-level type if simpleName is non-null,
     * and an anonymous type otherwise. <p>
     */
    //@ requires !(enclosingEnv instanceof EnvForCU);
    protected TypeSig(/*@ nullable */ String simpleName,
		      /*@ non_null @*/ Env enclosingEnv,
		      /*@ non_null @*/ TypeDecl decl) {
	super();   //@ nowarn Pre; // can't do set before super()

	member = false;

	this.simpleName = simpleName;
	this.enclosingEnv = enclosingEnv;

	this.enclosingType = enclosingEnv.getEnclosingClass();
	//@ assume this.enclosingType != this;
	Assert.notNull(this.enclosingType);

	// We inherit our packageName and CompilationUnit from our
	// enclosing type:
	this.packageName = this.enclosingType.packageName;
	this.CU = this.enclosingType.getCompilationUnit();

	setDecl(decl, this.CU);    //@ nowarn Invariant; // helper function
    }


    /**
     * Create a TypeSig that represents a member type. <p>
     *
     * This constructor is designed to be private to TypeSig;  
     * It is protected to support the Types inst pattern, but 
     * clients should use
     * OutsideEnv.lookup[Deffered] and getSig to get TypeSig's
     * representing member types. <p>
     *
     * We represent a package-member type if enclosingType is
     * non-null, and a type-member type otherwise. <p>
     *
     * Note: packageName should never be modified by the caller after
     * this call.<p>
     *
     * CU must be the CompilationUnit that decl belongs to.<p>
     */
    //@ requires \nonnullelements(packageName);
    //@ requires (enclosingType != null) ==> (decl != null);
    //@ requires (decl==null) == (CU==null);
    protected TypeSig(/*@non_null*/ String[] packageName,
		    /*@ non_null @*/ String simpleName,
		    /*@ nullable */ TypeSig enclosingType,
		    /*@ nullable */ TypeDecl decl, /*@ nullable */ CompilationUnit CU) {
	super();		//@ nowarn Pre; // can't do set before super

	member = true;

	this.packageName = packageName;
	this.simpleName = simpleName;
	this.enclosingType = enclosingType;

	this.enclosingEnv = null;      // be lazy...
	if (decl != null)
	    setDecl(decl, CU);    //@ nowarn Invariant; // helper function
    }


    /**
     * Create a TypeSig that represents an internal-use-only
     * package-member type.  Such types do not have external names
     * and should be avoided if at all possible since they are
     * kludges. <p>
     *
     * This constructor should only be used by
     * PrepTypeDeclaration.getRootInterface().<p>
     *
     * Note: packageName should never be modified by the caller after
     * this call.<p>
     *
     * CU must be the CompilationUnit that decl belongs to.<p>
     */
    //@ requires \nonnullelements(packageName);
    protected TypeSig(/*@non_null*/ String[] packageName,
		    /*@ non_null @*/ String simpleName,
		    /*@ non_null @*/ TypeDecl decl,
		    /*@ non_null @*/ CompilationUnit CU) {
	this(packageName, simpleName, null, decl, CU);
    }



//    //@ requires \nonnullelements(packageName);
//    //@ requires (enclosingType != null) ==> (decl != null);
//    //@ requires (decl==null) == (CU==null);
//    //@ ensures \result != null;
// UNUSED
//    private static TypeSig make(String[] packageName,
//				/*@ non_null @*/ String simpleName,
//				TypeSig enclosingType,
//				TypeDecl decl, 
//				CompilationUnit CU) {
//	TypeSig t = Types.makeTypeSig(packageName,
//				    simpleName,
//				    enclosingType,
//				    decl, 
//				    CU);
//	return t;
//    }
    
    /***************************************************
     *                                                 *
     * Interning by External Name for package-members: *
     *                                                 *
     **************************************************/

    /*
     * We maintain a mapping from fully-qualified names to
     * package-member types so that OutsideEnv never creates 2
     * TypeSigs with the same external name.
     *
     * This map is complete in the sense that all created
     * package-member types are in its range.
     *
     * These functions are intended for use *only* by OutsideEnv!
     */


    /**
     * The map from fully-qualified external names of package-member
     * types to the TypeSigs representing them. <p>
     * 
     * Invariant: map's range includes all existing package-member
     * TypeSigs; map sends getKey(P,T) to the unique package-member
     * TypeSig with external name P.T.<p>
     *
     * The domain type of map is String and its range type is (non-null)
     * TypeSigs.<p>
     */
    //  Old specs from original full JML spec files.  Must be
    //  rewritten for current java.util.Hashtable specs.
    //- invariant map.keyType == \type(String);
    //- invariant map.elementType == \type(TypeSig);
    //@ spec_public
    private static final /*@ non_null */ Hashtable map = new Hashtable(101);

    public static final void clear() {
	map.clear();
    }

    /**
     * Compute the key for map for fully-qualified type P.T.
     */
    //@ requires \nonnullelements(P);
    //@ ensures \result != null;
    private static String getKey(/*@non_null*/ String[] P, /*@ non_null @*/ String T) {
	String key = "";

	for (int i=0; i<P.length; i++)
	    key += "."+P[i];
	key += "."+T;

	return key;
    }


    /**
     * If a TypeSig representing the package-member type named P.T
     * has been created, return it; otherwise, return null. <p>
     *
     * This function should only be called by OutsideEnv. <p>
     */
    //@ requires \nonnullelements(P);
    static public /*@ nullable */ TypeSig lookup(/*@non_null*/ String[] P, /*@ non_null @*/ String T) {
	return (TypeSig)map.get(getKey(P,T));
    }

	// FIXME - the type lookup ends up parsing a qualified name into
	// pieces and then reassembling them through getKey into a 
	// qualified name.  This could be improved by providing a 
	// direct lookup from a fully qualified name

    /**
     * If a TypeSig representing the package-member type named P.T
     * has been created, return it; otherwise, create such a TypeSig
     * then return it. <p>
     *
     * This function should only be called by OutsideEnv. <p>
     */
    //@ requires \nonnullelements(P);
    //@ ensures \result != null;
    static /*package*/ /*@ non_null */ TypeSig get(/*@non_null*/ String[] P, /*@ non_null @*/ String T) {
	String key = getKey(P,T);
	TypeSig result = (TypeSig)map.get(key);
	if (result != null)
	    return result;
       
	result = Types.makeTypeSig(P, T, (TypeSig)null, (TypeDecl)null,
				  (CompilationUnit)null);
	map.put(key, result);
	return result;
    }

    /***************************************************
     *                                                 *
     * Lazy loading of TypeDecls:      		       *
     *                                                 *
     **************************************************/

    /**
     * Get the non-null TypeDecl we are associated with. <p>
     *
     * (Under the covers, for package-member types, this method may
     * cause <code>OutsideEnv</code> to parse the
     * <code>TypeDecl</code> in question, but this should be
     * transparent to most clients.) <p>
     */
    //@ ensures \result != null;
    //@ ensures state>=PARSED;
    public /*@ non_null */ TypeDecl getTypeDecl() {
	if (myTypeDecl==null)
	     preload();

	return myTypeDecl;
    }

    /**
     * Get the non-null CompilationUnit we are associated with. <p>
     *
     * (Under the covers, for package-member types, this method may
     * cause <code>OutsideEnv</code> to parse the type in question,
     * but this should be transparent to most clients.) <p>
     */
    //@ ensures \result != null;
    public /*@ non_null */ CompilationUnit getCompilationUnit() {
	if (myTypeDecl==null)
	     preload();

	return CU;
    }


    /**
     * Ensure that we have loaded our TypeDecl, invoking OutsideEnv
     * if needed to load a TypeDecl into us via load below.  We abort
     * via an assertion failure if OutsideEnv fails to load us.
     */
    //@ ensures myTypeDecl != null;
    private void preload() {
	if (myTypeDecl != null)
	    return;

	/*
	 * This call should always call our load method (below) with a
	 * TypeDecl.
         */
	OutsideEnv.load(this);
	Assert.notNull(myTypeDecl);
    }


    /*
     * The below functions are provided for use by OutsideEnv in
     * loading us.
     */


    /**
     * Is our TypeDecl already loaded?
     */
    //@ ensures !member ==> \result;
    //@ ensures \result == (myTypeDecl != null);
    public boolean isPreloaded() {
	return myTypeDecl != null;
    }


    /**
     * Load the non-null TypeDecl decl as our TypeDecl.
     * This method is used by OutsideEnv to install a TypeDecl in us once
     * it has been loaded.<p>
     *
     * If we have already been loaded with a different TypeDecl, then
     * an appropriate fatal error results. <p>
     *
     * Note: This method should *only* be called by OutsideEnv.load.<p>
     *
     * CU must be the CompilationUnit that decl belongs to.<p>
     */
    //@ ensures myTypeDecl != null;
    /*package*/ void load(/*@ non_null @*/ TypeDecl decl,
			  /*@ non_null @*/ CompilationUnit CU) {
	// If no (potential) duplication, just install decl and return:
	if (myTypeDecl==null) {
	    setDecl(decl, CU);
	    return;
	}

	if (myTypeDecl==decl && this.CU==CU)
	    return;

	ErrorSet.fatal(decl.loc, "type " + this.getExternalName()
		+ " already defined at "
		+ Location.toString(myTypeDecl.loc));
    }

    /***************************************************
     *                                                 *
     * Name accessor functions:			       *
     *                                                 *
     **************************************************/

    /**
     * Return our package name as a human-readable string
     * suitable for display.  If we are in the unnamed package,
     * the constant THE_UNNAMED_PACKAGE is returned.
     */
    //@ ensures \result != null;
    public /*@ non_null */ String getPackageName() {
      if (packageName.length == 0)
	return THE_UNNAMED_PACKAGE;

      StringBuffer P = new StringBuffer(5*packageName.length);
      for (int i = 0; i < packageName.length; i++) {
	if (i != 0)
	  P.append('.');
	P.append(packageName[i]);
      }

      return P.toString();
    }

    public final static /*@ non_null */ String THE_UNNAMED_PACKAGE = "<the unnamed package>";


    /**
     * Return our exact type name, omitting the package name, as a
     * human-readable string suitable for display.
     */
    //@ ensures \result != null;
    public /*@ non_null */ String getTypeName() {
	if (enclosingType==null) {
	    // package-member type:
	    return simpleName;
	}

	String parent = enclosingType.getTypeName();
	if (member)
	    return parent+"$"+simpleName;
	if (simpleName != null)
	    return parent+"$"
		+Location.toLineNumber(getTypeDecl().getStartLoc())+"$"
		+simpleName;
	else
	    return parent+"$"
		+Location.toLineNumber(getTypeDecl().getStartLoc())+"$"
		+"<anonymous>";
    }	    


    /**
     * Return our exact fully-qualified external name as a
     * human-readable string suitable for display.  The package name is
     * omitted if it is the unnamed package.
     */
    //@ ensures \result != null;
     public /*@ non_null */ String getExternalName() {
         String P = getPackageName();
         if (P==THE_UNNAMED_PACKAGE)
             return getTypeName();
         else
             return P+"."+getTypeName();
     }


    /**
     * Returns a String that represents the value of this Object.
     */
    public /*@non_null*/ String toString() {
	return getExternalName();
    }

    /***************************************************
     *                                                 *
     * ASTNode functions:			       *
     *                                                 *
     **************************************************/

    public int childCount() { 
	return 0; 
    }

    public /*@ nullable */ Object childAt(int i) {
	throw new IndexOutOfBoundsException();
    }	//@ nowarn Exception;

    public int getTag() {
	return TagConstants.TYPESIG;
    }

    public void accept(/*@ non_null */ Visitor v) {
	// v does not have a special visitor method for TypeSig
	// so just use the one for Type
	v.visitType( this );
    }

    
    public /*@ non_null */ Object accept(/*@ non_null */ VisitorArgResult v, /*@ nullable */ Object o) {
	return v.visitType( this,o );
    }
    
/*
    // We don't promise any meaningful locations...
    //@ invariant !syntax;
    {
	// Deal with can't handle non-injective fields & invariants problem:
	//@ assume (\forall MethodDecl m; m.returnType != this);
	//@ assume (\forall GenericVarDecl g; g.type != this);
	//@ set syntax = false;
    }
*/

    //@ public represents startLoc <- getTypeDecl().getStartLoc();
    public int getStartLoc() {
	return getTypeDecl().getStartLoc();
    }

    //@ also public normal_behavior
    //@ ensures \result == getTypeDecl().getEndLoc();
    public int getEndLoc() {
	return getTypeDecl().getEndLoc();
    }

    /***************************************************
     *                                                 *
     * Looking up type members:			       *
     *                                                 *
     **************************************************/

    /**
     * Do we have a type member named id (including inherited type
     * members)? <p>
     *
     * If we have exactly one such type member, then return it.  If
     * we have no such type member, return null.  If we have more
     * than one such type member, then if loc != Location.NULL then a
     * fatal error is reported at that location via ErrorSet else one
     * of the relevant type members is returned.<p>
     *
     * If caller is null, this routine does not check that the
     * resulting type (if any) is accessible.  If caller is not
     * null, then the resulting type is checked to be accessible
     * from the caller. <p>
     */
    public /*@ non_null */ TypeSig lookupType(/*@ non_null */ TypeSig caller,
				/*@ non_null @*/ Identifier id, int loc) {
	// Look locally first:
	TypeSig result = lookupLocalType(caller,id);
	if (result != null)
	    return result;

	/*
	 * Then try our superclass (if any):
	 */
	resolveSupertypeLinks();
	TypeSig s = superClass();
	if (s != null) {
	    result = s.lookupType(caller,id,loc);
	}

	/*
	 * and our superinterfaces, checking for duplicates:
	 */
	TypeDecl decl = getTypeDecl();
	for (int i=0; i<decl.superInterfaces.size(); i++ ) {
	    TypeName superInterfaceName = decl.superInterfaces.elementAt(i);
	    TypeSig newResult = getSig(superInterfaceName).lookupType(caller,id, loc);
	    if (newResult==null)
		continue;

	    if (result==null || result==newResult)
		result = newResult;
	    else {
		if (loc != Location.NULL)
		    ErrorSet.fatal(loc, "Reference to type member `"
				   + id.toString()
				   + "' of type " + toString()
				   + " is ambiguous; it may refer to type "
				   + result.toString()
				   + " or to type "
				   + newResult.toString()
				   + ".");
		else
		    return result;
	    }
	}

	return result;
    }


    /**
     * Do we have a type member named id (*not* including inherited
     * type members)?  If so, return it; otherwise, return null. <p>
     *
     * If caller is null, then this routine does not check that the 
     * resulting type (if any) is actually accessable. If caller is
     * non_null, then the resulting type is checked that it is 
     * accessible from the caller.<p>
     */

     public /*@ nullable */ TypeSig lookupLocalType(/*@ nullable */ TypeSig caller, /*@ non_null */ Identifier id) {
	TypeDeclElemVec elems = getTypeDecl().elems;
	for (int i=0; i<elems.size(); i++) {
	    TypeDeclElem e = elems.elementAt(i);
	    if (e instanceof TypeDecl && ((TypeDecl)e).id==id) {
		TypeDecl td = (TypeDecl)e;
		TypeSig t = getSig(td);
		if (caller == null) return t;
		TypeSig c = caller;
		do {
		    if (TypeCheck.inst.canAccess(c, this, td.modifiers,
						    td.pmodifiers) ) {
			return t;
		    }
		    c = c.enclosingType;
		} while (c != null);
	    }
	}
	return null;
    }

    /***************************************************
     *                                                 *
     * Environments:				       *
     *                                                 *
     **************************************************/

    /**
     * Return our enclosing environment.
     */
    //@ ensures \result != null;
    public Env getEnclosingEnv() {
	if (enclosingEnv != null)
	    return enclosingEnv;

	if (enclosingType==null)
	    enclosingEnv = new EnvForCU(getCompilationUnit());
	else
	    enclosingEnv = enclosingType.getEnv(isStatic());

	return enclosingEnv;
    }


    /**
     * Return an environment for use in checking code inside us. <p>
     *
     * Our instance members are considered accessible iff
     * staticContext is false.
     */
    //@ ensures \result != null;
    public EnvForTypeSig getEnv(boolean staticContext) {
	return new EnvForTypeSig(getEnclosingEnv(), this, staticContext);
    }

    /***************************************************
     *                                                 *
     * Misc:					       *
     *                                                 *
     **************************************************/

    /**
     * Are we (possibly implicitly) static? <p>
     *
     * This differs from using Modifiers.isStatic because it does not
     * rely on Prep'ing having added the static modifier where
     * implicit so we can be used by Prep itself.
     */
    public boolean isStatic() {
	if (Modifiers.isStatic(getTypeDecl().modifiers))
	    return true;

	// interfaces are implicitly static:
	if (getTypeDecl() instanceof InterfaceDecl)
	    return true;

	// interface members are implicitly static:
	if (enclosingType != null && 
	    enclosingType.getTypeDecl() instanceof InterfaceDecl)
	    return true;

	return false;
    }


    /**
     * Are we a top-level type? <p>
     *
     * True iff either we are a package-level type or a static member
     * of a top-level type. <p>
     *
     * Note: This may be called during any state; it may bump
     * TypeSigs to the Parsed state. <p>
     */
    //@ ensures enclosingType==null ==> \result;
    public boolean isTopLevelType() {
	if (enclosingType==null)
	    return true;

	if (!isStatic())
	    return false;

	return enclosingType.isTopLevelType();
    }

    //************************************************************
    //************************************************************
    //************************************************************
    //************************************************************
    //************************************************************

    /***************************************************
     *                                                 *
     * TypeSig states and transition functions:        *
     *                                                 *
     **************************************************/

    /*
     * The states that a TypeSig can be in.  The variable state (below)
     * holds our current state, which must be one of the following
     * values:
     */
    public final static int CREATED		= 1;
    public final static int PARSED		= 2;
    public final static int RESOLVINGLINKS	= 3;  // used by SLResolution
    public final static int LINKSRESOLVED	= 4;
    public final static int PREPPED		= 5;
    public final static int CHECKED		= 6;

    /**
     * The current state of <code>this</code>.  Must be one of
     * <code>CREATED</code>, <code>PARSED</code>,
     * <code>LINKSRESOLVED</code>, <code>PREPPED</code>, or
     * <code>CHECKED</code>.  In most circumstances, this field
     * should not be written to; rather, methods of
     * <code>TypeSig</code> should be called to effect changes to it.
     * This field must never be decreased after creation time.
     */
    public int state = CREATED;


    /*
     * Note: The transition to PARSED from CREATED is performed
     * automatically by setDecl.
     */

    /**
     * Transition <code>this</code> to the supertype links resolved
     * state. <p>
     *
     * See the TypeSig type comments for details of what this involves.<p>
     *
     * A fatal error may be reported if we cannot resolve a supertype
     * name, or detect a cycle in the type hierarchy.<p>
     */
    //@ modifies state;
    //@ ensures state >= TypeSig.LINKSRESOLVED;
    public void resolveSupertypeLinks() {
	if (state<LINKSRESOLVED)
	    SLResolution.transition(this);
    }

    /**
     * Transition <code>this</code> to the "prepped" state. <p>
     *
     * A prepped declaration has all of the <code>TypeName</code>
     * nodes defining the types of its members resolved.  For a field
     * declaration, this means resolving the <code>type</code> field;
     * For a routine declaration, this means resolving
     * <code>TypeName</code>s that occur in the <code>args</code>,
     * <code>raises</code>, and <code>returnType</code> fields.<p>
     *
     * See the TypeSig type comments for more details of what this
     * involves.<p>
     */
    //@ modifies state;
    //@ ensures state >= PREPPED;
    public void prep() {
	if (state >= TypeSig.PREPPED)
	    return;

	if (Info.on) Info.out("[prepping-slinks " + this + "]");
	resolveSupertypeLinks();
	if (Info.on) Info.out("[prepping " + this + "]");
	PrepTypeDeclaration.inst.prepTypeSignature(this);
	if (Info.on) Info.out("[prepping-complete " + this + "]");

	state = TypeSig.PREPPED;
    }

    /**
     * Transition <code>this</code> to the "checked" state. <p>
     *
     * See the TypeSig type comments for details of what this involves.<p>
     *
     * A fatal error may be reported if we cannot resolve a supertype
     * name, or detect a cycle in the type hierarchy.<p>
     */
    //@ modifies state;
    //@ ensures state >= CHECKED;
    public void typecheck() {
	if (this.state >= TypeSig.CHECKED)
	    return;
	if (this.state < TypeSig.PREPPED)
	    prep();

        typecheckSuperTypes();
	long start = 0;
	if (Info.on) start = javafe.Tool.currentTime();
	if (Info.on) Info.out("[typechecking " + this + "]");
	TypeCheck.inst.makeFlowInsensitiveChecks().checkTypeDeclaration(this);
	if (Info.on) Info.out("[typechecking-end " + this + " " + javafe.Tool.timeUsed(start) + "]");
        // TODO: @review kiniry 31 Aug - Why is this commented out?
	// FlowSensitiveChecks.checkTypeDeclaration(this);
	this.state = TypeSig.CHECKED;
    }

  /**
   * Typecheck the superclass of the current classtype being typecheck and
   * typecheck all interfaces that the current classtype implements.
   */
    public void typecheckSuperTypes() {
        TypeSig t = superClass();
        // If we are typechecking an inner class whose parent is the enclosing
        // environment, then do not try to typecheck the parent as that is what we
        // are doing at this moment!
        if ((t != null) && getEnclosingEnv().getEnclosingClass() != enclosingType)
          t.typecheck();
        TypeDecl decl = getTypeDecl();
        for (int i=0; i<decl.superInterfaces.size(); i++ ) {
            TypeName superInterfaceName = decl.superInterfaces.elementAt(i);
            TypeSig ts = getSig(superInterfaceName);
            ts.typecheck();
        }
    }

    /***************************************************
     *                                                 *
     * Looking up fields, methods, and constructors:   *
     *                                                 *
     **************************************************/

    //// Fields and methods associated with preparation of a TypeSig

    /** After preparation, this field contains all field members of
    the <code>TypeDecl</code> associated with <code>this</code>,
    including inherited ones. */

    // "invariant" fields.<each element>.hasParent
    //@ invariant state >= PREPPED ==> fields != null;
    //@ spec_public
    protected FieldDeclVec fields;

    // Note: 'fields' contains all visible fields
    // 'hiddenfields' contains all the others (e.g. hidden or not accessible)
    //@ invariant state >= PREPPED ==> hiddenfields != null;
    protected FieldDeclVec hiddenfields;

    /** After preparation, this field contains all method members of
    the <code>TypeDecl</code> associated with <code>this</code>,
    including inherited ones. */

    // "invariant" methods.<each element>.hasParent
    //@ invariant state >= PREPPED ==> methods != null;
    //@ spec_public
    protected MethodDeclVec methods;

    /** Returns all fields of the type declaration associated with
    <code>this</code>, including inherited ones.  (If
    <code>this</code> has not been prepped yet, this method will prep
    it (possibly triggering parsing and/or processing of other
    types).) If allFields is true, then all declared fields, including 
    hidden and inaccessible fields, are returned; if allFields is false,
    then only visible fields are returned.  */

    // "ensures" <result elements>.hasParent
    //@ ensures \result != null;
    public FieldDeclVec getFields(boolean allFields) {
        prep();
        Assert.notNull( fields );
	if (!allFields) return fields;
	FieldDeclVec v = fields.copy();
	v.append(hiddenfields);
	return v;
    }

    public FieldDeclVec getFieldsRaw() { return fields; }
    
    public FieldDeclVec getHiddenFields() { return hiddenfields; }

    /** Similar to <code>getFields</code>, except for methods. */

    // "ensures" <result elements>.hasParent
    //@ ensures \result != null;
    public MethodDeclVec getMethods() {
        prep();
        Assert.notNull( methods );
        return methods;
    }
    
    
    /** TBW */
    
    //@ requires \nonnullelements(args) && caller != null;
    //@ ensures \result != null;
    public ConstructorDecl lookupConstructor(Type[] args, TypeSig caller) 
            throws LookupException
    {
        prep();
        
        // Holds the most specific, applicable, accessible constructor found so far
        ConstructorDecl mostSpecific = null;
        boolean somethingFound = false;
        TypeDecl decl = getTypeDecl();
        
        search:
        for(int i = 0; i < decl.elems.size(); i++) {
            TypeDeclElem elem = decl.elems.elementAt(i);
            if (elem instanceof ConstructorDecl) {
                ConstructorDecl md = (ConstructorDecl)elem;
                if (md.args.size() == args.length
                    && TypeCheck.inst.canAccess(caller, this, md.modifiers,
                                                md.pmodifiers) ) {
                    // accessible, same name and number of args
                    
                    somethingFound = true;
                    for(int j = 0; j < args.length; j++)
                        if (! Types.isInvocationConvertable(args[j], 
                                                            md.args.elementAt(j).type))
                            continue search;
                    // accessible and applicable
                    
                    if (mostSpecific == null ||
                        Types.routineMoreSpecific(md, mostSpecific))
                        mostSpecific = md;
                    else if (! Types.routineMoreSpecific(mostSpecific, md))
                        throw new LookupException( LookupException.AMBIGUOUS );
                }
            }
        }
        
        if (mostSpecific != null)  
            return mostSpecific;
        else if (somethingFound) 
            throw new LookupException( LookupException.BADTYPECOMBO );
        else 
            throw new LookupException( LookupException.NOTFOUND );
    }
    
    /** TBW */
    
    //@ ensures \result != null;
    //@ ensures \result.id == id;
    public /*@ non_null */ FieldDecl lookupField(/*@ non_null */ Identifier id, /*@ non_null */ TypeSig caller) 
            throws LookupException
    {
        FieldDeclVec fields = getFields(false);
        FieldDecl r = null;
        for(int i=0; i<fields.size(); i++ ) {
            FieldDecl fd = fields.elementAt(i);
            if (fd.id == id)
                if (r == null) r = fd;
                else throw new LookupException( LookupException.AMBIGUOUS );
        }
        
        if (r == null) 
            throw new LookupException( LookupException.NOTFOUND );
        else if (! TypeCheck.inst.canAccess(caller, this, r.modifiers,
                                            r.pmodifiers))
            throw new LookupException( LookupException.NOTACCESSIBLE );
        else return r;
    }
    
    
    /** TBW */
    
    public boolean hasField(/*@ non_null */ Identifier id) {
        FieldDeclVec fields = getFields(false);
        for(int i=0; i<fields.size(); i++)
            if (fields.elementAt(i).id == id) return true;
        return false;
    }
    
    public MethodDecl hasMethod(Identifier id, Type[] args) {
	try {
	    return lookupMethod(id,args,this);
	} catch (LookupException e) {
	    return null;
	}
    }

    /** TBW */
    
    //@ requires \nonnullelements(args) && caller != null;
    //@ ensures \result != null;
    //@ ensures \result.id == id;
    public MethodDecl lookupMethod(Identifier id, Type[] args, TypeSig caller) 
            throws LookupException
    {
        MethodDeclVec methods = getMethods();
        
        // Holds the most specific, applicable, accessible method found so far
        MethodDecl mostSpecific = null;
        boolean somethingFound = false;
        
        search:
        for(int i = 0; i < methods.size(); i++) {
            MethodDecl md = methods.elementAt(i);
            if (md.id == id 
                && md.args.size() == args.length
                && TypeCheck.inst.canAccess(caller, this, md.modifiers,
                                            md.pmodifiers)) {
                // accessible, same name and number of args

                somethingFound = true;
                for(int j=0; j<args.length; j++) {
		    // FIXME - the argument (particularly the second) might be just a
		    // Typename - ought to resolve it once and for all, instead of doing so
		    // each time the following method is called.
                    if(! Types.isInvocationConvertable(args[j], 
                                                       md.args.elementAt(j).type)) {
                        continue search;
		    }
		}
                // accessible and applicable
      
                if (mostSpecific == null 
                    || Types.routineMoreSpecific(md, mostSpecific))
                    mostSpecific = md;
                else if (! Types.routineMoreSpecific(mostSpecific, md))
                    throw new LookupException( LookupException.AMBIGUOUS );
            }
        }

        if (mostSpecific != null) 
            return mostSpecific;
        else if (somethingFound) 
            throw new LookupException( LookupException.BADTYPECOMBO );
        else 
            throw new LookupException( LookupException.NOTFOUND );
    }

    //************************************************************
    //************************************************************
    //************************************************************
    //************************************************************
    //************************************************************


    /**
     * Gets the TypeSig recorded by <code>setSig</code>, or null.
     */
    public static TypeSig getRawSig(/*@ non_null @*/ TypeName n) {
	TypeSig r = (TypeSig)sigDecoration.get(n);
	return r;
    }

    /**
     * Gets the TypeSig recorded by <code>setSig</code>.
     *
     * Precondition: n has been resolved.
     */
    //@ ensures \result != null;
    public static TypeSig getSig(/*@ non_null @*/ TypeName n) {
	TypeSig r = (TypeSig)sigDecoration.get(n);
	if (r==null) {
	    ErrorSet.error(n.getStartLoc(),
			   "Internal error: getSig called on a TypeName ("
			   + n + ") that has not been resolved!");
	    System.out.flush();
	    Assert.precondition( //@ nowarn Pre; // punt on catching this
				"See previous error message");
	}

	return r;
    }

    public static void setSig(/*@ non_null @*/ TypeName n,
			      /*@ non_null @*/ TypeSig sig) {
	sigDecoration.set(n, sig);
    }

    public final boolean isSubtypeOf(TypeSig s2) {
	if (state < TypeSig.LINKSRESOLVED)
	    resolveSupertypeLinks();

	TypeSig jlo = Types.javaLangObject();

	if (this == s2 || s2 == jlo)
	    return true;
	if (this == jlo)
	    return false;

	TypeDecl d = getTypeDecl();
	if (d.getTag() == TagConstants.CLASSDECL &&
	    ((ClassDecl)d).superClass != null) {
	    TypeSig s1 = getSig(((ClassDecl)d).superClass);
	    if (s1 == s2)
		return true;
	    if (s1.isSubtypeOf(s2))
		return true;
	}

	for(int i = 0; i < d.superInterfaces.size(); i++) {
	    TypeSig s1 = getSig(d.superInterfaces.elementAt(i));
	    if (s1 == s2)
		return true;
	    if (s1.isSubtypeOf(s2))
		return true;
	}

	return false;
    }

    //// Helper functions

    //@ requires s != null;
    public final boolean inSamePackageAs(TypeSig s) {
        String[] p1 = this.packageName;
        String[] p2 = s.packageName;
        if (p1.length != p2.length) return false;
        for(int i = 0; i < p1.length; i++)
            if (! p1[i].equals(p2[i])) return false;
        return true;
    }

    /** Check invariants of a TypeSig, raising an exception if
     they don't hold. */

    public void check() {
        Assert.notFalse(state != RESOLVINGLINKS);		    //@ nowarn Pre;

        if (state >= CREATED) {
            if (state == CREATED)
                Assert.notFalse(myTypeDecl == null);
        }

        if (state >= PARSED) {
            Assert.notNull(myTypeDecl);
            Assert.notFalse(this == getSig(myTypeDecl));	    //@ nowarn Pre;
        }
    }

    /** Check invariants of a TypeSig, raising an exception if
     they don't hold. */

    public void deepCheck() {
        Info.out("[checking deep invariants on " + this + "]");
        check();
        if (state >= PARSED) {
            myTypeDecl.check();
            CheckInvariants.checkTypeDeclOfSig(this);
        }
    }

    private TypeSig superClass = null;

    // Probably should use this only after super types have been resolved.
    public TypeSig superClass() {
	if (superClass != null) return superClass;
	TypeDecl decl = getTypeDecl();
	if (decl instanceof ClassDecl) {
	    TypeName n = ((ClassDecl)decl).superClass;
	    if (n != null) {
		superClass = getSig(n);
	    }
	}
	return superClass;
    }

    /* This class returns a Collection containing all of the super interfaces
	of the receiver TypeSig.  No interface is repeated.
    */
    public java.util.Collection superInterfaces() {
	TypeSig t = this;
	java.util.ArrayList result = new java.util.ArrayList();
	int j = 0;
	while (true) {
	    TypeNameVec tv = t.getTypeDecl().superInterfaces;
	    for (int i = 0; i<tv.size(); ++i) {
		TypeName tt = tv.elementAt(i);
		TypeSig tts = getSig(tt);
		if (!result.contains(tts)) result.add(tts);
	    }
	    if (j >= result.size()) break;
	    t = (TypeSig)result.get(j++);
	}
	return result;
    }

} // end of class TypeSig

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
