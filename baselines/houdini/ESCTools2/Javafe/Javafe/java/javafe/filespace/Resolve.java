/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.filespace;


import java.io.IOException;


/**
 * This module encapsulates how to resolve an ambiguous multi-part
 * identifier (i.e., X.Y.Z) into a package + (possibly a) reference
 * type + a type field/member multi-part identifier, using the output
 * of ClassPath.<p>
 *
 * I.e., it would split java.util.zip.Foo.Subclass.x.y into the package
 * java.util.zip, the (inner) type Foo$Subclass, and the field/member
 * identifier x.y.<p>
 */

public class Resolve {

    /***************************************************
     *                                                 *
     * Primitive functions for looking up identifiers: *
     *                                                 *
     **************************************************/

    /**
     * Does a package contain a reference type with a given simple
     * name?<p>
     *
     * Currently true if a source or a binary for that type exists
     * (directly) in the given package.<p>
     */
    public static boolean typeExists(/*@ non_null @*/ Tree P, 
                                     /*@ non_null @*/ String typeName) {
	if (P.getChild(typeName+".java") != null)
	    return true;
	if (P.getChild(typeName+".class") != null)
	    return true;

	return false;
    }

    // The class Resolve_Result would be here as the static class Result if
    // inner classes were being used.<p>
    //
    // See its comments.<p>


    // The class Resolve_AmbiguousName would be here as the static class
    // AmbiguousName if inner classes were being used.<p>
    //
    // See its comments.<p>


    /**
     * Lookup a multi-part identifier in a Java filespace in the same
     * way that the Java compiler does so.<p>
     *
     * Precondition: the identifier parts should not contain the '.' or
     * '$' characters, and each must be non-null and non-empty.<p>
     *
     * The leftmost part of the identifier is assumed to refer to an
     * existing package; the longest such name is used.  The returned
     * package will always be non-null because package names may be
     * empty ("" refers to the top package).<p>
     *
     * The remaining part of the identifier (may be empty) is then
     * assumed to be the concatenation of a (inner) reference-type name
     * and a remainder part.  Again, as with the package name, the type
     * name is assumed to be the longest such name that refers to an
     * existing type, with the proviso that a type is considered to
     * exist only if all prefix types (i.e., X and X$Y for X$Y$Z)
     * also exist.  If no non-empty prefix names an existing type,
     * then the type name is taken to be empty and the returned
     * typeName will be null.  Otherwise, the returned typeName
     * contains the (non-null) name of the type, with its parts
     * separated by '$'s.<p>
     *
     * The remaining part of the identifier after the package name and
     * type name have been removed is returned as the remainder part.<p>
     *
     *
     * EXCEPTION: if, while identifying the package name, a package is
     * encountered that has the same name as a reference type, then the
     * exception Resolve_AmbiguousName will be thrown with
     * ambiguousPackage set to the package with the ambiguous name.
     * Such package/type naming conflicts are illegal according to the
     * Java documentation.<p>
     */
    //@ requires (\forall int i; (0 <= i && i < identifier.length) ==> identifier[i] != null);
    //@ ensures \result.myPackage != null;
    public static /*@ non_null @*/ Resolve_Result lookup(/*@ non_null @*/ Tree filespace, 
                                                         /*@ non_null @*/ String[] identifier)
		throws Resolve_AmbiguousName {
	// Resulting package starts with the top package:
	Tree P = filespace;

	int i = 0;	// Index into identifier as we scan it L->R

	/*
	 * While the next identifier part is the simple name of a
	 * subpackage of the current package, but *not* the simple name
         * of a direct type of the current package, then move down the
	 * package tree.  If it is both, throw Resolve_AmbiguousName.
	 */	
	while (i<identifier.length && !typeExists(P, identifier[i])) {
	  Tree tp = P.getChild(identifier[i]);
	  if (tp == null)
	    break;
	  P = tp;
	  i++;
        }
	if (i<identifier.length && typeExists(P, identifier[i])) {
	    Tree ambiguousPackage = P.getChild(identifier[i]);
	    if (ambiguousPackage != null) {
		throw new Resolve_AmbiguousName("ambiguous name: "
			     + PkgTree.getPackageName(ambiguousPackage)
			     + " is both a class or interface type and"
			     + " a package",
			     ambiguousPackage);
	    }
	}

	/*
	 * Now, find the longest type name that exists (directly) in
	 * the current package.  If X$Y does not exist, then we assume
	 * that X$Y$Z does not exist.
	 */
	String typeName = "";
	while (i<identifier.length &&
	        typeExists(P, combineNames(typeName,identifier[i],"$")))
	    typeName = combineNames(typeName, identifier[i++], "$");
	if (typeName.equals(""))
	    typeName = null;

	// Finally, return the results:
	Resolve_Result answer = new Resolve_Result();
	answer.myPackage = P;
	answer.myTypeName = typeName;
	answer.remainder = new String[identifier.length-i];
	for (int j=0; j+i<identifier.length; j++)
	    answer.remainder[j] = identifier[j+i];
	return answer;
    }


    /***************************************************
     *                                                 *
     * Handling names:				       *
     *                                                 *
     **************************************************/

    /**
     * Combine two names using a separator if both are non-empty. <p>
     *
     */
    //@ requires first != null && second != null;
    //@ ensures \result != null;
    public static String combineNames(String first, String second,
					String separator) {
	if (first.equals("") || second.equals(""))
	    return first+second;
	else
	    return first+separator+second;
    }

    /**
     * Convert a multi-part identifier into a path.  Returns null if
     * the identifier is badly formed (i.e., contains empty
     * components).  id must be non-null. <p>
     *
     * Only uses '.' as a separator.  If you wish to allow '$' as well,
     * use tr first to map all the '$'s in the name into '.'s.<p>
     */
    //@ requires id != null;
    /*@ ensures \result != null ==>
	(\forall int i; (0<=i && i<\result.length) ==> \result[i] != null); */   
    public static String[] parseIdentifier(String id) {
	String[] path = StringUtil.parseList(id, '.');

	for (int i=0; i<path.length; i++)
	   if (path[i].equals(""))
		return null;

	return path;
    }

    /**
     * Do a lookup using the result of parseIdentifier extended to
     * allow '$' as an additional separator.<p>
     *
     * Complains to System.err then returns null if the name is badly
     * formed.  identifier and filespace must be non-null.<p>
     */
    //@ requires filespace != null && identifier != null;
    //@ ensures \result != null ==> \result.myPackage != null;
    public static Resolve_Result lookupName(Tree filespace, String identifier)
				throws Resolve_AmbiguousName {
	// Allow '$' as an additional separator:
	identifier = tr(identifier, '$', '.');

	String[] idPath = Resolve.parseIdentifier(identifier);
	if (idPath==null) {
	    System.err.println(identifier + ": badly formed name");
	    return null;
	}

	return lookup(filespace, idPath);
    }


    /***************************************************
     *                                                 *
     * Maintaining a notion of a current namespace:    *
     *                                                 *
     **************************************************/

    /**
     * The current Java namespace; must be a non-null filespace.<p>
     *
     * Starts out empty.<p>
     */ 
    //@ invariant namespace != null;
    public static Tree namespace = PathComponent.empty();


    /**
     * Attempt to set the current namespace to a new non-null class path.<p>
     *
     * Complains about any errors to System.err.  The current namespace
     * remains unchanged in the case of an error.<p>
     *
     * Iff complain is set, we complain if non-existent
     * or ill-formed path components are present in the classpath.<p>
     */
    //@ requires classpath != null;
    public static void set(String classpath, boolean complain) {
	try {
	    namespace = ClassPath.open(classpath, complain);
        } catch (IOException E) {
	    System.err.println("I/O error: " + E.getMessage());
        }
    }

    /**
     * Attempt to set the current namespace to current classpath (cf.
     * ClassPath).<p>
     *
     * Complains about any errors to System.err.  The current namespace
     * remains unchanged in the case of an error.<p>
     *
     * Iff complain is set, we complain if non-existent
     * or ill-formed path components are present in the classpath.<p>
     */
    public static void init(boolean complain) {
	set(ClassPath.current(), complain);
    }

    /**
     * Convenience function: do a lookupName using the current namespace
     */
    //@ requires identifier != null;
    //@ ensures \result != null ==> \result.myPackage != null;
    public static Resolve_Result lookupName(String identifier) 
				throws Resolve_AmbiguousName {
	return lookupName(namespace, identifier);
    }


    /***************************************************
     *                                                 *
     * Error handling:				       *
     *                                                 *
     **************************************************/

    /**
     * Check the result of a lookup to ensure that it refers to an
     * (inner) reference type or a package.  I.e., that there are no
     * remainder parts.<p>
     *
     * If the check fails, complains appropriately to System.err and then
     * returns null.  If answer is already null, returns null
     * immediately.<p>
     *
     * Otherwise, returns its argument unchanged; the argument will
     * always have a remainder of length 0 in this case.<p>
     */
    /*@ requires answer != null ==> answer.myPackage != null; */
    //@ ensures \result != null ==> \result.myPackage != null;
    public static Resolve_Result ensureUnit(Resolve_Result answer) {
	// Return if check succeeds or answer already null:
	if (answer==null || answer.remainder.length==0)
	    return answer;

	String packageName = answer.myPackage.getQualifiedName(".");
	String unresolved  = answer.remainder[0];

	if (answer.myTypeName==null) {
	    // Didn't find any (potentially enclosing) type at all:
	    System.err.println(Resolve.combineNames(packageName, 
	        unresolved, ".")
		+ ": no such package, class, or interface");
	} else {
	    // Found a potentially enclosing type, but not one
	    // of the inner ones we need:
	    System.err.println(
		Resolve.combineNames(packageName,
		 answer.myTypeName+"$"+unresolved, ".")
		    + ": no such class or interface");
        }

        return null;
    }

    /**
     * Check the result of a lookup to ensure that it refers to an
     * (inner) reference type.<p>
     *
     * If the check fails, complains appropriately to System.err and then
     * returns null.  If answer is already null, returns null
     * immediately.<p>
     *
     * Otherwise, returns its argument unchanged; the argument will
     * have a non-null myTypeName and a remainder with length 0 in
     * this case.<p>
     */
    //@ requires answer != null ==> answer.myPackage != null;
    //@ ensures \result != null ==> \result.myTypeName != null;
    //@ ensures \result != null ==> \result.myPackage != null;
    public static Resolve_Result ensureType(Resolve_Result answer) {
	// Handle the cases where we didn't find a type or a package:
	answer = ensureUnit(answer);
	if (answer==null)
	    return null;

	// Handle the case where typeName names a package:
	if (answer.myTypeName==null) {
	    System.err.println(PkgTree.getPackageName(answer.myPackage)
		+ ": names a package, not a class or interface");
	    return null;
        }

	return answer;
    }


    /***************************************************
     *                                                 *
     * Utility functions:			       *
     *                                                 *
     **************************************************/

    /**
     * Convert 1 character to another everywhere it appears in a given
     * string.
     *
     */
    //@ requires input != null;
    //@ ensures \result != null;
    public static String tr(String input, char from, char to) {
	StringBuffer chars = new StringBuffer(input);

	for (int i=0; i<input.length(); i++)
	    if (chars.charAt(i)==from)
		chars.setCharAt(i,to);

	return chars.toString();
    }


    /***************************************************
     *                                                 *
     * Debugging functions:			       *
     *                                                 *
     **************************************************/

    /** A simple test driver */
    //@ requires args != null;
    /*@ requires (\forall int i; (0<=i && i<args.length)
		==> args[i] != null); */
    public static void main(String[] args) throws IOException {
	/*
	 * Parse command arguments:
	 */
	if (args.length != 1) {
	    System.out.println("Resolve: usage <identifier>");
	    return;
	}

	init(false);
	Resolve_Result answer;
	try {
	    answer = lookupName(args[0]);
	    if (answer==null)
	        return;
	} catch (Resolve_AmbiguousName name) {
	    System.err.println(args[0] + ": " + name.getMessage());
	    return;
	}


	System.out.println("Package name: "
		+ PkgTree.getPackageName(answer.myPackage));
	if (answer.myTypeName==null)
	    System.out.println("No reference-type name");
	else
	    System.out.println("(inner) reference-type name: "
		+ answer.myTypeName);
	System.out.print("Remaining identifier parts: ");
	for (int i=0; i<answer.remainder.length; i++)
	    System.out.print("." + answer.remainder[i]);
	System.out.println();
 
	System.out.println();
	System.out.println("Checking that it's a package or a reference type:");
	ensureUnit(answer);
	System.out.println("Checking that it's a reference type:");
	ensureType(answer);
   }
}
