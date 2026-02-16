/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.tc;

import javafe.ast.*;
import javafe.util.*;

/**
 * This module is responsible for handling <code>CompilationUnit</code>-level type
 * checks.
 *
 * <p> In practice, these are mostly checks that the set of import declarations is
 * legal. </p>
 */

public class CheckCompilationUnit
{
    /**
     * A new field for CompilationUnits: iff it is non-null then we have already
     * checked that CompilationUnit.
     */
    //@ invariant checkedField != null;
    //@ invariant checkedField.decorationType == \type(Boolean);
    //@ spec_public
    private static ASTDecoration checkedField =
	new ASTDecoration("javafe.tc.CheckCompilationUnit.checked");

    /**
     * Check a <code>CompilationUnit</code>.
     *
     * <p> If this method is called multiple times on the same
     * <code>CompilationUnit</code>, it has no effect the second and later
     * times. </p>
     *
     * <p> Precondition: <code>cu</code> must have already been loaded by
     * <code>OutsideEnv</code>. </p>
     *
     * <p> Any resulting errors or warnings are reported via
     * <code>ErrorSet</code>. </p>
     */
    //@ requires cu != null;
    public static void checkCompilationUnit(CompilationUnit cu) {
	// Check any given CompilationUnit at most once:
	if (checkedField.get(cu) != null)
	    return;
	checkedField.set(cu, Boolean.TRUE);

	Info.out("[checkCompilationUnit: checking for "
			+ Location.toFileName(cu.loc) + "]");


	// Warn about "file local" types declared public:
	checkPublic(cu);

	// Check to make sure that each import is individually well formed:
	checkImports(cu);
	
	// Don't declare two decls with same name in same package 
	//   This is implemented by OutsideEnv -- mdl.
	// Don't declare two types with the same name in the same file
	//   This is implemented in OutsideEnv -- mdl.

	ImportDeclVec imports = cu.imports;
	TypeDeclVec elems = cu.elems;

	// Check for "import P.T" and "import Q.T" pairs.  Such a pair * is legal iff
        // P=Q.

	// Moreover, check for "import P.T" where T is defined in this
	// CompilationUnit.  Such imports are legal iff P is the name of this
	// package.

	for (int i = 0; i < imports.size(); i++) {
	    // Skip to the next single-type import:
	    ImportDecl idecl = imports.elementAt(i);
	    if (!(idecl instanceof SingleTypeImportDecl))
		continue;

	    // Get the full name and unqualified name of the type it's importing:
	    Name N1 = ((SingleTypeImportDecl)idecl).typeName.name;
	    Identifier T1 = N1.identifierAt(N1.size()-1);

	    // Check for import N1, import Q.T1 pairs:

	    for (int j = i+1; j < imports.size(); j++) {
		// Skip to the next single-type import:
		ImportDecl jdecl = imports.elementAt(j);
		if (!(jdecl instanceof SingleTypeImportDecl))
		    continue;

		// Get the full name and unqualified name of the type it's importing:
		Name N2 = ((SingleTypeImportDecl)jdecl).typeName.name;
		Identifier T2 = N2.identifierAt(N2.size()-1);
	  
		if (T1==T2 && !(N1.equals(N2)))
		    ErrorSet.fatal(N1.locIdAt(0),
			"Attempt to import both " + N1.printName()
			    + " and " + N2.printName()
			    + " as " + T1
			    + " using single-type-import declarations");
	    }

	    // Check to see if N1 is of the form P.T where T is declared in this
	    // CompilationUnit and P is not this CompilationUnit's package:
	    for (int j=0; j<elems.size(); j++) {
		if (T1 != elems.elementAt(j).id)
		    continue;

		// Skip if N1=<our package>.T1 :
		if (cu.pkgName==null) {
		    if (N1.size()==1)
			continue;
		} else {
		    if (N1.size()>1 &&
			(N1.prefix(N1.size()-1).equals(cu.pkgName)))
			continue;	// @bug fix for inner classes !!!!
		}

	        ErrorSet.fatal(N1.locIdAt(0),
			N1.printName()
			+ " can not be imported here because "
			+ T1 + " is already defined at "
			+ Location.toString(elems.elementAt(j).loc));
	    }
	}
    }

    // Private methods

    /**
     * Check a <code>CompilationUnit<code> to make sure that the only type that may
     * be declared public in it is the one with the same name as the file it occurs
     * in.  Violations result in warnings only.
     */

    //@ requires cu != null;
    private static void checkPublic(CompilationUnit cu) {
	// Get the basename of the file we loaded cu from:
	String localname = Location.toFile(cu.loc).getLocalName();
	String basename = javafe.filespace.Extension.getBasename(localname);

	// Iterate over all the TypeDecls representing package-member
	// types in cu:
	TypeDeclVec elems = cu.elems;
	for (int i=0; i<elems.size(); i++) {
	    TypeDecl decl = elems.elementAt(i);
	    if (Modifiers.isPublic(decl.modifiers)) {
		String T = decl.id.toString();		// decl's typename
		if (!T.equals(basename)) {
		    String msg =
			("type " + TypeSig.getSig(decl).getExternalName()
			 + " must be declared in a file named "
			 + T + ".java if it is to be declared public.");
		    ErrorSet.caution(decl.locId, msg);
		}
	    }
	}
    }


    /**
     * Check a <code>CompilationUnit<code> to make sure that each import is
     * individually well formed.
     *
     * <p> In particular,
     * <ul>
     *   <li> Type in single import must exist </li>
     *   <li> Single imports from other packages must be public (not implemented)
     *   </li>
     *   <li> Packages in all imports must be "accessible" (disabled) </li>
     * </ul>
     * <p> This routine does not check legality of pairs of import statements or
     * import statements and the types declared in the rest of the
     * CompilationUnit. </p>
     */
    //@ requires cu != null;
    private static void checkImports(CompilationUnit cu) {
	ImportDeclVec imports = cu.imports;

	for (int i = 0; i < imports.size(); i++) {
	    ImportDecl idecl = imports.elementAt(i);
	    if (idecl instanceof SingleTypeImportDecl) {
		// Single-type import:
		Name N = ((SingleTypeImportDecl)idecl).typeName.name;
		int sz = N.size();
		String[] P = N.toStrings(sz-1);
		Identifier T = N.identifierAt(sz-1);

		TypeSig r = EnvForCU.lookupWithoutInheritence(null, P, T.toString());
		if (r==null)
		    ErrorSet.error(N.getStartLoc(),
				   "No such type: " +
				   PrettyPrint.inst.toString(N));
	    } else {
		// On-demand import:
/* [disabled]
		Name N = ((OnDemandImportDecl)idecl).pkgName;
		int sz = N.size();
		if (sz>0) {
		    String[] P = N.toStrings(sz-1);
		    Identifier T = N.identifierAt(sz-1);

		    TypeSig r =
			EnvForCU.lookupWithoutInheritence(P, T.toString());
		    if (r != null)
			continue;
		}
		if (!OutsideEnv.reader.accessable(N.toStrings()))
		    ErrorSet.error(N.getStartLoc(),
				   PrettyPrint.inst.toString(N)
				   + ": no such type or package");
*/
	    }
	}
    }
} // end of class CheckCompilationUnit

/*
 * Local Variables:
 * Mode: Java
 * fill-column: 85
 * End:
 */
