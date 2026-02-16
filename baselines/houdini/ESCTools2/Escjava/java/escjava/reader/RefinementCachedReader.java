/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.reader;

import javafe.reader.CachedReader;
import javafe.reader.Reader;
import javafe.ast.CompilationUnit;
import javafe.genericfile.*;
import javafe.util.ErrorSet;
import javafe.util.Location;
import javafe.ast.Modifiers;
import javafe.tc.TypeSig;
import javafe.tc.OutsideEnv;
import javafe.ast.*;
import escjava.ast.*;
import escjava.AnnotationHandler;
import escjava.RefinementSequence;

import java.util.*;

/**
 * RefinementCachedReader caches compilation units that have been read,
 * as does its super class.  However, before doing so, it retrieves
 * all files of a refinement sequence, combines them and caches the
 * combined result against all of the file names.  Thus if any file in
 * the refinement sequence is read again, the refinement combination will
 * be produced.
 *
 * <p> Reads from GenericFiles with null canonicalIDs are not cached.
 */

public class RefinementCachedReader extends CachedReader
{
    /***************************************************
     *                                                 *
     * Creation:				       *
     *                                                 *
     **************************************************/

    /**
     * The underlying Reader whose results we are caching.
     */
    //@ invariant underlyingReader != null;

    protected /*@ non_null @*/ AnnotationHandler annotationHandler = new AnnotationHandler();

    /**
     * Creating a cached version of a Reader:
     */
    //@ requires reader != null;
    public RefinementCachedReader(/*@ non_null */ Reader reader) {
	super(reader);

	//+@ set cache.keyType = \type(String);
	//+@ set cache.elementType = \type(Object);
    }


    /***************************************************
     *                                                 *
     * The Cache itself:			       *
     *                                                 *
     **************************************************/

	// Inherited from the super class


    /***************************************************
     *                                                 *
     * Caching methods:				       *
     *                                                 *
     **************************************************/

    public /*@ nullable */ CompilationUnit isAlreadyRead(/*@ non_null */ GenericFile target) {
	return (CompilationUnit)get(target);
    }

    /***************************************************
     *                                                 *
     * Reading:					       *
     *                                                 *
     **************************************************/

    /**
     * Attempt to read and parse a CompilationUnit from target.
     * Any errors encountered are reported via javafe.util.ErrorSet.
     * Null is returned iff an error was encountered.<p>
     *
     *
     * By default, we attempt to read only a spec (e.g., specOnly is set
     * in the resulting CompilationUnit) to save time.  If avoidSpec is
     * true, we attempt to return a non-spec, although this may not
     * always be possible.<p>
     *
     *
     * The results of this function (including null results, but not
     * the action of reporting error messages) are cached.
     * Only the value of avoidSpec used the first time a given file is
     * read is used.  This may result in a spec being returned
     * unnecessarily when avoidSpec is true.<p>
     *
     * Target must be non-null.<p>
     */
    public /*@ nullable */ CompilationUnit read(/*@non_null*/GenericFile target, boolean avoidSpec) {
	// Note - reading has the side effect of caching
	CompilationUnit cu = (CompilationUnit)get(target);
	if (cu != null) {
	    cu.duplicate = true;
	    return cu;
/*
Don't complain, but don't do it twice either. ???
	    CompilationUnit cu = (CompilationUnit)get(target);
	    if (cu != null) {
		// If we are processing a package, we do not want to complain
		// about multiple files from a refinement sequence.
		Name n = cu.pkgName;
		if (n != null && escjava.Main.options().
			packagesToProcess.contains(n.printName()))
		    return null;
	    }
	    ErrorSet.caution("Duplicate command-line file " +
				"(or part of a refinement sequence): " +
				target.getHumanName());
*/
	    
	    //return null;
	}

	// not cached - read and do refinement combination
	refinementSequence = null;
	cu = super.read(target,avoidSpec);
	if (cu == null) return null;
	CompilationUnit result = readRefinements(cu,avoidSpec);
	if (result == null) {
	    // Some error occurred (already reported).  In particular, this
	    // null return can happen if the type declared in this file has
	    // already been loaded by (declared in) another file.
	    return result;
	}

	// Do anything to pragmas that must be done before type signatures
	// are created - includes, for example, handling model imports.
	if (result != null) annotationHandler.handlePragmas(result);

	// Put the composite CU in the cache for all elements of the RS.
	Iterator i = refinementSequence.iterator();
	while (i.hasNext()) {
	    CompilationUnit rcu = (CompilationUnit)i.next();
	    put(rcu.sourceFile(),result);
	}
	return result;
    }

    protected /*@ nullable */ ArrayList refinementSequence;

    public /*@ nullable */ CompilationUnit readRefinements(/*@ non_null @*/ CompilationUnit cu, boolean avoidSpec) {
     
	    // Get and parse the package name
	    Name pkgName = cu.pkgName;
	    String pkg = pkgName == null ? "" : pkgName.printName();
	    String[] pkgStrings = pkgName == null ? new String[0] : 
					pkgName.toStrings();


	    // Look through all the types in the compilation unit and
	    // find the public one.  That is the one whose name should
	    // be used to find the refinement files.
	    TypeDeclVec types = cu.elems;
	    Identifier type = null;
	    for (int i=0; i<types.size(); ++i) {
		TypeDecl td = types.elementAt(i);
		if (Modifiers.isPublic(td.modifiers)) {
		    type = td.id;
		    break;
		}
	    }

	    if (type == null) {
		String s = cu.sourceFile().getLocalName();
		int p = s.indexOf('.');
		type = Identifier.intern(s.substring(0,p));
	    }

	    // Check one of the ids to see if the type is already loaded.
	    if (types.size() != 0) {
		String typeToCheck;
		if (type != null) typeToCheck = type.toString();
		else typeToCheck = types.elementAt(0).id.toString();
		TypeSig sig = TypeSig.lookup(pkgStrings,typeToCheck);
		if (sig != null && sig.isPreloaded()) {
		    CompilationUnit pcu = sig.getCompilationUnit();
		    ErrorSet.caution("Type " + pkg + 
			(pkgName==null?"":".") + typeToCheck + 
			" in " + cu.sourceFile().getHumanName() + 
			" is already loaded from " + 
			pcu.sourceFile().getHumanName());
		    return null;
		}
	    }

	    // See if there is a java file for this type.
	    // Note that if there is no public type, then type == null,
	    // and we don't look for a java file.
	    GenericFile javafile = null;
	    if (type == null) {
		// No public type declaration
		// So there are no other files to be found
		// Have to do the refinement processing, because that makes
		// a (singleton) composite set of pragma modifiers

		String s = cu.sourceFile().getLocalName();
		if (s.endsWith(".java")) javafile = cu.sourceFile();
		
	    } else {
	    	javafile = ((EscTypeReader)OutsideEnv.reader()).
			findSrcFile(pkgStrings,type.toString()+".java");
		if (javafile == null &&
			cu.sourceFile().getLocalName().endsWith(".java")) {
		    javafile = cu.sourceFile();
		    ErrorSet.caution(Location.createWholeFileLoc(javafile),
			"Using given file as the .java file, even though it is not the java file for " + pkg + (pkgName==null?"":".") + type + " on the classpath");
		}

	    }

	    // Now find the refinement sequence belonging to the given type.
	    // If there is none, or if type is null, then a refinement sequence
	    // consisting of just the one compilation unit cu is returned.
	    // Note that this parses each of the files in the RS.
	    // Note also that 'cu' need not be in its own RS if it isn't,
	    // then it is not part of the list returned.
	    refinementSequence = getRefinementSequence(pkgStrings, type, cu, avoidSpec);
		// Error occurred (already reported) such that we don't
		// want to add a new compilation unit to the environment
	    if (refinementSequence == null) return null;

	    if (javafe.util.Info.on) {
		java.util.Iterator i = refinementSequence.iterator();
		System.out.print("Refinement Sequence: [");
		while (i.hasNext()) {
			System.out.print(" "+ ((CompilationUnit)i.next()).
					sourceFile().getHumanName());
		}
		System.out.println(" ]");
	    }

	    // Now find the compilation unit for the java file.  If it is
	    // already in the RS or is the same as cu, we don't read it again.  
	    CompilationUnit javacu = null;
	    if (javafile != null) {
		if (javafile.getCanonicalID().equals(cu.sourceFile().getCanonicalID())) javacu = cu;
		else for (int i=0; javacu == null && i<refinementSequence.size(); ++i) {
		    CompilationUnit rcu = (CompilationUnit)refinementSequence.get(i);
		    if (rcu.sourceFile().getCanonicalID().equals(javafile.getCanonicalID()))
			    javacu = rcu;
		}
	    }
	    //System.out.println("HAVE " + cu.sourceFile().getHumanName());
	    if (javacu == null) {
		// We don't already have a CU for the java file (that is, it
		// is not cu or in the RS) so we need to parse the source or
		// binary file.

		// Note - if we are not using any implementation or 
		// specs from the source file, why not just read the 
		// binary if it is available?   FIXME
		if (javafile != null) {
		    // We have a .java source file so read from it.
		    javafe.util.Info.out("Reading source file "
			+ javafile.getHumanName());
		    ErrorSet.caution("The file " + javafile.getHumanName() + 
			" is not in the refinement sequence that begins with " +
			cu.sourceFile().getHumanName() + 
			"; it is used to generate a class signature, but no refinements within it are used.");
		    javacu = underlyingReader.read(javafile, false);
			// The false above means only read a signature and not
			// the implementation or the annotations.
		} else {
		    javacu = getCombinedBinaries(pkgName,pkgStrings,refinementSequence);
		}
	    }

// FIXME - really want a routine that reads binary if up to date otherwise 
// source, simply to get signature.  Read java with bodies if it is one of the
// files to be checked.  The above should be able to be greatly improved!!!!
     
	    CompilationUnit newcu = new RefinementSequence(refinementSequence,
						javacu,annotationHandler);

	    
	    javafe.util.Info.out("Constructed refinement sequence");
	    return newcu;
	}

    /*@ nullable */ CompilationUnit getCombinedBinaries(/*@nullable*/ Name pkgName, 
					/*@ non_null @*/ String[] pkg, 
					/*@ non_null @*/ ArrayList rs) 
    {
	    CompilationUnit combination = null;
	    java.util.List failures = new java.util.LinkedList();
	    Iterator i = rs.iterator();
	    while (i.hasNext()) {
		CompilationUnit cu = (CompilationUnit)i.next();
		TypeDeclVec tdv = cu.elems;
		for (int j=0; j<tdv.size(); ++j) {
		    TypeDecl td = tdv.elementAt(j);
		    Identifier id = td.id;
		    boolean found = false;
		    if (combination != null) {
			for (int k = combination.elems.size()-1; k>=0; --k) {
			    if (combination.elems.elementAt(k).id == id) {
				found = true;
				break;
			    }
			}
		    }
		    if (!found) {
			GenericFile javafile = ((EscTypeReader)OutsideEnv.reader()).
			    findBinFile(pkg,id.toString()+".class");
			if (javafile != null) {
			    javafe.util.Info.out("Reading class file "
				+ javafile.getHumanName());
			    CompilationUnit
				javacu = ((EscTypeReader)OutsideEnv.reader()).
				    binaryReader.read(javafile, false); 
			    if (combination == null)
				combination = javacu;
			    else {
				TypeDeclVec ntdv = javacu.elems;
				for (int n=0; n<ntdv.size(); ++n) {
				    combination.elems.addElement(ntdv.elementAt(n));
				}
			    }
			} else {
				failures.add(
					(pkgName==null? id.toString() :
					    pkgName.printName() + "." + id));
			}
		    }
	        }
	    }
	    if (combination != null && failures.size() != 0) {
		// FIXME - should marak the source location for these
		String s = "Failed to find some but not all binary files: ";
		Iterator ii = failures.iterator();
		while (ii.hasNext()) s += ii.next();
		ErrorSet.error(s);
	    }
	    return combination;
	}


	// result is a list of CompilationUnits
	// result will contain something, perhaps just the given cu
    //@ ensures \result != null;
	/*@ non_null */ ArrayList getRefinementSequence(/*@ non_null @*/ String[] pkgStrings,
					/*@ nullable @*/ Identifier type, 
					/*@ non_null @*/ CompilationUnit cu, 
					boolean avoidSpec) {
	    ArrayList refinements = new ArrayList();
	    GenericFile mrcufile;
	    GenericFile gf = cu.sourceFile();
	    String gfid = gf.getCanonicalID();
	    if (type == null) {
		mrcufile = gf;
	    } else {
		mrcufile =
		    ((EscTypeReader)OutsideEnv.reader()).findFirst(
			    pkgStrings,type.toString());
	    }
	    javafe.util.Info.out( mrcufile==null ? "No MRCU found" :
			"Found MRCU " + mrcufile);
	    // If no MRCU is found in the sourcepath, then we presume that
	    // the file on the command line is that.
	    GenericFile gfile = (mrcufile == null) ? gf : mrcufile ;
	    CompilationUnit ccu;
	    boolean foundCommandLineFileInRS = false;
	    while(gfile != null) {
		if (gfile.getCanonicalID().equals(gfid)) {
		    // Avoid parsing a file twice
		    ccu = cu;
		    foundCommandLineFileInRS = true;
                } else {
		    ccu = underlyingReader.read(gfile,avoidSpec);
		}
		annotationHandler.parseAllRoutineSpecs(ccu);
		refinements.add(ccu);
		gfile = findRefined(pkgStrings,ccu);
		if (gfile != null) {
		  if (!gfile.getLocalName().startsWith(type.toString() + ".")){
			ErrorSet.caution("The refinement file " + 
			    gfile.getHumanName() +
			    " in the sequence beginning with " + 
			    mrcufile.getHumanName() +
			    " has a prefix that does not match the type name "
			    + type);
		  }
		  for (int i=0; i<refinements.size(); ++i) {
		    if ( ((CompilationUnit)refinements.get(i)).sourceFile().
			getCanonicalID().equals( gfile.getCanonicalID() )) {
			ErrorSet.error(gfile.getHumanName() + 
				" is circularly referenced in a refinement sequence");
			gfile = null;
			break;
		    }
		  }
		}
	    }
	    if (!foundCommandLineFileInRS) {
		String pkg = cu.pkgName == null ? "" : 
				cu.pkgName.printName() + ".";
		if (refinements.size() == 0) {
		    // If no refinement sequence was found, we simply use the
		    // file on the command line, even if it is not on the
		    // classpath.
		    refinements.add(cu);
		} else {
		    StringBuffer err = new StringBuffer(
			"The command-line argument " 
			+ cu.sourceFile().getHumanName() 
			+ " was not in the refinement sequence for type " 
			+ pkg + type.toString() + ":");
		    for (int k = 0; k<refinements.size(); ++k) {
			err.append(" ");
			err.append(((CompilationUnit)refinements.get(k)).
					sourceFile().getHumanName());
		    }
		    // If the command-line file is not in the refinement
		    //	sequence, we use the refinement sequence, since,
		    //	if the type was referenced from another class it
		    //	is the refinement sequence that would be found.
		    //
		    ErrorSet.error(err.toString());
		}
	    }
	    javafe.util.Info.out("Found refinement sequence files");
	    return refinements;
	}

    public static /*@ nullable */ GenericFile findRefined(/*@ non_null @*/ String[] pkgStrings, 
					  /*@ non_null @*/ CompilationUnit cu) 
    {
	    LexicalPragmaVec v = cu.lexicalPragmas;
	    for (int i=0; i<v.size(); ++i) {
		if (v.elementAt(i) instanceof RefinePragma) {
		    RefinePragma rp = (RefinePragma)v.elementAt(i);
		    String filename = rp.filename;
			// FIXME - what if we are refining a class file ???
		    GenericFile gf = ((EscTypeReader)OutsideEnv.reader()).findSrcFile(pkgStrings,filename);
		    if (gf == null) ErrorSet.error(rp.loc,
			"Could not find file referenced in refine annotation: " + filename);
		    return gf;
     
    // FIXME - should be able to have refined files that are not in regular files as well
		}
	    }
	    return null;
	}

}
