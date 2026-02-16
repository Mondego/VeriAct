/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.reader;

import java.io.*;

import javafe.ast.CompilationUnit;
import javafe.ast.TypeDecl;
import javafe.ast.TypeDeclVec;
import javafe.ast.TypeDeclElemVec;
import javafe.ast.ImportDeclVec;
import javafe.ast.PrettyPrint; // Debugging methods only

import javafe.genericfile.*;

/**
 * A BinReader is a Reader that reads in CompilationUnits from binary files
 * (.class files).
 * 
 * <p>
 * BinReaders do not cache the results of their reading and always return spec
 * files.
 * 
 * <p>
 * TypeDecls produced by BinReaders always have specOnly set.
 * 
 * <p>
 * This version is not known to work on Java 1.1 class files. Later versions are
 * planned to return CompilationUnits with stubs where inner classes should be.
 * It is then the caller's responsibility to call this class repeatedly to
 * obtain all the inner classes then stitch them together in a single seamless
 * CompilationUnit. (Most likely, a new class will be introduced to perform this
 * function, mapping P.N's to Compilation Units.
 */

public class BinReader extends Reader {
	/***************************************************************************
	 * * Reading: * *
	 **************************************************************************/

	/**
	 * Attempt to read and parse a CompilationUnit from *binary* target. Any
	 * errors encountered are reported via javafe.util.ErrorSet. Null is
	 * returned iff an error was encountered.
	 * <p>
	 * 
	 * 
	 * We always return a spec file. (avoidSpec is ignored)
	 * <p>
	 * 
	 * 
	 * This function is not cached.
	 * <p>
	 * 
	 * Target must be non-null.
	 * <p>
	 */
	public CompilationUnit read(/*@non_null */GenericFile target,
			boolean avoidSpec) {
		javafe.util.Info.out("[loading " + target.getHumanName() + "]");

		// Use BCEL to parse Java 1.5 and later bytecode
		if ((System.getProperty("java.version").indexOf("1.7") != -1)||
                    (System.getProperty("java.version").indexOf("1.6") != -1)||
		    (System.getProperty("java.version").indexOf("1.5") != -1)) {

			BCELReader bcelReader = new BCELReader();
			return bcelReader.read(target, avoidSpec);
		} else {
			try {
				ASTClassFileParser parser = new ASTClassFileParser(target,
						false);
				// if the 2nd argument above is false, omit all bodies.
				// For some reason, despite the comment above, bodies are
				// sometimes included.
				TypeDeclVec types = TypeDeclVec
						.make(new TypeDecl[] { parser.typeDecl });
				TypeDeclElemVec emptyTypeDeclElemVec = TypeDeclElemVec.make();
				ImportDeclVec emptyImportDeclVec = ImportDeclVec.make();
				CompilationUnit result = CompilationUnit.make(
						parser.classPackage, null, emptyImportDeclVec, types,
						parser.classLocation, emptyTypeDeclElemVec);
				return result;
			} catch (IOException e) {
				javafe.util.ErrorSet.error("I/O error: " + e.getMessage());
			} catch (ClassFormatError e) {
				javafe.util.ErrorSet.error("Class format error: "
						+ e.getMessage());
			}

			return null;
		}
	}

	/***************************************************************************
	 * * Test methods: * *
	 **************************************************************************/

	//@ requires \nonnullelements(args);
	public static void main(String[] args) {
		if (args.length != 1) {
			System.err.println("BinReader: <source filename>");
			System.exit(1);
		}

		GenericFile target = new NormalGenericFile(args[0]);
		BinReader reader = new BinReader();

		CompilationUnit cu = reader.read(target, false);
		if (cu != null)
			PrettyPrint.inst.print(System.out, cu);
	}
}
