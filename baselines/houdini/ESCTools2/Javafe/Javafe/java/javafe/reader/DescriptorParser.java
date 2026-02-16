/* Copyright 2000, 2001, Compaq Computer Corporation */

/* =========================================================================
 * DescriptorParser.java
 * ========================================================================= */

package javafe.reader;

import java.util.*;

import javafe.ast.*;
import javafe.util.*;

/* -------------------------------------------------------------------------
 * DescriptorParser
 * ------------------------------------------------------------------------- */

/**
 * Parses various kinds of class-file-format descriptor strings into
 * abstract syntax.
 */

class DescriptorParser
{
  /* -- package class methods ---------------------------------------------- */

  /**
   * Return a type name for a given class-file class-name string.
   * @param s  the class name to parse
   * @return   the type name encoded by s
   */
    static /*@non_null*/TypeName parseClass(/*@ non_null @*/ String s)
    throws ClassFormatError
  {
    // tokenize the string into a sequence of identifiers delimited by slashes
    // check syntax of identifiers? ???

    StringTokenizer tokenizer   = new StringTokenizer(s, "/$");

    int             count       = tokenizer.countTokens();
    javafe.util.Assert.notFalse(count>0);	//@ nowarn Pre;

    Identifier[]    identifiers = new Identifier[count];
    int[]           locations1  = new int[count];
    int[]           locations2  = new int[count-1];

    for (int i = 0; tokenizer.hasMoreTokens(); i++)
    {
      identifiers[i] = Identifier.intern(tokenizer.nextToken());
      //@ assert identifiers[i] != null;
      locations1[i]  = classLocation;

      if (i<count-1)
	locations2[i] = classLocation;
    }
    Identifier[/*#@non_null*/] nn_identifiers = identifiers;
    //@ assert \nonnullelements(identifiers);
    /*@ assume (\forall int i; (0<=i && i<locations1.length)
			==> locations1[i] != Location.NULL); */
    /*@ assume (\forall int i; (0<=i && i<locations2.length)
			==> locations2[i] != Location.NULL); */

    return TypeName.make(Name.make(locations1, locations2,
				   IdentifierVec.make(nn_identifiers)));
  }

  /**
   * Return a type for a given class-file field descriptor string.
   * @param s  the field descriptor to parse
   * @return   the type encoded by s
   */
  //@ ensures \result.syntax;
  static /*@non_null*/Type parseField(/*@non_null*/String s)
    throws ClassFormatError
  {
    // parse the descriptor as a type and make sure it's only a type

    StringScanner scanner = new StringScanner(s);
    Type          type    = parseType(scanner);

    if (scanner.index<s.length())
      throw new ClassFormatError("junk after field descriptor");

    return type;
  }

  /**
   * Return a method signature for a given class-file method descriptor string.
   * @param s  the method descriptor to parse
   * @return   the method signature encoded by s
   */
  static /*@non_null*/MethodSignature parseMethod(/*@non_null*/String s)
    throws ClassFormatError
  {
    // check the format of the method descriptor and construct a string scanner

    int length = s.length();

    if (length<3)
      throw new ClassFormatError("incomplete method descriptor");

    if (s.charAt(0) != '(')
      throw new ClassFormatError("invalid method descriptor");

    StringScanner scanner = new StringScanner(s);

    scanner.index++;

    // parse the parameter types in the method descriptor and accumulate them
    // in a method signature

    MethodSignature signature = new MethodSignature(classLocation);

    while (s.charAt(scanner.index) != ')')
    {
      signature.appendParameter(parseType(scanner));
      
      if (scanner.index>=length)
	throw new ClassFormatError("incomplete method descriptor");
    }

    scanner.index++;

    // parse the return type of the method descriptor and store it in the
    // method signature

    signature.setReturn(parseReturn(scanner));

    if (scanner.index<length)
      throw new ClassFormatError("junk after method descriptor");

    return signature;
  }

  /* -- package class variables -------------------------------------------- */

  /**
   * A dummy location representing the class being parsed.
   * Should be set externally.
   */
  //@ invariant classLocation != Location.NULL;
  static int classLocation = Location.createFakeLoc("[unknown]");

  /* -- private class methods ---------------------------------------------- */

  /**
   * Parse a type from a given class-file field descriptor string.
   * @param scanner  the string scanner to parse from (modified to point to
   *                 the next character after the parsed type)
   * @return         the type encoded by scanner
   */
  //@ ensures \result.syntax;
  private static /*@non_null*/Type parseType(/*@non_null*/StringScanner scanner)
    throws ClassFormatError
  {
    // parse the type according to the leading character
    // it would be more efficient to share primitive types ???

    String s     = scanner.s;
    int    index = scanner.index;

    if (index>=s.length())
      throw new ClassFormatError("empty type descriptor");

    switch (s.charAt(index))
    {
    case 'B':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.BYTETYPE, classLocation);

    case 'C':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.CHARTYPE, classLocation);

    case 'D':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.DOUBLETYPE, classLocation);

    case 'F':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.FLOATTYPE, classLocation);

    case 'I':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.INTTYPE, classLocation);

    case 'J':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.LONGTYPE, classLocation);

    case 'L':
      {
	// extract the class name and parse it

	int start = index+1, stop = s.indexOf(';', start);

	if (stop<0)
	  throw new ClassFormatError("unterminated type name");

	scanner.index = stop+1;

	return parseClass(s.substring(start, stop));
      }

    case 'S':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.SHORTTYPE, classLocation);

    case 'Z':
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.BOOLEANTYPE, classLocation);

    case '[':
      // parse the element type and construct an array type from it

      scanner.index++;

      return ArrayType.make(parseType(scanner), classLocation);

    default:
      throw new ClassFormatError("unknown type character");
    }
  }

  /**
   * Parse a type from a given class-file return descriptor string.
   * @param scanner  the string scanner to parse from (modified to point to
   *                 the next character after the parsed type)
   * @return         the type encoded by scanner
   */
  //@ ensures \result.syntax;
  private static /*@non_null*/Type parseReturn(/*@non_null*/StringScanner scanner)
    throws ClassFormatError
  {
    // look for the void return descriptor

    String s     = scanner.s;
    int    index = scanner.index;

    if (index>=s.length())
      throw new ClassFormatError("empty return descriptor");

    if (s.charAt(index)=='V')
    {
      scanner.index++;
      return JavafePrimitiveType.make(ASTTagConstants.VOIDTYPE, classLocation);
    }

    return parseType(scanner);
  }

}

/* -------------------------------------------------------------------------
 * StringScanner
 * ------------------------------------------------------------------------- */

/**
 * A string and the index of the next character to scan from it.
 */

class StringScanner {

  /* -- package instance methods ------------------------------------------- */

  /**
   * Construct a new string scanner from a given string.
   * @param s  the string
   */
  StringScanner(/*@non_null*/String s)
  {
    this.s = s;
  }

  /* -- package instance variables ----------------------------------------- */

  /**
   * The string to be scanned.
   * Initialized by constructor.
   */
  /*@non_null*/String s;

  /**
   * The index of the next character to scan.
   */
  int index = 0;

}

