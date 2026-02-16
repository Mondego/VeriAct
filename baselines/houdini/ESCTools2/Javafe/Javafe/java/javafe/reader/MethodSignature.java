/* Copyright 2000, 2001, Compaq Computer Corporation */

/* =========================================================================
 * MethodSignature.java
 * ========================================================================= */

package javafe.reader;

import java.util.*;

import javafe.ast.*;
//@ model import javafe.util.Location;

/* -------------------------------------------------------------------------
 * MethodSignature
 * ------------------------------------------------------------------------- */

/**
 * Represents the signature of a method in terms of AST elements.
 */

class MethodSignature
{
  /* -- package instance methods ------------------------------------------- */

  /**
   * Construct a new method signature with an empty sequence of parameter
   * types and a void return type.
   */
  //@ requires classLocation != Location.NULL;
  MethodSignature(int classLocation)
  {
    this.parameters = new Vector();
    this.return_    = JavafePrimitiveType.make(ASTTagConstants.VOIDTYPE, classLocation);

    //@ set parameters.elementType = \type(Type);
    //@ set parameters.containsNull = false;
  }

  /**
   * Count the number of parameter types in this method signature.
   * @return  the number of parameter types
   */
  //@ ensures \result>=0;
  //@ ensures \result==parameters.elementCount;
  int countParameters()
  {
    return parameters.size();
  }

  /**
   * Return a parameter type from this method signature.
   * @param index  the index of the parameter type to return
   * @return       the parameter type at index index
   */
  //@ requires 0<=index && index<parameters.elementCount;
  //@ ensures \result.syntax;
  /*@non_null*/Type parameterAt(int index)
  {
    return (/*+@non_null*/Type)parameters.elementAt(index);
  }    //@ nowarn Post;  // Unenforceable invariant on parameters

  /**
   * Append a parameter type to this method signature.
   * @param parameterType  the parameter type to append
   */
  void appendParameter(/*@non_null*/Type parameterType)
  {
    parameters.addElement(parameterType);
  }

  /**
   * Return the return type of this method signature.
   * @return  the return type
   */
  //@ ensures \result.syntax;
  /*@non_null*/Type getReturn()
  {
    return return_;
  }

  /**
   * Change the return type of this method signature.
   * @param return_  the new return type
   */
  //@ requires return_.syntax;
  void setReturn(/*@non_null*/Type return_)
  {
    this.return_ = return_;
  }

  /* -- private instance variables ----------------------------------------- */

  /**
   * The parameter types of this method signature.
   * Initialized by constructor.
   */
  //@ invariant parameters.elementType == \type(Type);
  //@ invariant !parameters.containsNull;
  // Unenforceable invariant: contents are syntax
  /*@spec_public*/ private /*@non_null*/Vector parameters;

  /**
   * The return type of this method signature.
   * Initialized by constructor.
   */
  //@ invariant return_.syntax;
  //@ spec_public
  private /*@non_null*/Type return_;

}

