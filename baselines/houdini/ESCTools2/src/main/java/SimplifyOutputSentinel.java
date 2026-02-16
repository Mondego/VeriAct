/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.prover;

/**
 * Objects of this class represent the summary part of the normal output
 * from Simplify: valid, invalid, or unknown.
 * 
 * @see Simplify
 * @see escjava.prover.CECEnum
 * @see SExp
 */

public class SimplifyOutputSentinel extends SimplifyOutput
{
    /*@ spec_public @*/ int number;

    /*@ normal_behavior
      @   requires kind == VALID || kind == INVALID || kind == UNKNOWN;
      @   modifies this.number;
      @   ensures this.kind == kind;
      @   ensures this.number == number;
      @*/
    SimplifyOutputSentinel(int kind, int number) {
        super(kind);
        this.number = number;
    }
                                                
    public /*@ pure non_null @*/ String toString() {
        return super.toString() + " number=" + number;
    }
}
