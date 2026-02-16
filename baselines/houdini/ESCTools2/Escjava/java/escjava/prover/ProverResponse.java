package escjava.prover;

import java.util.Properties;

public class ProverResponse
{
    /**
     * A singleton response code to indicate everything is fine.
     */
    public static /*@ non_null @*/ ProverResponse OK =
	new ProverResponse();

    /**
     * A singleton response code to indicate that something is seriously
     * wrong with the corresponding call and/or the prover and a failure
     * has taken place.  A response code of FAIL typically indicates a
     * non-correctable situation.  The {@link #info} field should be
     * consulted for additional information, and no further calls, but
     * for {@link ProverInterface#stop_prover()} should be made.
     */
    public static /*@ non_null @*/ ProverResponse FAIL =
	new ProverResponse();

    /**
     * A singleton response code to indicate a positive response to the
     * last command.
     */
    public static /*@ non_null @*/ ProverResponse YES =
	new ProverResponse();

    /**
     * A singleton response code to indicate a negative response to the
     * last command.
     */
    public static /*@ non_null @*/ ProverResponse NO =
	new ProverResponse();

    /**
     * A singleton response code to indicate a counter-example is
     * available.  The counter-example is contained in the {@link
     * #formula} field of this response object.
     */
    public static /*@ non_null @*/ ProverResponse COUNTER_EXAMPLE =
	new ProverResponse();

    /**
     * A singleton response code to indicate a syntax error in the
     * corresponding prover call.  The {@link #info} field should be
     * consulted for additional information.
     */
    public static /*@ non_null @*/ ProverResponse SYNTAX_ERROR =
	new ProverResponse();

    /**
     * A singleton response code to indicate that some progress
     * information is available from the prover.  The {@link #info}
     * field should be consulted for additional information.
     */
    public static /*@ non_null @*/ ProverResponse PROGRESS_INFORMATION =
	new ProverResponse();

    /**
     * A singleton response code to indicate that the prover has timed
     * out on the corresponding prover call.  The {@link #info} and/or
     * {@link #formula} fields should be consulted for additional
     * information.
     */
    public static /*@ non_null @*/ ProverResponse TIMEOUT =
	new ProverResponse();

    /**
     * A singleton response code to indicate an inconsistency warning
     * from the prover for one or more of the previous {@link
     * ProverInterface#declare_axiom(Formula)} and {@link
     * ProverInterface#make_assumption(Formula)} calls.  The {@link
     * #info} and/or {@link #formula} fields should be consulted for
     * additional information.
     */
    public static /*@ non_null @*/ ProverResponse INCONSISTENCY_WARNING =
	new ProverResponse();

    /**
     * A formula.  Typically, this field is used to represent a
     * counter-example in response to a call to {@link
     * ProverInterface#is_valid(Formula)}.
     */
    public Formula formula;

    /**
     * A set of properties.  Typically, this field is used to represent
     * property/value pairs specific to reporting prover progress,
     * state, resource utilization, etc.
     */
    public Properties info;

    /**
     * A private constructor that is only to be used during static
     * initialization.
     */
    protected ProverResponse() {
	; // skip
    }

    // placeholder for factory for building ProverResponses
    /*@   ensures \result == ProverResponse.OK ||
      @   \result == ProverResponse.FAIL ||
      @   \result == ProverResponse.YES ||
      @   \result == ProverResponse.NO ||
      @   \result == ProverResponse.COUNTER_EXAMPLE ||
      @   \result == ProverResponse.SYNTAX_ERROR ||
      @   \result == ProverResponse.PROGRESS_INFORMATION ||
      @   \result == ProverResponse.TIMEOUT ||
      @   \result == ProverResponse.INCONSISTENCY_WARNING;
      @*/
    static public /*@ pure non_null @*/ ProverResponse factory(int return_code) {

	/*
	 * Naive implementation, should be redefined in subclasses
	 */

	if( return_code >= 0 )
	    return ProverResponse.OK;
	else
	    return ProverResponse.FAIL;

    }
}
