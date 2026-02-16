package escjava.sortedProver.cvc3;

import cvc3.FormulaValue;

/** A truth value of a formula. */
public class Cvc3FormulaValue extends cvc3.FormulaValue {
    protected Cvc3FormulaValue(String result) {
	super(result);
    }

    // names of c++ enum values
    public static final FormulaValue NOT_TRUE = new Cvc3FormulaValue("NOT_TRUE");
    public static final FormulaValue NOT_FALSE = new Cvc3FormulaValue("NOT_FALSE");

    // the FormulaValue corresponding to a c++ enum value by name
    public static FormulaValue get(String value) throws cvc3.DebugException {
	if (value.equals(NOT_TRUE.toString())) {
	    return NOT_TRUE;
	} else if (value.equals(NOT_FALSE.toString())) {
	    return NOT_FALSE;
	} else {
	    return cvc3.FormulaValue.get(value);
	}
    }
}
