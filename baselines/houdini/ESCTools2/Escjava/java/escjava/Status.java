package escjava;

import java.awt.Color;

public class Status {
    final static public int ILLEGAL = -1; // A value that should only be used
					  // for temporary initialization
    final static public int NOTPROCESSED = 0;
    final static public int RESOLVED_OK = 1;
    final static public int RESOLVED_CAUTION = 2;
    final static public int RESOLVED_ERROR = 3;
    final static public int RESOLVED_COMPLETE = RESOLVED_ERROR;
    final static public int PARSED_OK = 4;
    final static public int PARSED_CAUTION = 5;
    final static public int PARSED_ERROR = 6;
    final static public int PARSING_COMPLETE = PARSED_ERROR;
    final static public int TYPECHECKED_ERROR = 7;
    final static public int TYPECHECKED_CAUTION = 8;
    final static public int TYPECHECKED_WAITING = 9;
    final static public int TYPECHECKED_OK = 10;
    final static public int TYPECHECK_COMPLETE = TYPECHECKED_OK;
    final static public int STATICCHECKED_ERROR = 11;
    final static public int STATICCHECKED_CAUTION = 12;
    final static public int STATICCHECKED_OK = 13;
    final static public int STATICCHECKED_PASSEDIMMED = 14;
    final static public int STATICCHECKED_TIMEOUT = 15;
    final static public int CHILDERROR = 16;

    public static boolean isOK(int s) {
	return  s == RESOLVED_OK || s == PARSED_OK
		|| s == TYPECHECKED_OK || s == STATICCHECKED_OK
		|| s == TYPECHECKED_WAITING
		|| s == STATICCHECKED_PASSEDIMMED;
    }

    public static boolean isError(int s) {
	return s == RESOLVED_ERROR || s == PARSED_ERROR
		|| s == TYPECHECKED_ERROR || s == STATICCHECKED_TIMEOUT
		|| s == STATICCHECKED_ERROR;
    }

    final static public boolean checkComplete(int s) {
	return s > TYPECHECK_COMPLETE;
    }

    final static public boolean typecheckComplete(int s) {
	return s > PARSING_COMPLETE;
    }

    final static public boolean parsingComplete(int s) {
	return s > RESOLVED_COMPLETE;
    }

    final static public boolean resolved(int s) {
	return s > NOTPROCESSED;
    }

    static public /*@ non_null @*/ String toString(int status) {
	switch (status) {
	    case NOTPROCESSED: return "Not Processed";
	    case RESOLVED_OK: return "Resolved OK";
	    case RESOLVED_CAUTION: return "Resolved with Cautions";
	    case RESOLVED_ERROR: return "Resolved with Errors";
	    case PARSED_OK: return "Parsed OK";
	    case PARSED_CAUTION: return "Parsed with Cautions";
	    case PARSED_ERROR: return "Parsed with Errors";
	    case TYPECHECKED_ERROR: return "Typechecked with Errors";
	    case TYPECHECKED_CAUTION: return "Typechecked with Cautions";
	    case TYPECHECKED_WAITING: return "Typechecking children";
	    case TYPECHECKED_OK: return "Typechecked OK";
	    case STATICCHECKED_ERROR: return "Static checked with Warnings";
	    case STATICCHECKED_CAUTION: return "Static checked with Cautions";
	    case STATICCHECKED_OK: return "Static checked OK";
	    case STATICCHECKED_TIMEOUT: return "Static checking timed out";
	    case STATICCHECKED_PASSEDIMMED: return "Static checking passed - no body to check";
	    case CHILDERROR: return "Child node has an error";
	}
	return "ILLEGAL VALUE";
    }

    static public /*@ nullable */ Color toColor(int status) {
	switch (status) {
	    case TYPECHECKED_WAITING:
	    case NOTPROCESSED: return null;
	    case RESOLVED_ERROR:
	    case PARSED_ERROR: return ColorOptions.typecheckError;
	    case RESOLVED_CAUTION:
	    case PARSED_CAUTION: return ColorOptions.typecheckCaution;
	    case RESOLVED_OK:
	    case PARSED_OK: return ColorOptions.typecheckOK;
	    case TYPECHECKED_ERROR: return ColorOptions.typecheckError;
	    case TYPECHECKED_CAUTION: return ColorOptions.typecheckCaution;
	    case TYPECHECKED_OK: return ColorOptions.typecheckOK;
	    case STATICCHECKED_ERROR: return ColorOptions.staticcheckError;
	    case STATICCHECKED_CAUTION: return ColorOptions.typecheckCaution;
	    case STATICCHECKED_PASSEDIMMED: return ColorOptions.staticcheckOK;
	    case STATICCHECKED_OK: return ColorOptions.staticcheckOK;
	    case STATICCHECKED_TIMEOUT: return ColorOptions.staticcheckTimeout;
	    case CHILDERROR: return ColorOptions.childError;
	}
	return null;
    }
}
