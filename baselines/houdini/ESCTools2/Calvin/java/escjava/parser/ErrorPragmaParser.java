/* Copyright Hewlett-Packard, 2002 */

package escjava.parser;

import javafe.parser.PragmaParser;
import javafe.parser.Token;

import javafe.util.Location;
import javafe.util.CorrelatedReader;
import javafe.util.ErrorSet;

import java.io.IOException;


/**
 ** This class produces a PragmaParser that reports an client-chosen
 ** error message each time an annotation comment is encountered.
 ** (Javadoc comments are also considered annotations.)
 **/

public class ErrorPragmaParser implements PragmaParser {
  

    /** The error message to report **/
    public String msg;

    /** Create a new ErrorPragmaParser that report error message msg **/
    public ErrorPragmaParser(String msg) {
	this.msg = msg;
    }


    /** We consider both ESC and Javadoc comments to be annotations **/
    public boolean checkTag(int tag) {
      return tag == '@' || tag == '*';
    }

    /** Report an error for each annotation comment **/
    public void restart(CorrelatedReader in, boolean eolComment) {
	try {
	    int c = in.read();
	} catch (IOException e) {
	    ErrorSet.fatal(in.getLocation(), e.toString());
	}

	ErrorSet.error(in.getLocation(), msg);
    }


    /** Produce no actual pragmas **/
    public boolean getNextPragma(Token dst) { return false; }

    /** No work to close us **/
    public void close() { };
}
