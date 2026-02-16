/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;

import javafe.util.ClipPolicy;
import escjava.ast.TagConstants;

public class AssocDeclClipPolicy extends ClipPolicy {

  public boolean containsEndOfConstruct(/*@ non_null @*/ String s, int pos) {
    if (s.startsWith(TagConstants.toString(TagConstants.NON_NULL), pos) ||
	s.startsWith(TagConstants.toString(TagConstants.UNINITIALIZED), pos) ||
	(s.startsWith(TagConstants.toString(TagConstants.MONITORED), pos) &&
	 !s.startsWith(TagConstants.toString(TagConstants.MONITORED_BY),
		       pos))) {
      return true;
    }

    // look forward
    FORWARD:
    for (int i = pos; i < s.length(); i++) {
      char ch = s.charAt(i);
      // System.out.println("DEBUG:  FORWARD ch='" + ch + "'");
      if (ch == '\"') {
	// Find end of string literal.
	do {
	  i = s.indexOf('\"', i+1);
	  if (i == -1) {
	    // Something's strange is going on; this should never happen.
	    // End the forward search
	    break FORWARD;
	  }
	  // check the character at i-1 for being a backslash
	} while (s.charAt(i-1) == '\\');
      } else if (ch == '\'') {
	// Find end of character literal
	// The following loop should never execute more than twice.
	do {
	  i = s.indexOf('\'', i+1);
	  if (i == -1) {
	    // Something's strange is going on; this should never happen.
	    // End the forward search
	    break FORWARD;
	  }
	  // check the character at i-1 for being a backslash
	} while (s.charAt(i-1) == '\\');
      } else if (ch == '\\') {
	// escape character
	i++;  // This will skip over one of the escaped characters, but that's
              // good enough even if there's more than one escaped character.
      } else if (ch == ';') {
	// This indicates the end of the pragma
	return true;
      } else if (s.startsWith("//", i)) {
	// The rest of the line is a comment, so we have not yet found the end
	// of the current construct.
	break FORWARD;
      } else if (s.startsWith("/*", i)) {
	int k = s.indexOf("*/", i+2);
	if (k == -1) {
	  // This should never happen.
	  break FORWARD;
	}
	// skip nested comment
	i = k+1 /* +1 as will be done by the 'for' loop */;
      } else if (s.startsWith("*/", i) || s.startsWith("</esc>", i)) {
	return true;
      }
    }

    // look backward (but if we reach character 0, we have no hope, so let's
    // loop back only until 1; this also means that we can use i-1 as an index
    // inside the loop body)
    for (int i = pos; 1 <= --i; ) {
      char ch = s.charAt(i);
      // System.out.println("DEBUG:  BACKWARD ch='" + ch + "'");
      if (ch == '/') {
	if (s.charAt(i-1) == '/') {
	  // the pragma ends at end-of-line
	  return true;
	}
      } else if (ch == '\"') {
	// Search backwards for the beginning of the string literal.
	do {
	  i = s.lastIndexOf('\"', i-1);
	  if (i == -1) {
	    // This should never happen.  Since things seem screwed up, let's just
	    // return 'true', which will have the effect of not introducing an
	    // ellipsis (which would probably just confuse the user even more).
	    return true;
	  }
	} while (0 < i && s.charAt(i-1) == '\\');
      } else if (ch == '\'') {
	// Search backwards for the beginning of the character literal.
	// The following loop should never execute more than twice.
	do {
	  i = s.lastIndexOf('\'', i-1);
	  if (i == -1) {
	    // This should never happen.  Since things seem screwed up, let's just
	    // return 'true', which will have the effect of not introducing an
	    // ellipsis (which would probably just confuse the user even more).
	    return true;
	  }
	} while (0 < i && s.charAt(i-1) == '\\');
      } else if (s.startsWith("*/", i-1) ||
		 (2 <= i && s.startsWith("*/*", i-2))) {
	// This must be a nested /*...*/-comment (in the second case, it's
	// a nested comment followed by a "*"), which is legal only inside
	// a //-comment, so we might as well end our search now.
	return true;
      } else if (s.startsWith("/*", i-1)) {
	// We are now looking at either the start of the comment.
	// So, as far as we can tell, the construct may spill over
	// to the next line.
	return false;
      } else if (4 <= i && s.startsWith("<esc>", i-4)) {
	// The begin-comment marker must have been "/*", because if it
	// were "//" we would have found the "</esc>" in the forward loop
	// above.  So, as far as we can tell, the construct may spill over
	// to the next line.
	return false;
      }
    }

    // We have been unable to determine that the construct ends on this line
    return false;
  }
}
