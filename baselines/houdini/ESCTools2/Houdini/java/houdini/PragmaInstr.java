/* Copyright 2000, 2001, Compaq Computer Corporation */

// ESC/Java annotation wizard
// Copyright (c) 1999, Compaq Computer Corporation
// Change history:
//    2 Apr 1999  rustan     Created

package houdini;


class PragmaInstr extends Instr {
  int fileidNowarn;
  //@ invariant 0 <= fileidNowarn;
  int lineNowarn;
  //@ invariant 1 <= lineNowarn;
  int colNowarn;
  //@ invariant 0 <= colNowarn;
  /*@ non_null */ String errorType;
  
  /*@ non_null */ String descriptionPragma;
  int fileidPragma;
  //@ invariant 0 <= fileidPragma;
  
  int placement;
  static public final int BEFORE = 0;
  static public final int BEFORE_PARAM = 1;
  static public final int PREVLINE = 2;
  static public final int PREVLINE_CONSTR = 3;
  static public final int NEXTLINE = 4;
  static public final int RIGHTAFTER = 5;
  static public final int RIGHTBEFORE = 6;
  //@ invariant BEFORE <= placement && placement <= RIGHTBEFORE;

  int linePragma;
  //@ invariant 0 <= linePragma;
  int colPragma;
  //@ invariant 0 <= colPragma;
  /*@ non_null */ String idPragma;
  /*@ non_null */ String pragma;


  //@ requires 0 <= fileidNowarn;
  //@ requires 1 <= lineNowarn;
  //@ requires 0 <= colNowarn;
  //@ requires 0 <= fileidPragma;
  //@ requires BEFORE <= placement && placement <= RIGHTBEFORE;
  //@ requires 0 <= linePragma;
  //@ requires 0 <= colPragma;
  PragmaInstr(int fileidNowarn, int lineNowarn, int colNowarn,
	      /*@ non_null */ String errorType,
	      /*@ non_null */ String descriptionPragma, int fileidPragma,
	      int placement, int linePragma, int colPragma,
	      /*@ non_null */ String idPragma, /*@ non_null */ String pragma) {
    this.fileidNowarn = fileidNowarn;
    this.lineNowarn = lineNowarn;
    this.colNowarn = colNowarn;
    this.errorType = errorType;
    this.descriptionPragma = descriptionPragma;
    this.fileidPragma = fileidPragma;
    this.placement = placement;
    this.linePragma = linePragma;
    this.colPragma = colPragma;
    this.idPragma = idPragma;
    this.pragma = pragma;
  }

  public String toString() {
    return "File " + fileidNowarn + ", line " + lineNowarn + "," + colNowarn +
      " (" + errorType + ") <" + descriptionPragma +">; file " +
      fileidPragma + ", " + placementToString(placement) + " " +
      linePragma + "," + colPragma + " " + idPragma + " '" +
      pragma + "'";
  }

  //@ ensures BEFORE <= \result && \result <= NEXTLINE;
  public static int stringToPlacement(/*@ non_null */ String str) {
    if (str.equals("<")) {
      return BEFORE;
    } else if (str.equals("<P")) {
      return BEFORE_PARAM;
    } else if (str.equals("<<")) {
      return PREVLINE;
    } else if (str.equals("<|")) {
      return PREVLINE_CONSTR;
    } else if (str.equals(">>")) {
      return NEXTLINE;
    } else if (str.equals(">")) {
      return RIGHTAFTER;
    } else if (str.equals(".")) {
      return RIGHTBEFORE;
    } else {
      error("unknown placement code: " + str);
      //@ unreachable;
      return 0;
    }
  }

  //@ requires BEFORE <= placement && placement <= NEXTLINE;
  public static String placementToString(int placement) {
    switch (placement) {
      case BEFORE:  return "<";
      case BEFORE_PARAM:  return "<P";
      case PREVLINE:  return "<<";
      case PREVLINE_CONSTR:  return "<|";
      case NEXTLINE:  return ">>";
      case RIGHTAFTER:  return ">";
      case RIGHTBEFORE:  return ".";
      default:
	//@ unreachable;
	return null;
    }
  }

  WorkItem process(FileCollection sources) {
    //@ assume fileidPragma < sources.n;
    //@ assume fileidNowarn < sources.n;
    FileInfo fi = sources.getFileInfo(fileidPragma);
    if (!fi.canModify()) {
      return processNowarn(fileidNowarn, lineNowarn, colNowarn, errorType,
			   "cannot modify " + fi.getFilename() +
			   " to insert '" + pragma + "' " +
			   placementToString(placement) + " " +
			   linePragma + "," + colPragma + " for '" +
			   idPragma + "'",
			   sources);
    }
    /* If the file can be modified, it isn't a binary file, so the
     * "linePragma" denotes an actual line (as opposed to denoting
     * "binary file").
     */
    //@ assume linePragma != 0;

    byte[] data = fi.getData();
    int offsetStartLine = fi.getLineOffset(linePragma);
    int position = offsetStartLine + colPragma;
    //@ assume position < data.length;
    if (!startsWith(data, position, idPragma)) {
      fileChanged("expected '" + idPragma + "' at " +
		  linePragma + "," + colPragma + " in " + fi.getFilename());
    }

    String pr = "/*@(" + descriptionPragma + ") " + pragma + " */";

    switch (placement) {
      case BEFORE:
      case BEFORE_PARAM:
      case PREVLINE:
	{
	  if (placement != BEFORE_PARAM &&
	      isMultipleDeclAfter(data, position, fi)) {
	    // Resort to nowarn pragma
	    return processNowarn(fileidNowarn, lineNowarn, colNowarn,
				 errorType,
				 "cannot decorate '" + idPragma + "' " +
				 placementToString(placement) + " " +
				 linePragma + "," + colPragma + " in " +
				 fi.getFilename() + " with '" +
				 pragma + "' because it occurs in a " +
				 "declaration of several variables",
				 sources);
	  }

	  int o = position;
	  while (true) {
	    o = backOffsetOfNonComment(data, o, true);
	    if (o < 0) {
	      fileChanged("expected type before '" + idPragma +
			  "' at " + linePragma + "," + colPragma +
			  " in " + fi.getFilename());
	    }
	    char ch = (char)data[o];   
	    if (!Character.isWhitespace(ch)) {
	      if (ch == '[' || ch == ']') {
		// skip
	      } else if (ch == ',') {
		// Resort to nowarn pragma
		return processNowarn(fileidNowarn, lineNowarn, colNowarn,
				     errorType,
				     "cannot decorate '" + idPragma + "' " +
				     placementToString(placement) + " " +
				     linePragma + "," + colPragma + " in " +
				     fi.getFilename() + " with '" +
				     pragma + "' because it occurs in a " +
				     "declaration of several variables",
				     sources);
	      } else {
		break;
	      }
	    }
	  }

	  // exit of loop leaves o on the last character of the id.
	  o++;
	  
	  //@ loop_invariant 0 <= o && o < position;
	  while (true) {
	    // at this point, o-1 is the end of the identifier
	    o = skipBackToStartOfId(data, o, fi);
	    int possibleDot = o;
	    do {
		possibleDot = backOffsetOfNonComment(data, possibleDot, true);
	    } while (Character.isWhitespace((char)data[possibleDot]));
	    if (possibleDot == -1 || (char)data[possibleDot] != '.') {
		break;
	    }
	    o = possibleDot;
	  }
	  if (placement != PREVLINE) {
	    return new WorkItem(fileidPragma, o, pr + " ");
	  }

	  int m = backOffsetOf(data, o, '\n', false) + 1;
	  String prefix = substring(data, m, o);
	  //@ assume prefix != null;
	  if (isModifiersOnly(prefix)) {
	    prefix = getWhitePrefix(prefix);
	    return new WorkItem(fileidPragma, m, prefix + pr + "\n");
	  } else {
	    return new WorkItem(fileidPragma, o, pr + " ");
	  }
	}

      case PREVLINE_CONSTR:
	{
	  String prefix = substring(data, offsetStartLine, position);
	  //@ assume prefix != null;
	  if (isModifiersOnly(prefix)) {
	    prefix = getWhitePrefix(prefix);
	    return new WorkItem(fileidPragma, offsetStartLine,
				prefix + pr + "\n");
	  } else {
	    return new WorkItem(fileidPragma, position, pr + " ");
	  }
	}

      case NEXTLINE:
	{
	  String prefix = getWhitePrefix(substring(data, //@ nowarn NonNull;
						   offsetStartLine, position));
	  int semi = offsetOf(data, position, ';', true, fi) + 1;
	  int eol = offsetOf(data, semi, '\n', false, fi) + 1;
	  if (isWhite(substring(data, semi, eol), false)) { //@ nowarn NonNull;
	    return new WorkItem(fileidPragma, eol, prefix + pr + "\n");
	  } else {
	    return new WorkItem(fileidPragma, semi, " " + pr + " ");
	  }
	}

      case RIGHTAFTER:
	{
	    return new WorkItem(fileidPragma, position+1, "\n"+pr);
	}

      case RIGHTBEFORE:
	{
	    return new WorkItem(fileidPragma, position, "\n"+pr);
	}

      default:
	//@ unreachable;
	return null;  // dummy return
    }
  }

  /** Returns the offset into <code>data</code> of the "first" occurrence
    * of character <code>ch</code> not among the first <code>start</code>
    * bytes of <code>data</code>.  If <code>honorComments</code> is
    * <code>false</code>, "first" means "first"; otherwise, "first" means
    * "first not occurring in a comment".<p>
    *
    * If the requested character does exist, reports an error and does not
    * terminate.  It is to be able to print an appropriate filename that
    * the <code>fi</code> parameter exists.<p>
    *
    * Caveats:  Does not work properly for String literals that contain
    * comment begin or end tokens.
    **/

  //@ requires 0 <= start && start <= data.length;
  //@ ensures start <= \result && \result < data.length;
  static int offsetOf(/*@ non_null */ byte[] data, int start, char ch,
		      boolean honorComments, /*@ non_null */ FileInfo fi) {
    int i = start;
    while (true) {
      i = offsetOfNonComment(data, i, honorComments, fi);
      if (i == data.length) {
	fileChanged("expected '" + ch + "' after offset " + start +
		    " in file " + fi.getFilename());
      }
      if ((char)data[i] == ch) {
	return i;
      }
      i++;
    }
  }

  /** Returns the offset into <code>data</code> of the "first" character
    * not among the first <code>start</code> bytes of <code>data</code>.
    * If <code>honorComments</code> is <code>false</code>, "first" means
    * "first"; otherwise, "first" means "first not occurring in a comment".<p>
    *
    * If <code>honorComments</code> is <code>true</code> and <code>data</code>
    * starts a comment at index <code>start</code> that doesn't terminate
    * before the end of <code>data</code>, the an error is reported and
    * this method does not terminate.  It is to be able to print an
    * appropriate filename that the <code>fi</code> parameter exists.<p>
    *
    * Caveats:  Does not work properly for String literals that contain
    * comment begin or end tokens.
    **/

  //@ requires 0 <= start && start <= data.length;
  //@ ensures start <= \result && \result <= data.length;
  static int offsetOfNonComment(/*@ non_null */ byte[] data, int start,
				boolean honorComments,
				/*@ non_null */ FileInfo fi) {
    boolean inTraditionalComment = false;
    boolean inSingleLineComment = false;
    
    //@ loop_invariant !inTraditionalComment || !inSingleLineComment;
    for (int i = start; i < data.length; i++) {

      char cur = (char)data[i];
      if (inSingleLineComment) {
	if (cur == '\n') {
	  inSingleLineComment = false;
	}
      } else if (inTraditionalComment) {
	if (cur == '*' && i+1 < data.length && (char)data[i+1] == '/') {
	  i++;
	  inTraditionalComment = false;
	}
      } else if (honorComments && cur == '/' &&
		 i+1 < data.length && (char)data[i+1] == '/') {
	i++;
	inSingleLineComment = true;
      } else if (honorComments && cur == '/' &&
		 i+1 < data.length && (char)data[i+1] == '*') {
	i++;
	inTraditionalComment = true;
      } else {
	return i;
      }
    }

    if (inSingleLineComment || inTraditionalComment) {
      fileChanged("expected end-of-comment to terminate the comment that " +
		  "starts at offset " + start + " in file " +
		  fi.getFilename());
    }
    return data.length;
  }

  /** Returns the offset into <code>data</code> of the "last" occurrence
    * of character <code>ch</code> among the first <code>start</code>
    * bytes of <code>data</code>.  If <code>honorComments</code> is
    * <code>false</code>, "last" means "last"; otherwise, "last" means
    * "last not occurring in a comment".<p>
    *
    * If the requested character does exist, -1 is returned.<p>
    *
    * Caveats:  Does not work correctly for traditional comments that contain
    * occurrences of slash-star, or for single-line comments that contain
    * occurrences of star-slash.  Also, does not work properly for String
    * literals that contain comment begin or end tokens.
    **/

  //@ requires 0 <= start && start <= data.length;
  //@ ensures -1 <= \result && \result < start;
  static int backOffsetOf(/*@ non_null */ byte[] data, int start, char ch,
			  boolean honorComments) {
    int j = start;
    while (true) {
      j = backOffsetOfNonComment(data, j, honorComments);
      if (j < 0) {
	return -1;
      }
      if ((char)data[j] == ch) {
	return j;
      }
    }
  }

  /** Returns the offset into <code>data</code> of the "last" character
    * among the first <code>start</code> bytes of <code>data</code>.
    * If <code>honorComments</code> is <code>false</code>, "last" means
    * "last"; otherwise, "last" means "last not occurring in a comment".<p>
    *
    * If the requested character does exist, -1 is returned.<p>
    *
    * Caveats:  Does not work correctly for traditional comments that contain
    * occurrences of slash-star, or for single-line comments that contain
    * occurrences of star-slash.  Also, does not work properly for String
    * literals that contain comment begin or end tokens.
    **/

  //@ requires 0 <= start && start <= data.length;
  //@ ensures -1 <= \result && \result < start;
  //@ ensures \result < start;
  static int backOffsetOfNonComment(/*@ non_null */ byte[] data, int start,
				    boolean honorComments) {
    boolean inTraditionalComment = false;
    int i = start;
    while (true) {
      i--;
      if (i < 0) {
	// not found
	return -1;
      }

      char cur = (char)data[i];
      if (inTraditionalComment) {
	if (cur == '*' && 0 <= i-1 && (char)data[i-1] == '/') {
	  i--;
	  inTraditionalComment = false;
	}
      } else if (honorComments && cur == '/' &&
		 0 <= i-1 && (char)data[i-1] == '*') {
	i--;
	inTraditionalComment = true;
      } else if (honorComments && cur == '\n') {
        // check if this looks like a single-line comment
        int j = i;
        while (true) {
          j--;
	  if (j < 0) {
	    // not a single-line comment
	    return i;
	  }
	  char chx = (char)data[j];
	  if (chx == '\n') {
	    // not a single-line comment
	    return i;
	  } else if (chx == '/' && 0 <= j-1 && (char)data[j-1] == '/') {
	    // a single-line comment
	    i = j-1;
	    //@ assert i < j;
	    break;
	  } else if (chx == '/' && 0 <= j-1 && (char)data[j-1] == '*') {
	    // presumed not to be a single-line comment (see Caveats above)
	    return i;
	  }
	}
      } else {
	return i;
      }
    }
  }

  /** Returns the sequences of bytes of <code>data</code> from
    * <code>start</code> to less than <code>end</code> as a character
    * string.
    **/

  //@ requires 0 <= start;
  //@ requires start <= end;
  //@ requires end <= data.length;
  static String substring(/*@ non_null */ byte[] data, int start, int end) {
    char[] a = new char[end-start];
    for (int i = start; i < end; i++) {
      a[i-start] = (char)data[i];
      // The line above used to be the erroneous:
      //   a[i] = (char)data[i];
      // ESC/Java detected this error.
    }
    return new String(a);
  }

  /** Returns whether or not the identifier presumed at offset
    * <code>start</code> in <code>data</code> is part of a variable
    * declaration that declares another identifier after this one.<p>
    *
    * Caveats: This method does not properly handle String or character
    * literals.
    **/
  
  static boolean isMultipleDeclAfter(/*@ non_null */ byte[] data,
				     int start,
				     /*@ non_null */ FileInfo fi) {
    boolean inInitializingExpr = false;
    int parenDepth = 0;
    int j = start;
    //@ loop_invariant 0 <= parenDepth;
    //@ loop_invariant parenDepth != 0 ==> inInitializingExpr;
    while (true) {
      
      j = offsetOfNonComment(data, j, true, fi);
      if (j == data.length) {
	// who knows what the identifier is part of, but it isn't a
	// multiple-variable declaration
	return false;
      }
      char ch = (char)data[j];
      if (ch == ';') {
	// "parenDepth" is presumed 0
	return false;
      } else if (ch == ',') {
	if (parenDepth == 0) {
	  return true;
	}
      } else if (!inInitializingExpr && ch == '=') {
	inInitializingExpr = true;
      } else if (ch == '(' || ch == '{') {
	if (inInitializingExpr) {
	  parenDepth++;
	} else {
	  // this is probably the open round paren of a method declaration
	  return false;
	}
      } else if (ch == ')' || ch == '}') {
	// "parenDepth" ought to be positive
	if (0 < parenDepth) {
	  parenDepth--;
	}
      }
      j++;
    }
  }
  
  /** Returns as a string the prefix of <code>s</code> that contains
    * only spaces and tabs.
    **/

  //@ ensures \result != null;
  static String getWhitePrefix(/*@ non_null */ String s) {
    int i = 0;
    while (i < s.length()) {
      char ch = s.charAt(i);
      if (ch != ' ' && ch != '\t' && ch != '\r') {
	break;
      }
      i++;
    }
    return s.substring(0, i);
  }

  /** Returns whether or not the first <code>s.length()</code> bytes
    * of <code>data</code> starting at <code>start</code> correspond
    * to the characters in <code>s</code>.
    **/

  //@ requires 0 <= start && start <= data.length;
  static boolean startsWith(/*@ non_null */ byte[] data, int start,
			    /*@ non_null */ String s) {
    int end = start + s.length();
    if (data.length < end) {
      return false;
    }
    for (int i = start; i < end; i++) {
      char ch = (char)data[i];
      if (ch != s.charAt(i-start)) {
	return false;
      }
    }
    return true;
  }

  /** Returns whether each character in <code>s</code> is either
    * a space, a tab, a newline, or is part of a comment that starts
    * and ends in <code>s</code>.  If <code>considerPragmaWhite</code>
    * is <code>true</code>, a pragma-containing comment is considered
    * a comment; otherwise, an encountered pragma-containing comment
    * will cause a result value of <code>false</code>.
    **/

  static boolean isWhite(/*@ non_null */ String s,
			 boolean considerPragmaWhite) {
    return indexOfNonWhite(s, 0, considerPragmaWhite) == s.length();
  }

  /** Returns the index into <code>s</code> of the first non-white
    * character occurring after the first <code>start</code> characters,
    * that is, the first character that is not among the first
    * <code>start</code> characters, is not a space, is not a tab, is
    * not a newline, and is not part of a "eligible" comment
    * that starts and ends in <code>s</code>.  A comment is "eligible"
    * if <code>considerPragmasWhite</code> is <code>true</code> or if
    * the comment does not begin with the ESC/Java pragma character
    * "@".<p>
    * 
    * If all characters of <code>s</code> are white, then
    * <code>s.length()</code> is returned.
    **/

  //@ requires 0 <= start && start <= s.length();
  /*@ ensures 0 <= \result && \result <= s.length(); */ //@ nowarn Post;
  static int indexOfNonWhite(/*@ non_null */ String s, int start,
			     boolean considerPragmasWhite) {
    boolean inTraditionalComment = false;
    boolean inSingleLineComment = false;

    /*--- readable_if inTraditionalComment || inSingleLineComment; ---*/
    int startOfComment = 0;

    //@ loop_invariant !inTraditionalComment || !inSingleLineComment;
    for (int i = start; i < s.length(); i++) {

      char ch = s.charAt(i);
      if (ch == ' ' || ch == '\t' || ch == '\r') {
	// skip
      } else if (ch == '\n') {
	if (inSingleLineComment) {
	  inSingleLineComment = false;
	}
      } else if (ch == '*' && inTraditionalComment &&
		 i+1 < s.length() && s.charAt(i+1) == '/') {
	i++;
	inTraditionalComment = false;
      } else if (inTraditionalComment || inSingleLineComment) {
	// skip
      } else if (ch == '/' && i+1 < s.length() && s.charAt(i+1) == '*' &&
		  (considerPragmasWhite ||
		   !(i+2 < s.length() && s.charAt(i+2) == '@'))) {
	startOfComment = i;
	i++;
	inTraditionalComment = true;
      } else if (ch == '/' && i+1 < s.length() && s.charAt(i+1) == '/' &&
		  (considerPragmasWhite ||
		   !(i+2 < s.length() && s.charAt(i+2) == '@'))) {
	startOfComment = i;
	i++;
	inSingleLineComment = true;
      } else {
	return i;
      }
    }
    if (inTraditionalComment || inSingleLineComment) {
      // comment in not completely contained within "s", so return index
      // of first character in comment.
      return startOfComment;
    } else {
      return s.length();
    }
  }

  /** Returns whether each character in <code>s</code> is a white space
    * character (space, tab, newline), is part of a non-pragma comment
    * (contained entirely in <code>s</code>), or is part of a Java
    * modifier keyword (entirely contained in <code>s</code>).
    *
    * Note, this method does not check whether or not modifier keywords are
    * separated by white space or comments (so for example, it will return
    * <code>true</code> given the string "publicabstract"), but clients
    * don't care.
    **/

  static boolean isModifiersOnly(/*@ non_null */ String s) {
    int i = 0;
    while (true) {
      i = indexOfNonWhite(s, i, false);
      int j = i;
      for (; j < s.length(); j++) {
	char ch = s.charAt(j);
	if (ch < 'a' || 'z' < ch) {
	  break;
	}
      }
      if (i == j) {
	// loop did not scan any letters
	return i == s.length();
      }
      String n = s.substring(i, j);
      if (! n.equals("public") &&
	  ! n.equals("private") &&
	  ! n.equals("protected") &&
	  ! n.equals("static") &&
	  ! n.equals("final") &&
	  ! n.equals("synchronized") &&
	  ! n.equals("volatile") &&
	  ! n.equals("transient") &&
	  ! n.equals("native") &&
          ! n.equals("abstract")) {
	return false;
      }
      i = j;
    }
  }
  
  /** Scans backwards in <code>data</code> from offset less than
    * <code>start</code>, expecting to find an identifier, possibly
    * after finding some white space and comments.  Returns the offset
    * of the first character of that identifier.<p>
    *
    * If <code>data</code> has an unexpected form (in particular, if
    * no such identifier is found), then this will print an error message
    * and won't return.  It is to be able to print an appropriate filename
    * that the <code>fi</code> parameter exists.
    **/

  //@ requires 0 <= start && start <= data.length;
  //@ ensures 0 <= \result && \result < start;
  static int skipBackToStartOfId(/*@ non_null */ byte[] data, int start,
				 /*@ non_null */ FileInfo fi) {
    int j = start;
    while (true) {
      j = backOffsetOfNonComment(data, j, true);
      if (j < 0) {
	fileChanged("expected identifier before offset " + start +
		    " in file " + fi.getFilename());
      }
      char ch = (char)data[j];
      if (!Character.isWhitespace(ch)) {
	// This character is presumed to be the last character of an
	// identifier.
	if (!(('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z') ||
	      ('0' <= ch && ch <= '9') || ch == '_')) {
	  fileChanged("expected identifier before offset " + start +
		      " in file " + fi.getFilename());
	}
	// Step backwards until the beginning of the identifier.
	while (('a' <= ch && ch <= 'z') || ('A' <= ch && ch <= 'Z') ||
	       ('0' <= ch && ch <= '9') || ch == '_') {
	  j--;
	  if (j < 0) {
	    break;
	  }
	  ch = (char)data[j];
	}
	return j+1;
      }
    }
  }

  //@ ensures false;
  static void fileChanged(/*@ non_null */ String msg) {
      if(msg != null) throw new RuntimeException(msg);
    AnnotationInserter.error(msg + "; perhaps file was changed during " +
			     "run of Wizard?");
  }
}
