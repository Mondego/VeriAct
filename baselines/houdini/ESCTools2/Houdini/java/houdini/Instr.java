/* Copyright 2000, 2001, Compaq Computer Corporation */

// ESC/Java annotation wizard
// Copyright (c) 1999, Compaq Computer Corporation
// Change history:
//    2 Apr 1999  rustan     Created

package houdini;


abstract class Instr {
  //@ ensures \result != null;
  static Instr make(/*@ non_null */ String instr,
		    /*@ non_null */ FileCollection sources) {
    int k, start = 0;

    k = next(instr, start, ' ');
    String filename = instr.substring(start, k);
    int fileidNowarn = sources.getFileId(filename);
    start = k+1;

    k = next(instr, start, ' ');
    int lineNowarn = toNumber(instr.substring(start, k));
    //@ assume 1 <= lineNowarn;
    start = k+1;

    k = next(instr, start, ' ');
    int colNowarn = toNumber(instr.substring(start, k));
    start = k+1;

    k = next(instr, start, ' ');
    String errorType = instr.substring(start, k);
    start = k+1;

    if (! instr.startsWith("<", start)) {
      error(instr);
    }
    k = next(instr, start+1, '>');
    String dr = instr.substring(start+1, k);
    start = k+1;

    if (start == instr.length()) {
      return new NowarnInstr(fileidNowarn, lineNowarn, colNowarn,
			     errorType, dr);
    } else if (! instr.startsWith(" ", start)) {
      error(instr);
    }
    start++;

    k = next(instr, start, ' ');
    filename = instr.substring(start, k);
    int fileidPragma = sources.getFileId(filename);
    start = k+1;

    k = next(instr, start, ' ');
    String placementStr = instr.substring(start, k);
    int placement = PragmaInstr.stringToPlacement(placementStr);
    start = k+1;

    k = next(instr, start, ' ');
    int linePragma = toNumber(instr.substring(start, k));
    start = k+1;

    k = next(instr, start, ' ');
    int colPragma = toNumber(instr.substring(start, k));
    start = k+1;

    k = next(instr, start, ' ');
    String idPragma = instr.substring(start, k);
    start = k+1;

    if (! instr.startsWith("'", start)) {
      error(instr);
    }
    k = next(instr, start+1, '\'');
    String pragma = instr.substring(start+1, k);
    start = k+1;

    if (start == instr.length()) {
      return new PragmaInstr(fileidNowarn, lineNowarn, colNowarn,
			     errorType, dr,
			     fileidPragma, placement, linePragma, colPragma,
			     idPragma, pragma);
    }

    error("unrecognized instruction, line too long: " +
	  instr.substring(start));
    //@ unreachable;
    return null;
  }

  static int next(/*@ non_null */ String s, int start, char ch) {
    int k = s.indexOf(ch, start);
    if (k == -1) {
      error(s);
    }
    return k;
  }

  //@ ensures 0 <= \result;
  static int toNumber(/*@ non_null */ String s) {
    int k = 0;
    for (int i = 0; i < s.length(); i++) {
      char ch = s.charAt(i);
      if (ch < '0' || '9' < ch) {
	error("expected number, found '" + s + "'");
      }
      k = 10*k + ch - '0';
    }
    return k;
  }

  //@ ensures false;
  static void error(/*@ non_null */ String instr) {
    AnnotationInserter.error("unrecognized instruction: " + instr);
  }

  //@ ensures \result != null ==> \result.fileid < sources.n;
  //@ ensures \result != null ==> \result.next == null;
  abstract WorkItem process(/*@ non_null */ FileCollection sources);

  //@ requires 0 <= fileid && fileid < sources.n;
  //@ requires 1 <= line;
  //@ ensures \result != null ==> \result.fileid < sources.n;
  //@ ensures \result != null ==> \result.next == null;
  WorkItem processNowarn(int fileid, int line, int col,
			 /*@ non_null */ String errorType,
			 /*@ non_null */ String reason,
			 /*@ non_null */ FileCollection sources) {
    FileInfo fi = sources.getFileInfo(fileid);

    if (fi.getFilename().equals("nofile"))
      return null;

    if (!fi.canModify()) {
      AnnotationInserter.error("Wizard work item requires modifying file '" +
			       fi.getFilename() +
			       "', which wizard was instructed not to modify");
    }

    String s = "/*@(" + reason + ") nowarn " + errorType + " */";
    
    // go as far as possible on this line, but not passed any comment
    byte[] data = fi.getData();
    int offset = fi.getLineOffset(line) + col;
    for (; offset < data.length; offset++) {
      char ch = (char)data[offset];
      if (ch == '\n') {
	// Note:  initial space is needed so that first pragma slash doesn't
	// become the second slash of a division operator found at the end of
	// a line.
	s = " " + s;
	break;
      } else if (ch == '/' && offset+1 < data.length) {
	char chch = (char)data[offset+1];
	if (chch == '/' || chch == '*') {
	  s = s + " ";
	  break;
	}
      }
    }
    return new WorkItem(fileid, offset, s);
  }
}
