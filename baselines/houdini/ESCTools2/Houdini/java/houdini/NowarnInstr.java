/* Copyright 2000, 2001, Compaq Computer Corporation */

// ESC/Java annotation wizard
// Copyright (c) 1999, Compaq Computer Corporation
// Change history:
//    2 Apr 1999  rustan     Created

package houdini;


class NowarnInstr extends Instr {
  int fileid;
  //@ invariant 0 <= fileid;
  int line;
  //@ invariant 1 <= line;
  int col;
  //@ invariant 0 <= col;
  /*@ non_null */ String errorType;
  /*@ non_null */ String reason;

  //@ requires 0 <= fileid;
  //@ requires 1 <= line;
  //@ requires 0 <= col;
  NowarnInstr(int fileid, int line, int col,
	      /*@ non_null */ String errorType,
	      /*@ non_null */ String reason) {
    this.fileid = fileid;
    this.line = line;
    this.col = col;
    this.errorType = errorType;
    this.reason = reason;
  }
  
  public String toString() {
    return "File " + fileid + ", line " + line + "," + col +
      " (" + errorType + ") <" + reason +">";
  }

  WorkItem process(FileCollection sources) {
    //@ assume fileid < sources.n;
    return processNowarn(fileid, line, col, errorType, reason, sources);
  }
}
