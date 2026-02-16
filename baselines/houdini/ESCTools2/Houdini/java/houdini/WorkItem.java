/* Copyright 2000, 2001, Compaq Computer Corporation */

// ESC/Java annotation wizard
// Copyright (c) 1999, Compaq Computer Corporation
// Change history:
//    2 Apr 1999  rustan     Created

package houdini;


class WorkItem {
  int fileid;
  //@ invariant 0 <= fileid;
  int offset;
  /*@ non_null */ String s;
  WorkItem next = null;  // for linking inside FileInfo objects

  //@ requires 0 <= fileid;
  //@ ensures this.fileid == fileid;
  //@ ensures this.offset == offset;
  //@ ensures this.s == s;
  //@ ensures this.next == null;
  WorkItem(int fileid, int offset, /*@ non_null */ String s) {
    this.fileid = fileid;
    this.offset = offset;
    this.s = s;
  }

  public String toString() {
    return "File " + fileid + ", offset " + offset + ": '" + s + "'";
  }
}
