/* Copyright 2000, 2001, Compaq Computer Corporation */

// ESC/Java annotation wizard
// Copyright (c) 1999, Compaq Computer Corporation
// Change history:
//    2 Apr 1999  rustan     Created

package houdini;

import java.io.*;


class FileInfo {
  /*@ non_null */ File file;
  int id;
  byte[] data;

  FileInfo(/*@ non_null */ File file, int id, boolean fReadFile) {
    this.file = file;
    this.id = id;
    if (fReadFile) {
      this.data = getFileContents(file);
    }
  }

  //@ ensures \result != null;
  static byte[] getFileContents(/*@ non_null */ File file) {
    byte[] b = new byte[8192];
    int n = 0;
    try {
      FileInputStream in = new FileInputStream(file);
      // read file into a buffer
      while (true) {
	int r = in.read(b, n, b.length - n);
	if (r == -1)
	  break;
	n += r;
	if (n == b.length) {
	  byte[] a = b;
	  b = new byte[2 * a.length];
	  System.arraycopy(a, 0, b, 0, n);
	}
      }
      // close file
      in.close();
    } catch (FileNotFoundException e) {
      AnnotationInserter.error("file '" + file + "'not found");
    } catch (IOException e) {
      AnnotationInserter.error("some IOException occurred");
    }
    byte[] a = b;
    b = new byte[n];
    System.arraycopy(a, 0, b, 0, n);
    return b;
  }

  //@ ensures \result == data;
  byte[] getData() {
    return data;
  }

  //@ ensures \result != null;
  String getFilename() {
    return file.toString();
  }

  /** Returns offset into file of first character on line <code>line</code>,
    * where <code>line</code> is 1-based and denotes an existing line.
    **/

  //@ requires data != null;
  //@ requires 1 <= line;
  //@ ensures 0 <= \result;
  int getLineOffset(int line) {
    int cur = 1;
    int offset = 0;
    while (cur < line) {
      // Since "line" is presumed to be an existing line, "offset" is presumed
      // to be a proper index into "data"
      //@ assume offset < data.length;
      while ((char)data[offset] != '\n') {
	offset++;
      }
      offset++;  // skip the '\n', too
      cur++;
    }
    return offset;
  }

  //@ ensures \result ==> data != null;
  boolean canModify() {
    return data != null;
  }

  WorkItem wiList = null;
  //@ invariant wiList != null ==> data != null;
  
  //@ requires wi.fileid == this.id;
  //@ requires wi.next == null;
  //@ requires this.data != null;
  void addWorkItem(/*@ non_null */ WorkItem wi) {
    WorkItem prev = null;
    WorkItem cur = wiList;
    while (cur != null && cur.offset <= wi.offset) {
      if (cur.offset == wi.offset && cur.s.equals(wi.s)) {
	// duplicate item; don't insert it again
	return;
      }
      prev = cur;
      cur = cur.next;
    }
    wi.next = cur;
    if (prev == null) {
      wiList = wi;
    } else {
      prev.next = wi;
    }
  }

  void update() {
    if (wiList == null) {
      return;
    }
    //@ assert data != null;

    String outName = file.toString() + ".wizardUpdate.tmp";
    FileOutputStream out = null;
    try {
      out = new FileOutputStream(outName);
    } catch (SecurityException se) {
      AnnotationInserter.error("SecurityException raised when trying to " +
			       "open '" + outName + "' for writing");
    } catch (FileNotFoundException fnfe) {
      AnnotationInserter.error("FileNotFoundException raised when trying to " +
			       "open '" + outName + "' for writing");
    } catch (IOException ioe) {
      AnnotationInserter.error("IOException raised when trying to " +
			       "open '" + outName + "' for writing");
    }

    try {
      int n = 0;
      WorkItem wi = wiList;
      while (wi != null || n < data.length) {
	if (wi == null) {
	  out.write(data, n, data.length - n);
	  n = data.length;
	} else if (n < wi.offset) {
	  out.write(data, n, wi.offset - n);
	  n = wi.offset;
	} else {
	  for (int i = 0; i < wi.s.length(); i++) {
	    byte by = (byte)wi.s.charAt(i);
	    out.write(by);
	  }
	  wi = wi.next;
	}
      }
      out.close();
    } catch (IOException ioe) {
      AnnotationInserter.error("IOException raised when trying to " +
			       "operate on '" + outName + "'");
    }
  }
}
