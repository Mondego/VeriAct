/* Copyright 2000, 2001, Compaq Computer Corporation */

// ESC/Java annotation wizard
// Copyright (c) 1999, Compaq Computer Corporation
// Change history:
//    2 Apr 1999  rustan     Created

package houdini;

import java.io.File;


class FileCollection {
  /*@ non_null */ FileInfo[] fi;
  //@ invariant 0 < fi.length;
  //@ invariant \typeof(fi) == \type(FileInfo[]);
  int n;
  //@ invariant 0 <= n && n <= fi.length;
  //@ invariant (\forall int i; 0 <= i && i < n ==> fi[i] != null);
  // @ invariant (\forall int i; 0 <= i && i < n ==> fi[i].id == i);

  //@ requires 0 < initialSize;
  FileCollection(int initialSize) {
    fi = new FileInfo[initialSize];
    n = 0;
  }

  //@ modifies n;
  void add(/*@ non_null */ String filename) {
    File file = new File(filename);
    addW(file, true);
  }

  //@ modifies n;
  //@ ensures 0 <= \result && \result < n;
  private int addW(/*@ non_null */ File file, boolean fReadFile) {
    if (n == fi.length) {
      FileInfo[] nu = new FileInfo[2*n];
      System.arraycopy(fi, 0, nu, 0, n);
      fi = nu;
    }

    int id = n;
    fi[n] = new FileInfo(file, id, fReadFile);
    n++;
    return id;
  }

  //@ modifies n;
  //@ ensures 0 <= \result && \result < n;
  int getFileId(/*@ non_null */ String filename) {
    File file = new File(filename);
    for (int i = 0; i < n; i++) {
      if (file.equals(fi[i].file)) {
	return i;
      }
    }
    return addW(file, false);
  }

  //@ requires 0 <= id && id < n;
  //@ ensures \result != null;
  //@ ensures \result.id == id;
  FileInfo getFileInfo(int id) {
    //@ assume fi[id].id == id;
    return fi[id];
  }

  void update() {
    for (int i = 0; i < n; i++) {
      fi[i].update();
    }
  }
}
