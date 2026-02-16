/* Copyright 2000, 2001, Compaq Computer Corporation */

// ESC/Java annotation wizard, annotation inserter
// Copyright (c) 1999, Compaq Computer Corporation
// Change history:
//    2 Apr 1999  rustan     Created
//   11 Oct 2000  flanagan   Added ">" and ".", necy for some Houdini annotations

/*
# Output format: one of the following:
#   File line col errorType <description> File placement line col id 'pragma'
#   File line col errorType <reason>
# where
#   File is a filename
#   line is a number (the second "line" may be 0 to indicate a location in
#     a binary file; in that case, the second "col" is also 0)
#   col is a number
#   errorType is an ESC/Java error tag, like Null or Cast
#   description categorizes the suggested annotated entity
#   placement is one of <P, <, <<, <|, >>, >, or . which have the following
#   meanings:
#       <P  place just before type
#       <   place just before type (*)
#       <<  place just before type, or preferably on new previous line (*)
#       <|  place here, or preferably on new previous line
#       >>  place after semi-colon, preferably on new next line
#       >   place after id, preferably on new next line
#       .   place right before id
#    (*) but don't act if a comma is between the type and the current position,
#        or if a comma follows the current position before a semicolon or
#        open parenthesis does.
#   pragma is any string
#   reason is any string, explaining why there was no suggestion
# Note, if the suggestion refers to a whole-file location, the second "line"
# will be 0 (and so will "col").
*/

package houdini;

import java.io.File;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;


public class AnnotationInserter {
  //@ requires \nonnullelements(args);
  public static void main(String[] args) {
    if (args.length == 1 && args[0].equals("-ping")) {
      return;
    }
    if (args.length < 2) {
      error("usage: AnnotationInserter instructionFile [-f <file of sourcefiles>] sourcefiles...");
    }
    File instructionFile = new File(args[0]);
    String instructions = new String(FileInfo.getFileContents(instructionFile),
				     0);
    FileCollection sources = new FileCollection(2*(args.length-1));
    for (int i = 1; i < args.length; i++) {
      if (args[i].equals("-f")) {
	try {
	  i++; // yes, it's the loop counter.
	  //@ assume i < args.length;
	  BufferedReader in = 
	    new BufferedReader(new FileReader(args[i]));
	  String s;
	  while ((s = in.readLine()) != null) {
	    sources.add(s);
	  }
	} catch (IOException e) {
	  System.out.println(e.toString());
	  System.exit(1);
	} 
      } else {
	sources.add(args[i]);
      }
    }

    for (int i = 0; i < instructions.length(); ) {
      iCurrentLine++;
      int k = instructions.indexOf('\n', i);
      if (i == -1) {
	error(args[0] + " does not end with newline");
      }
      String instr = instructions.substring(i, k);
      process(instr, sources);
      i = k+1;
    }

    sources.update();
  }

  static int iCurrentLine = 0;

  //@ ensures false;
  static void error(/*@ non_null */ String s) {
    System.out.println("AnnotationInserter: " + s +
		       " [instruction " + iCurrentLine + "]");
    System.exit(1);
  }

  static void process(/*@ non_null */ String instr,
		      /*@ non_null */ FileCollection sources) {
    Instr ii = Instr.make(instr, sources);
    // System.out.println("@@ " + ii.toString());  System.out.flush();
    WorkItem wi = ii.process(sources);
    if (wi == null) {
      // System.out.println("## null");  System.out.flush();
    } else { 
      // System.out.println("## " + wi.toString());  System.out.flush();
      FileInfo fi = sources.getFileInfo(wi.fileid);
      //@ assume fi.data != null;
      fi.addWorkItem(wi);
    }
  }
}
