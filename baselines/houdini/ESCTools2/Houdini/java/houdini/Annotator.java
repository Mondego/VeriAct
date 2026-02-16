/* Copyright 2000, 2001, Compaq Computer Corporation */

// Copyright (c) 1999, Compaq Computer Corporation
// Authors:  Cormac Flanagan and Rustan Leino.

package houdini;

import javafe.ast.*;
import javafe.tc.*;
import javafe.util.*;


public class Annotator {
  /*@ non_null */ private String placement;
  private int loc;
  //@ invariant loc != Location.NULL;
  /*@ non_null */ private String id;
  /*@ non_null */ private String commentPrefix;
  /*@ non_null */ private String pragmaPrefix;

  //@ requires loc != Location.NULL;
  public Annotator(/*@ non_null */ String placement,
		   int loc,
		   /*@ non_null */ String id,
		   /*@ non_null */ String commentPrefix,
		   /*@ non_null */ String pragmaPrefix) {
    this.placement = placement;
    this.loc = loc;
    this.id = id;
    this.commentPrefix = commentPrefix;
    this.pragmaPrefix = pragmaPrefix;
  }

  public Annotator(/*@ non_null */ RoutineDecl rd,
		   /*@ non_null */ String commentPrefix,
		   /*@ non_null */ String pragmaPrefix) {
    if (rd instanceof ConstructorDecl) {
      this.placement = "<|";
    } else {
      this.placement = "<<";
    }
    this.loc = rd.locId;

    if (! (rd instanceof MethodDecl)) {
      //@ assume rd.parent != null;
      this.id = rd.parent.id.toString();
    } else {
      MethodDecl md = (MethodDecl)rd;
      this.id = md.id.toString();
    }

    this.commentPrefix = commentPrefix;
    this.pragmaPrefix = pragmaPrefix;
  }

  public void put(/*@ non_null */ String pragma) {
    put(pragma, null);
  }

  public void put(/*@ non_null */ String pragma, String comment) {
    String c = commentPrefix;
    if (comment != null && comment.length() != 0) {
      c += ":" + comment;
    }
    outputSuggestion(placement, loc, id,
		     pragmaPrefix + pragma, c, false);
  }

  public void putPermanent(/*@ non_null */ String pragma) {
    outputSuggestion(placement, loc, id,
		     pragmaPrefix + pragma, commentPrefix, true);
  }

  //@ requires loc != Location.NULL;
  private static void outputSuggestion(/*@ non_null */ String placement,
				       int loc,
				       /*@ non_null */ String id,
				       /*@ non_null */ String pragma,
              			       /*@ non_null */ String comment,
				       boolean permanent) {
      System.out.println("nofile 1 1 Null " +
			 (permanent ? "<houdini-permanent:" : "<houdini:") +
			 comment + "> " +
			 Location.toFileName(loc) + ' ' +
			 placement + ' ' + Location.toLineNumber(loc) + ' ' +
			 Location.toColumn(loc) + ' ' + id + ' ' +
			 '\'' + pragma + '\'');
  }
}
