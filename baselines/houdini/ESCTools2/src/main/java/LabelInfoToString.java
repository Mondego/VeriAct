/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.translate;

import java.io.*;
import java.util.Enumeration;
import javafe.ast.*;
import javafe.util.*;
import escjava.ast.*;
import escjava.ast.TagConstants;


public class LabelInfoToString {

  /** set of <code>String</code>, each one of which has the format
    * filename:line:col and has been interned.
    **/

  private static Set annotationLocations = new Set();
  private static Set theMark = new Set();

  public static void reset() {
    annotationLocations.clear();
    theMark.clear();
  }
  
  public static void mark() {
    theMark = annotationLocations;
    annotationLocations = new Set();
  }
  
  public static void resetToMark() {
    annotationLocations = new Set(theMark.elements());
  }
  
  public static String get() {
    StringBuffer sb = new StringBuffer();
    for (Enumeration annotations = annotationLocations.elements();
	 annotations.hasMoreElements(); ) {
      String location = (String)annotations.nextElement();
      sb.append(' ');
      sb.append(location);
    }
    return sb.toString();
  }
  
  public static void recordAnnotationAssumption(int locPragmaDecl) {
    if (escjava.Main.options().printAssumers && locPragmaDecl != Location.NULL) {
      String location = Location.toFileName(locPragmaDecl) + ':' +
	                Location.toLineNumber(locPragmaDecl) + ':' +
	                Location.toColumn(locPragmaDecl);
      annotationLocations.add(location.intern());
    }
  }
}
