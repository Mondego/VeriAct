/*
 * Copyright (C) 2000-2001 Iowa State University
 * 
 * This file is part of mjc, the MultiJava Compiler, and the JML Project.
 * 
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 2 of the License, or (at your option) any later
 * version.
 * 
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
 * FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
 * details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * this program; if not, write to the Free Software Foundation, Inc., 59 Temple
 * Place, Suite 330, Boston, MA 02111-1307 USA
 * 
 * $Id: Diff.java 71731 2008-06-06 14:35:43Z dcochran $
 */

package junitutils;

import java.util.StringTokenizer;
import java.util.ArrayList;

/**
 * Class for calculating a (somewhat) detailed comparison of two strings.
 * 
 * @author Curtis Clifton
 * @version $Revision: 71731 $
 */

public class Diff {

  //---------------------------------------------------------------------
  // CONSTRUCTORS
  //---------------------------------------------------------------------
  /**
   * Calculate a difference between the given strings.
   * 
   * @param oldTextLabel a label for the <code>oldText</code> parameter
   * @param oldText a value to be compared
   * @param newTextLabel a label for the <code>newText</code> parameter
   * @param newText a value to be compared
   */
  public Diff(/*@ non_null */ String oldTextLabel, 
	      /*@ non_null */ String oldText, 
	      /*@ non_null */ String newTextLabel,
	      /*@ non_null */ String newText) 
  {
    this.oldText = oldText;
    this.newText = newText;
    calculate(oldTextLabel, newTextLabel);
  }

  //---------------------------------------------------------------------
  // OPERATIONS
  //---------------------------------------------------------------------

  /**
   * Sets the values of _areDifferent and differences according to a comparison
   * between oldText and newText
   * 
   * @param oldTextLabel a label for the <code>oldText</code> parameter
   * @param newTextLabel a label for the <code>newText</code> parameter
   */
  //@ ensures _areDifferent == areDifferent;
  private /*@ helper */ void calculate(/*@ non_null */ String oldTextLabel, 
				       /*@ non_null */ String newTextLabel) {
    // Accumulate the diff in differencesSB
    StringBuffer differencesSB = new StringBuffer(newText.length());

    calculateDiffs(differencesSB);
    // Ignore local differences in the system class path for JAR files
    if (differencesSB.length() > 0 && !differencesSB.toString().contains(".jar")) {
      // Some diffs accumulated, so the strings are different
      _areDifferent = true;

      // Prepend a key to the results.
      differences = NEWLINE + OLD_CH + oldTextLabel + NEWLINE + NEW_CH
          + newTextLabel + NEWLINE + NEWLINE + differencesSB.toString();
    } else {
      // No diffs accumulated, so the strings must be the same or similar
      _areDifferent = false;
      differences = "";
    }
  }

  private /*@ helper */ void calculateDiffs(/*@ non_null */ StringBuffer differencesSB) {
    String[] oldTextLines = splitByLine(oldText);
    String[] newTextLines = splitByLine(newText);

    // match by lines
    int oPos = 0;
    int nPos = 0;
    int lastOldMatch = -1;
    int lastNewMatch = -1;
    if (oldTextLines.length > 0 && newTextLines.length > 0)
     while (oPos < oldTextLines.length || nPos < newTextLines.length) {
      // this is a pretty dumb algorithm that doesn't handle
      // things like the insertion of a single new line in one
      // string or grouping
      if (oPos >= oldTextLines.length) oPos = oldTextLines.length-1;
      if (nPos >= newTextLines.length) nPos = newTextLines.length-1;
      boolean matched = false;
      for (int i = lastOldMatch+1; i<=oPos; ++i) {
        if (oldTextLines[i].equals(newTextLines[nPos])) {
          // Got a match
          for (int j=lastOldMatch+1; j<i; ++j)
            differencesSB.append((j+1) + OLD_CH + oldTextLines[j] + NEWLINE);
          for (int j=lastNewMatch+1; j<nPos; ++j)
            differencesSB.append((j+1) + NEW_CH + newTextLines[j] + NEWLINE);
          lastOldMatch = i;
          lastNewMatch = nPos;
          oPos = i+1;
          nPos++;
          matched = true;
          break;
        }
      }
      if (matched) continue;
      for (int i = lastNewMatch+1; i<=nPos; ++i) {
        if (newTextLines[i].equals(oldTextLines[oPos])) {
          // Got a match
          for (int j=lastOldMatch+1; j<oPos; ++j)
            differencesSB.append((j+1) + OLD_CH + oldTextLines[j] + NEWLINE);
          for (int j=lastNewMatch+1; j<i; ++j)
            differencesSB.append((j+1) + NEW_CH + newTextLines[j] + NEWLINE);
          lastOldMatch = oPos;
          lastNewMatch = i;
          oPos++;
          nPos = i + 1;
          matched = true;
          break;        }
      }
      if (matched) continue;
      oPos++;
      nPos++;
    }
    // If we reached the end of one array before the other, then this
    // will print the remainders.
    for (oPos=lastOldMatch+1; oPos < oldTextLines.length; oPos++) {
      differencesSB.append((oPos + 1) + OLD_CH + oldTextLines[oPos] + NEWLINE);
    } // end of for ()
    for (nPos=lastNewMatch+1; nPos < newTextLines.length; nPos++) {
      differencesSB.append((nPos + 1) + NEW_CH + newTextLines[nPos] + NEWLINE);
    } // end of for ()
  }

  //@ ensures \nonnullelements(\result);
  private /*@ helper non_null */ String[] splitByLine(/*@ non_null */ String text) {
    // thanks to Windows ridiculous two character newlines it is
    // hard to detect blank lines, so we don't bother trying
    StringTokenizer toker = new StringTokenizer(text, DELIM, false);
    ArrayList lines = new ArrayList(oldText.length() / 60);
    while (toker.hasMoreTokens()) {
      String tok = toker.nextToken();
      lines.add(tok);
    }
    String[] result = new String[lines.size()];
    lines.toArray(result);
    return result;
  }

  /**
   * Returns true if strings on which this was constructed are different.
   */
  /*@ public normal_behavior
    @ ensures \result == areDifferent;
    @*/
  public /*@ pure @*/ boolean areDifferent() {
    return _areDifferent;
  }

  /**
   * Returns the differences between the given strings.
   *  
   */
  //@ private normal_behavior
  //@ ensures \result == differences;
  //@ pure
  public String result() {
    return differences;
  }

  //---------------------------------------------------------------------
  // PRIVILEGED DATA
  //---------------------------------------------------------------------

  /** This is the supplied old text, to be compared against the new text */
  private /*@ non_null */ String oldText;

  /** This is the supplied new text, to be compared against the old text */
  private /*@ non_null */ String newText;

  /** This is set to true if the oldText and newText are not the same */
  private boolean _areDifferent;

  /**
   * This output String holds the description of the differences between the old
   * and new text.
   */
  /*@ spec_public */ private /*@ non_null */ String differences;

  //@ public model boolean areDifferent;
  //@ public represents areDifferent <- !differences.equals("");

  //@ private invariant _areDifferent == areDifferent;

  /** This string holds line delimiters */
  private static final /*@ non_null */ String DELIM = "\n\r\f";

  /** This is the system new line character */
  private static final /*@ non_null */ String NEWLINE = /* @ (non_null String) */ System.getProperty("line.separator");

  /** This string is used to mark lines of old text */
  private static final /*@ non_null */ String OLD_CH = "< ";

  /** This string is used to mark lines of new text */
  private static final /*@ non_null */ String NEW_CH = "> ";
}
