package tohtml;

import java.io.*;
import java.util.*;

class LineNumberPrintWriter {
  private boolean inCodeSegment = false;
  private PrintWriter pw;
  private long lineNumber = 1;
  private boolean atBeginningOfLine = true;

  LineNumberPrintWriter(PrintWriter pw) {
    this.pw = pw;
  }

  public void enterCodeSegment() {
    inCodeSegment = true;
    atBeginningOfLine = true;
  }
  
  public void exitCodeSegment() {
    if (!atBeginningOfLine) {
      lineNumber++;
      atBeginningOfLine = true;
    }
    inCodeSegment = false;
  }

  public void print(String s) {
    int n = s.length();
    for (int i = 0; i < n; i++) {
      print(s.charAt(i));
    }
  }

  public void print(char ch) {
    if (inCodeSegment) {
      if (atBeginningOfLine) {
	String s = "     " + lineNumber;
	pw.print("<A NAME=\"" + lineNumber + "\"></A>");
	pw.print(Java2Html.markupLineNumberBegin);
	pw.print(s.substring(s.length()-5) + "  ");
	pw.print(Java2Html.markupLineNumberEnd);
	atBeginningOfLine = false;
      }
    }
    pw.print(ch);
    if (inCodeSegment && ch == '\n') {
      atBeginningOfLine = true;
      lineNumber++;
    }
  }

  public void println(String s) {
    print(s);
    print('\n');
  }

  public void close() {
    pw.close();
  }
}
