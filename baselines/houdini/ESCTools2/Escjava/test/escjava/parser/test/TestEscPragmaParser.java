/* $Id$ */
/* Copyright 2000, 2001, Compaq Computer Corporation */

package escjava.parser.test;

import escjava.ast.EscPrettyPrint;
import escjava.parser.EscPragmaParser;

import javafe.ast.PrettyPrint;
import javafe.ast.StandardPrettyPrint;
import javafe.ast.DelegatingPrettyPrint;
import javafe.parser.Lex;
import javafe.parser.test.TestParse;

import java.io.IOException;

public class TestEscPragmaParser {
  public static void main(String[] argv) throws IOException {
    escjava.Main.options = new escjava.Options();
    escjava.Main.options().nvu = true;
    escjava.Main.options().noSemicolonWarnings = true;

    // DelegatingPrettyPrint p = new javafe.tc.TypePrint();
    // p.setDel(new EscPrettyPrint(p, new StandardPrettyPrint(p)));
    DelegatingPrettyPrint p = new EscPrettyPrint();
    p.setDel(new StandardPrettyPrint(p));
    PrettyPrint.inst = p;

    TestParse.lexer = new Lex(new EscPragmaParser(), true);
    TestParse.main(argv);
  }
}
