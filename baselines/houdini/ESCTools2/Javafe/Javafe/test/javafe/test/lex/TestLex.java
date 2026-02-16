/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.test.lex;

import java.util.Random;


import javafe.ast.Identifier;
import javafe.ast.PrettyPrint;

import javafe.parser.Lex;

import javafe.util.Assert;
import javafe.util.FileCorrelatedReader;

import javafe.tc.TagConstants;


/**
 ** Tokenizes standard input and prints <TT>Lex.toString</TT> of the
 ** resulting stream, on token per output line.
 **/

public class TestLex {

   public static void main(String argv[]) {
	try {
	    Random r = new Random(System.currentTimeMillis());
	    Lex ll = new Lex(null, true);

	    ll.restart(new FileCorrelatedReader(System.in,"stdin"));

	    int lac = 3;
	    int la = ll.lookahead(lac);
	    while (ll.ttype != TagConstants.EOF) {
		Assert.notFalse(lac != 0 || ll.ttype == la,
			"Bad lookahead, is "
				+ PrettyPrint.inst.toString(ll.ttype)
			+ ", expected " + PrettyPrint.inst.toString(la));
		System.out.println(ll.ztoString());
		ll.getNextToken();
		if (lac == 0) {
		    lac = Math.abs(r.nextInt()) % 9 + 1;
	            la = ll.lookahead(lac);
		} else lac--;
		ll.zzz("");
	    }
	    Identifier.check();
	} catch (Exception e) { e.printStackTrace(System.out); }
    }
}
