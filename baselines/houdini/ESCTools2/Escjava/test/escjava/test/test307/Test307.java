public class Test307 {
	final /*@non_null*/String s444[/*#@nullable*/] = new String[3];
	
	void m111() {
		String s = "";
		for(int i = 0; i < s444.length; i++)
			s += s444[i]; // ok
	}
	void m333() {
		int s = 0;
		for(int i = 0; i < s444.length; i++)
			s += s444[i].length(); // error
	}

	//@ invariant (\forall int i; s444.length == s444.length);
	/*@ invariant (\forall int i; 0 <= i && i < s444.length;
	  @ // FIXME: THE NEXT LINE IS DISABLED DUE TO A BUG IN ESC.
	  @ //			s444[i] == s444[i]); // NO ERROR SHOULD BE REPORTED HERE.
	  @ 			true);
	  @*/
}
