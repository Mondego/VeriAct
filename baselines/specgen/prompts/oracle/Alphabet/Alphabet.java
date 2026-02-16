public class Alphabet {
   private boolean vowel_set = false;
   private boolean vowel;
   /*@ private invariant vowel_set ==> 
                     (vowel == (c == 'a' || c == 'A' || c == 'e' || c == 'E' 
                               || c == 'i' || c == 'I' || c == 'o'
			       || c == 'O' || c == 'u' || c == 'U')); @*/

   private boolean alphabetic_set = false;
   private boolean alphabetic;
   //@ private invariant alphabetic_set ==> alphabetic == (('a' <= c && c <= 'z')||('A' <= c && c <='Z'));
   private boolean digit_set = false;
   private boolean digit;
   //@ private invariant digit_set ==> digit == ('0' <= c && c <= '9');
   private boolean uppercase_set = false;
   private boolean uppercase;
   //@ private invariant uppercase_set ==> (uppercase == ('A' <= c && c <= 'Z'));
   private boolean lowercase_set = false;
   private boolean lowercase;
   //@ private invariant lowercase_set ==> (lowercase == ('a' <= c && c <= 'z'));

   private /*@ spec_public @*/ final char c;

   /*@ private normal_behavior
     @   ensures !vowel_set && !alphabetic_set && !digit_set;
     @   ensures !uppercase_set && !lowercase_set;
     @ public normal_behavior
     @   ensures this.c == c; @*/
   public /*@ pure @*/ Alphabet(char c) 
   {
	this.c = c;
   }

   /*@ private normal_behavior
     @   assignable vowel_set, vowel;
     @   ensures vowel_set && \result == vowel;
     @ public normal_behavior
     @   	ensures \result == (c == 'a' || c == 'A' || c == 'e' || c == 'E' 
     @    	                      || c == 'i' || c == 'I' || c == 'o'
     @				      || c == 'O' || c == 'u' || c == 'U'); @*/
   public boolean isVowel() 
   {
       setVowel();
       return vowel;
   }

   /*@ private normal_behavior
     @   assignable alphabetic_set, alphabetic;
     @   ensures alphabetic_set && \result == alphabetic; 
     @ public normal_behavior
     @   ensures \result == ('a' <= c && c <= 'z')||('A' <= c && c <='Z'); @*/
   public boolean isAlphabetic() 
   {
	setAlphabetic();
	return alphabetic;
   }

   /*@ private normal_behavior
     @   assignable uppercase_set, uppercase;
     @   ensures uppercase_set && \result == uppercase; 
     @ public normal_behavior
     @   ensures \result == ('A' <= c && c <= 'Z'); @*/
   public boolean isUppercase() 
   {
	setUppercase();
	return uppercase;
   }

   /*@ private normal_behavior
      @   assignable lowercase_set, lowercase;
      @   ensures lowercase_set && \result == lowercase; 
      @ public normal_behavior
      @   ensures \result == ('a' <= c && c <= 'z'); @*/
   public boolean isLowercase() 
   {
       setLowercase();
       return lowercase;
   }

   /*@ private normal_behavior
     @   assignable digit_set, digit;
     @   ensures digit_set && \result == digit; 
     @ public normal_behavior
     @   ensures \result == ('0' <= c && c <= '9'); @*/
   public boolean isDigit() 
   {
	setDigit();
	return digit;
   }

   /*@ private normal_behavior
     @   assignable vowel_set, vowel;
     @   ensures vowel_set;
     @   ensures vowel <==> (c == 'a' || c == 'A' || c == 'e' || c == 'E' 
     @                     	|| c == 'i' || c == 'I' || c == 'o'
     @	 	 	     	|| c == 'O' || c == 'u' || c == 'U'); @*/
   private /*@ spec_public @*/ void setVowel() 
   {
        vowel = false;
        switch (c) {
            case 'a' :
            case 'e' :
            case 'i' :
            case 'o' :
            case 'u' :
            case 'A' :
            case 'E' :
            case 'I' :
            case 'O' :
            case 'U' : vowel = true;
        }
        vowel_set = true;
   }

   /*@ private normal_behavior
     @    assignable alphabetic_set, alphabetic;
     @    ensures alphabetic_set;
     @    ensures alphabetic <==> ('a' <= c && c <= 'z')||('A' <= c && c <= 'Z'); @*/
   private /*@ spec_public @*/ void setAlphabetic() 
   {
	alphabetic = (('a' <= c && c <= 'z')||('A' <= c && c <= 'Z'));		
	alphabetic_set = true;
   }

   /*@ private normal_behavior
     @    assignable uppercase_set, uppercase;
     @    ensures uppercase_set;
     @    ensures uppercase <==> ('A' <= c && c <= 'Z'); @*/
   private /*@ spec_public @*/ void setUppercase() 
   {
	uppercase = ('A' <= c && c <= 'Z');		
	uppercase_set = true;
   }

   /*@ private normal_behavior
     @    assignable lowercase_set, lowercase;
     @    ensures lowercase_set;
     @    ensures lowercase <==> ('a' <= c && c <= 'z'); @*/
   private /*@ spec_public @*/ void setLowercase() 
   {
       lowercase = ('a' <= c && c <= 'z');
       lowercase_set = true;
   }

   /*@ private normal_behavior
     @    assignable digit_set, digit;
     @    ensures digit_set;
     @    ensures digit <==> ('0' <= c && c <= '9'); @*/
   private /*@ spec_public @*/ void setDigit() 
   {
	digit = ('0' <= c && c <= '9');	
	digit_set = true;
   }

   /*@ private normal_behavior
     @   ensures \result == alphabetic_set; @*/
   public /*@ pure @*/ boolean getAlphabetic_set()
   {
	return alphabetic_set;
   }

   /*@ private normal_behavior
     @   ensures \result == uppercase_set; @*/
   public /*@ pure @*/ boolean getUppercase_set()
   {
	return uppercase_set;
   }

   /*@ private normal_behavior
     @   ensures \result == lowercase_set; @*/
   public /*@ pure @*/ boolean getLowercase_set()
   {
	return lowercase_set;
   }

   /*@ private normal_behavior
     @   ensures \result == vowel_set; @*/
   public /*@ pure @*/ boolean getVowel_set()
   {
	return vowel_set;
   }

   /*@ private normal_behavior
     @   ensures \result == digit_set; @*/
   public /*@ pure @*/ boolean getDigit_set()
   {
	return digit_set;
   }
   
   /*@ requires 0 <= op && op <= 4;
     @ {|
     @      requires op == 0;
     @      ensures \result[0] ==> (c == 'a' || c == 'A' || c == 'e' || c == 'E' || c == 'i' || c == 'I' || c == 'o' ||
     @		 	         c == 'O' || c == 'u' || c == 'U'); 
     @	    ensures \result[1];
     @ also
     @      requires op == 1;
     @      ensures \result[0] ==> ('A' <= c && c <= 'Z');
     @	    ensures \result[2];
     @ also
     @      requires op == 2;
     @      ensures \result[0] ==> ('a' <= c && c <= 'z');
     @	    ensures \result[3];
     @ also
     @      requires op == 3;
     @      ensures \result[0] ==> ('0' <= c && c <= '9');
     @	    ensures \result[4];
     @ also
     @      requires op == 4;
     @	    ensures \result[5];
     @ |} @*/
   public boolean[] driver(int op) 
   {
	boolean[] result = new boolean[6];
	switch (op) {
		case 0 :
		result[0] = isVowel();
		result[1] = getVowel_set();
		break;
		
		case 1 :
		result[0] = isUppercase();
		result[2] = getUppercase_set();
		break;

		case 2 :
		result[0] = isLowercase();
		result[3] = getLowercase_set();
		break;

		case 3 :
		result[0] = isDigit();
		result[4] = getDigit_set();
		break;

		default :
		result[0] = isAlphabetic();
		result[5] = getAlphabetic_set();
		break;
	}
	return result;
   }
}
