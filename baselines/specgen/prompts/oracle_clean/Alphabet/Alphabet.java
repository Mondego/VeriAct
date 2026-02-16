public class Alphabet {
   private boolean vowel_set = false;
   private boolean vowel;
   

   private boolean alphabetic_set = false;
   private boolean alphabetic;
   
   private boolean digit_set = false;
   private boolean digit;
   
   private boolean uppercase_set = false;
   private boolean uppercase;
   
   private boolean lowercase_set = false;
   private boolean lowercase;
   

   private  final char c;

   
   public  Alphabet(char c) 
   {
	this.c = c;
   }

   
   public boolean isVowel() 
   {
       setVowel();
       return vowel;
   }

   
   public boolean isAlphabetic() 
   {
	setAlphabetic();
	return alphabetic;
   }

   
   public boolean isUppercase() 
   {
	setUppercase();
	return uppercase;
   }

   
   public boolean isLowercase() 
   {
       setLowercase();
       return lowercase;
   }

   
   public boolean isDigit() 
   {
	setDigit();
	return digit;
   }

   
   private  void setVowel() 
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

   
   private  void setAlphabetic() 
   {
	alphabetic = (('a' <= c && c <= 'z')||('A' <= c && c <= 'Z'));		
	alphabetic_set = true;
   }

   
   private  void setUppercase() 
   {
	uppercase = ('A' <= c && c <= 'Z');		
	uppercase_set = true;
   }

   
   private  void setLowercase() 
   {
       lowercase = ('a' <= c && c <= 'z');
       lowercase_set = true;
   }

   
   private  void setDigit() 
   {
	digit = ('0' <= c && c <= '9');	
	digit_set = true;
   }

   
   public  boolean getAlphabetic_set()
   {
	return alphabetic_set;
   }

   
   public  boolean getUppercase_set()
   {
	return uppercase_set;
   }

   
   public  boolean getLowercase_set()
   {
	return lowercase_set;
   }

   
   public  boolean getVowel_set()
   {
	return vowel_set;
   }

   
   public  boolean getDigit_set()
   {
	return digit_set;
   }
   
   
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
