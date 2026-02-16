/*
 Adaptation of Bergstra & Loots's Java Class Family number 13:
  
 Results are written to boolean variables `resulti' instead of
 being printed.
  
 This example involves mutual recursion.
  
 Sunday 13 June 99 12:49:02 bart@frustratie

 Added to ESC/Java2 repository by Joe Kiniry on 19 Jan 2004.
 */

package jcf13; 

class c {
    static boolean result1, result2, result3;
    
    static boolean b1 = d.b2;
    
    /*@ normal_behavior  
     @   requires true;  /// problem: need to know that c and d
     /// are not initialized yet.
     @ modifiable \everything;
     @    ensures result1 && result2 && result3;
     @*/
    public static void m() 
    {
	result1 = b1;
	result2 = d.b2;
	result3 = d.m();
    }
    
}

class d {
    static boolean b2 = !c.b1;
    
    public static boolean m() {
	return b2;
    }
    
}
