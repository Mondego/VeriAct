//@ typedeclelem INVARIANT
//@ lexical 100 ;

//@ lexical 100
//@ modifier PRE
//@lexical 110
// @bad

abstract class HasPragmas {

  //@ typedeclelem Hi
  //@typedeclelem there
  //@ lexical 100 ;
  //@lexical 110;
  //@ typedeclelem Hi     ;
  //@typedeclelem there;
  
  /*@ modifier Hi */ int i,j = 1, k /*@ modifier Hi */ ;
  /*@ modifier Hi */ int i2,j2 = 1, k2;
  int i3,j3 = 1, k3 /*@ modifier Hi */ ;

  //@ typedeclelem PRE_INNER
  //@ Modifier PRE_INNER_MOD
  public static class Inner { /*@ typedeclelem INNER; */ }

  /*@ modifier Hi */ public 
  /*@ modifier Hi */ static 
  /*@ modifier Hi */ 

  void main(/*@ modifier Hi */ 
	    String[] args )

    throws IOException
    
    //@ lexical 100 ;
    //@lexical 110;
    //@ modifier Hi     ;
    
    {
    
      int i;
      //@ statement Hi
      //@statement there
      //@ lexical 100 ;
      //@lexical 110;
      //@ statement Hi     ;
      
      i=0;
      
	;

      /*@ modifier Hi */ int k,j = 0, l /*@ modifier Hi */ ;
      int k2,j2 = 0, l2 /*@ modifier Hi */ ;
    }

   HasPragmas() /*@ modifier Hi */ {
   }
   HasPragmas(int i) throws IOException /*@ modifier Hi */ {
   }

   abstract void m() /*@ modifier Hi */;
}
