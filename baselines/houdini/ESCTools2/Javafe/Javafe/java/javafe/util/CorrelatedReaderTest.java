/* Copyright 2000, 2001, Compaq Computer Corporation */

package javafe.util;

import java.util.Random;
import java.io.*;

public class CorrelatedReaderTest
{
  static final int SEED          = 0xCAFEBABE;
  static final int MASKRESET     = 0x1f;
  static final int MASKSUBREADER = 0x1ff;
  static final int MASKDISCARD   = 0xf;

  //----------------------------------------------------------------------

  //@ ensures false;
  //@ diverges true;
  //@ signals (Exception) false;
  static void error() {
    System.out.println("Usage: java CorrelatedReader [locs|reset|subreader]");
    System.exit(1);
  }

  /** A test harness.
   */

  //@ requires \nonnullelements(argv);
  public static void main(String[] argv)
	throws IOException, IndexOutOfBoundsException {

    CorrelatedReader cin = new FileCorrelatedReader( System.in, "stdin" );
    Random random = new Random( SEED ); // deterministic test
    int c;
    
    try {
      if( argv.length != 1 ) {
	error();
      } 
      else if( argv[0].equals("locs") ) {
	while( (c=cin.read()) != -1 ) {
	  int loc=cin.getLocation();
	  System.out.println("["+c+"'"+ (new Character((char)c)).toString()
			     +"' loc:"+ loc+" "+Location.toString(loc)
			     +" offset "+Location.toOffset(loc)+"]");
	}
      } 
      else if( argv[0].equals("reset") ) {
      
	while( (c=cin.read()) != -1 ) {
	  cin.mark();
	  c=cin.read();
	  int loc=cin.getLocation();
	  /*
	    System.out.println("["+c+"'"+ (new Character((char)c)).toString()
			       +"' loc:"+ loc+" "+Location.toString(loc)
			       +" offset "+Location.toOffset(loc)+"]");
	  */
        
	  int toRead = random.nextInt() & MASKRESET ;
	  for(int j=0; j<toRead; j++) cin.read();
	  cin.reset();
	  int c2=cin.read();
	  int loc2 = cin.getLocation();
	  Assert.notFalse( c==c2 && loc==loc2,		//@ nowarn Pre;
			  "c="+c
			  +" c2="+c2
			  +" loc="+loc
			  +" loc2="+loc2);
	}
      } 
      else if( argv[0].equals("subreader") ) {
      
	while( (c=cin.read()) != -1 ) {
	
	  StringBuffer sb = new StringBuffer();
	  


	  // mark the position of the next character;
	  // then get the next character and its location:
	  cin.mark();
	  sb.append( (char)cin.read() );
	  int loc=cin.getLocation();


	  int toRead = random.nextInt() & MASKSUBREADER;
	  int read;
	
	  for(read=1; read<toRead && (c=cin.read()) != -1; read++) 
	    sb.append( (char)c );
	

	  int discard = random.nextInt() & MASKDISCARD;
	  if( discard > read )
	      discard = read;
	
	  CorrelatedReader subReader = cin.createReaderFromMark(discard);

	  // Test location if subReader not empty:
	  if (discard<read) {
	      // loc2 = location of first character in subReader:
	      subReader.mark();
	      subReader.read();
	      int loc2 = subReader.getLocation();
	      subReader.reset();
	
	      Assert.notFalse(loc==loc2,			//@ nowarn Pre;
			      "loc="+loc
			      +" loc2="+loc2
			      +" read="+read
			      +" discard="+discard );
	  }


	  for(int j=0; j<read-discard; j++) {
	    int sc = subReader.read();
	    Assert.notFalse(sb.charAt(j) == sc,		//@ nowarn Pre;
			    "sc='"+(char)sc+"'"
			    +" sb("+j+")='"+sb.charAt(j)+"'"
			    +"sb='"+sb+"'"
			    +" loc="+loc
			    +" read="+read+" discard="+discard );
	  }
	
	  Assert.notFalse( subReader.read() == -1 );	//@ nowarn Pre;
	}
      }
      else
	error();
    } catch( AssertionFailureException e ) {
      e.printStackTrace();
      System.exit(1);
    }
  }
}
