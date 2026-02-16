import java.io.UnsupportedEncodingException;

public class TestString extends LocalTestCase {

  public static final String empty = "";
  public static String s; // arbitrary string


  //@ requires empty == "";
  public void testConstructor() {
     assertTrue( (new String()).equals("") );
     assertTrue( new String() != empty);
     assertTrue( new String().length() == 0);
     String ss = new String();
     assertTrue ( ss != ss.intern() );
     //@ assert ( !String.isInterned(ss) );
     assertTrue ( "" == "".intern() );
     //@ assert ( String.isInterned("") );
     //@ assert( \fresh(ss) );
     assertTrue (ss != s);
     String sss = new String();
     assertTrue (ss != sss);
//@ assert false; // TEST FOR CONSISTENCY - 1
  }

  //@ requires empty == "";
  public void testConstructor2() {
     String ss = new String("one");
     //@ assert( \fresh(ss) );
     assertTrue( ss != s);
     assertTrue( ss != empty);
     assertTrue( ss.equals("one"));
     assertTrue( ss.length() == 3);    
     //@ assert ( !String.isInterned(ss) );
     assertTrue (ss != ss.intern() );

     assertTrue( new String("").equals(empty));
     assertTrue( new String("").length() == 0);  
     assertTrue( new String("").equals(new String()) );
     try {
         ss= new String((String)null);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof NullPointerException );
     }
//@ assert false; // TEST FOR CONSISTENCY - 2
  }

  
  public void testConstructor3() {  // FIXME - too much time
    char[] a = {'a','b','c'};
    String ss = new String(a);
    assertTrue( ss.length() == 3);
    //@ assert ( !String.isInterned(ss) );
    assertTrue ( ss != ss.intern() );
    assertTrue ( ss.charAt(1) == a[1] );  // FIXME
    String sss = new String(a);
    assertTrue( ss.equals(sss));
    assertTrue( ss != sss);
    a[1] = 'z';
    sss = new String(a);
    assertTrue( !ss.equals(sss));   // FIXME

//@ assert false; // TEST FOR CONSISTENCY - 3
  }

  //@ requires empty == "";
  public void testConstructor3a() {
    String ss = new String(  new char[]{} );
    assertTrue (ss.length() == 0);
    assertTrue( ss.equals("") );  
    assertTrue( ss.equals(empty) );
    assertTrue (ss != empty);
     try {
         ss= new String((char[])null);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof NullPointerException );
     }

//@ assert false; // TEST FOR CONSISTENCY - 4
  }

  //@ requires empty == "";
  public void testConstructor4() {  //FIXME - takes too much time when combined with 4b
    char[] a = {'a','b','c','d','e','f','g'};
    assertTrue( a.length == 7);
    String ss = new String(a,3,2);
    //@ assert !String.isInterned(ss);
    assertTrue (ss != ss.intern());
    assertTrueNP ( ss.equals(new String("de")) );
    assertTrue ( ss.length() == 2);
    assertTrue ( ss.charAt(0) == a[3] );   // FIXME
//@ assert false; // TEST FOR CONSISTENCY - 5
  }

  //@ requires empty == "";
  public void testConstructor4b() {  // FIXME - takes too much time
    char[] a = {'a','b','c','d','e','f','g'};
    String ss = new String(a,0,7);
    assertTrue( ss.equals(new String(a)) );
    assertTrue( a[0] == ss.charAt(0) );   // FIXME

    ss = new String(a,3,0);
    assertTrue ( ss.length() == 0);
    assertTrue (ss.equals(""));
    assertTrue (ss.equals(empty));
    assertTrue ( ss != empty);

//@ assert false; // TEST FOR CONSISTENCY - 5a
  }

  public void testConstructor4a() {
     char[] a = { 'a', 'b', 'c', 'd', 'e', 'f', 'g'};
     String ss;
     try {
         ss= new String((char[])null,3,2);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof NullPointerException );
     }

     try {
         ss= new String(a,-3,-2);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof StringIndexOutOfBoundsException );
     }

     try {
         ss= new String(a,-3,2);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof StringIndexOutOfBoundsException );
     }

     try {
         ss= new String(a,3,-2);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof StringIndexOutOfBoundsException );
     }

     try {
         ss= new String(a,3,12);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof StringIndexOutOfBoundsException );
     }

     try {
         ss= new String(a,7,0);
         assertTrue ( ss.length() == 0);
         //assertTrue (false);
     } catch (Exception e) {
         assertTrue (false);
         //assertTrue (e instanceof StringIndexOutOfBoundsException );
     }

     try {
         ss= new String(a,8,0);
         assertTrue (false);
     } catch (Exception e) {
         assertTrue (e instanceof StringIndexOutOfBoundsException );
     }
//@ assert false; // TEST FOR CONSISTENCY - 6
  }

  public void testConstructor5() {
     byte[] b = new byte[7];
     String sb = new String(b);
     assertTrue( sb.length() <= 7);
     assertTrue( sb != sb.intern());
     String sbb = new String(b,0,7);
     assertTrue ( sbb.equals(sb) );   // FIXME


     String sss = new String(b,3,2);
     assertTrue ( sss.length() <= 2); // 1 or 2 bytes per character
     //@ assert !String.isInterned(sss);
     assertTrue ( sss != sss.intern() );

     String s4 = new String(b,3,2);
     assertTrueNP ( sss.equals(s4) );  // FIXME - need better handling of charsets

     b[3] = 2;
     s4 = new String(b,3,2);
     assertTrueNP ( !sss.equals(s4) );  // FIXME

     byte[] bn = null;
     try {
         String ss = new String(bn);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof NullPointerException );
     }

     try {
         String ss = new String(bn,0,7);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof NullPointerException );
     }

     try {
         String ss = new String(bn,0,-7);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException ||
                     e instanceof NullPointerException );
     }

     try {
         String ss = new String(bn,-1,0);
         assertTrue( false );
     } catch (Exception e) {
         // the implementation actually does the StringIndexOutOfBoundsException
         assertTrue( e instanceof StringIndexOutOfBoundsException ||
                     e instanceof NullPointerException );
     }

     try {
         String ss = new String(b,-1,7);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException );
     }

     try {
         String ss = new String(b,0,-7);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException );
     }

     try {
         String ss = new String(b,0,17);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException );
     }

     try {
         String ss = new String(b,17,0);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException );
     }

     try {
         String ss = new String(b,7,0);
         assertTrue ( ss.length() == 0);
     } catch (Exception e) {
         assertTrue( false );
         //assertTrue( e instanceof StringIndexOutOfBoundsException );
     }

     try {
         String ss = new String(b,8,0);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException );
     }
//@ assert false; // TEST FOR CONSISTENCY - 7
  }

  //@ requires !String.supportedEncoding("xx");
  public void testConstructor6() {
     byte[] b = new byte[7];
     byte[] bn = null;
     try {
         String ss = new String(b,0,7,null);
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof NullPointerException );
     }

     try {
         String ss = new String(bn,null);
     } catch (NullPointerException e) {
         assertTrue( e instanceof NullPointerException );
     } catch (Exception e) {
         assertTrue ( false);
     }


     try {
         String ss = new String(b,null);
     } catch (NullPointerException e) {
         assertTrue( e instanceof NullPointerException );
     } catch (Exception e) {
         assertTrue ( false);
     }


     try {
         String ss = new String(bn,"xx");
     } catch (UnsupportedEncodingException e) {
         assertTrue( e instanceof UnsupportedEncodingException );
     } catch (NullPointerException e) {
         assertTrue( e instanceof NullPointerException );
     } catch (Exception e) {
         assertTrue ( false);
     }


     try {
         String ss = new String(b,"xx");
     } catch (UnsupportedEncodingException e) {
         assertTrue( e instanceof UnsupportedEncodingException );
     } catch (Exception e) {
         assertTrue ( false);
     }


     try {
         String ss = new String(bn,0,7,null);
     } catch (UnsupportedEncodingException e) {
         assertTrue( e instanceof UnsupportedEncodingException );
     } catch (NullPointerException e) {
         assertTrue( e instanceof NullPointerException );
     } catch (StringIndexOutOfBoundsException e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException );
     } catch (Exception e) {
         assertTrue ( false);
     }

     try {
         String ss = new String(bn,0,7,"xx");
         assertTrue( false );
     } catch (UnsupportedEncodingException e) {
         assertTrue( e instanceof UnsupportedEncodingException );
     } catch (NullPointerException e) {
         assertTrue( e instanceof NullPointerException );
     } catch (StringIndexOutOfBoundsException e) {
         assertTrue( e instanceof StringIndexOutOfBoundsException );
     } catch (Exception e) {
         assertTrue ( false);
     }

     try {
         String ss = new String(b,0,7,"xx");
         assertTrue( false );
     } catch (Exception e) {
         assertTrue( e instanceof UnsupportedEncodingException );
     }
//@ assert false; // TEST FOR CONSISTENCY  - 8
  }

  public void testToString() {

    String s = new String("asd");
    String ss = s.toString();
    assertTrue ( s == ss);
//@ assert false; // TEST FOR CONSISTENCY - 9
  }

  public void testToCharArray() {
    String s = new String("qwe");
    char[] a = s.toCharArray();
    assertTrue (a != null);
    assertTrue( a.length == 3);
    char[] b = s.toCharArray();
    assertTrue ( b[1] == s.charAt(1) );   // FIXME

    assertTrue( a != b);
    //@ assert( CharSequence.equal(a,b) );
  
    char[] aa = new char[]{'s','d','f','g'};
    String ss = new String(aa);
    a = ss.toCharArray();

    assertTrue( a != aa);
    assertTrue(a.length == aa.length);
    //@ assert( CharSequence.equal(a,aa) );

    aa[2] = 'z';
    assertTrueNP ( a[2] != aa[2] );
    // FIXME //@ assert( CharSequence.equal(a,aa) );

    a = ss.toCharArray();
    assertTrueNP ( a[2] != aa[2] );
    //@ assert( !CharSequence.equal(a,aa) );   // FIXME

//@ assert false; // TEST FOR CONSISTENCY - 10
  }

  public void testConcat() {
    String s = "asd";
    String ss = "defg";
    String sss = s.concat(ss);

    assertTrue( sss.equals( s+ss) );
    assertTrueNP ( sss.length() == s.length() + ss.length() ); // FIXME - should be provable
    //@ assume sss.length() > 3; // FIXME - won't need when we can prove the above
    assertTrue( sss.charAt(0) == s.charAt(0));   // FIXME
    assertTrue( sss.charAt(3) == ss.charAt(0));  // FIXME

    String s4 = s + ss;
    assertTrue( s4.length() == s.length() + ss.length() );  // FIXME
    assertTrue( sss != sss.intern() );
    assertTrue ( s4 != s4.intern() );

    assertTrue( sss + "" != sss);
    assertTrue( "" + sss != sss);

		 // FIXME - would like to be able to prove these eventually
    assertTrueNP ( (Null() + sss).equals("nullasddefg") );
    assertTrueNP ( (sss + Null()).equals("asddefgnull") );
    assertTrueNP ( (Null() + Null()).equals("nullnull")  );

    String s5 = sss.concat("");
    assertTrue (s5.equals(sss) );
    assertTrue (s5 == sss);
    assertTrue ("".concat(sss) != sss);

    try {
	s5 = s.concat(null);
        assertTrue (false) ;
    } catch (Exception e) {
        assertTrue( e instanceof NullPointerException);
    }

//@ assert false; // TEST FOR CONSISTENCY - 11 
  }

  public void testStringBuffer() {
    StringBuffer sb = new StringBuffer();
    sb.append("asdasdasd");

    String s = new String(sb);
    assertTrue( s.contentEquals(sb) );

    sb = null;
    try {
        String ss = new String(sb);
        assertTrue( false );
    } catch (Exception e) {
        assertTrue (e instanceof NullPointerException );
    }

    try {
        s.contentEquals(sb);
        assertTrue( false );
    } catch (Exception e) {
        assertTrue (e instanceof NullPointerException );
    }

//@ assert false; // TEST FOR CONSISTENCY - 12
  }

  public void testSubstring() {

    String s = "asdfghjkl";
    String ss = s.substring(2,3);
    assertTrue( ss.length() == 1 );
    assertTrue ( ss.charAt(0) == s.charAt(2) );  // FIXME

    ss = s.substring(0,9);
    assertTrue (ss.equals(s));

    ss = s.substring(0);
    assertTrue( ss.equals(s) );

    ss = s.substring(9);
    assertTrue( ss.length() == 0);
    assertTrue( ss.equals("") );

    ss = s.substring(6);
    assertTrue( ss.length() == 3);

    try {
      s.substring(-1);
      assertTrue( false );
    } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException ) ;
    }

    try {
      s.substring(10);
      assertTrue( false );
    } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException ) ;
    }

    try {
      s.substring(-1,7);
      assertTrue( false );
    } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException ) ;
    }

    try {
      s.substring(0,-1);
      assertTrue( false );
    } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException ) ;
    }

    try {
      s.substring(2,1);
      assertTrue( false );
    } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException ) ;
    }

    try {
      s.substring(10,11);
      assertTrue( false );
    } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException ) ;
    }

    try {
      s.substring(4,11);
      assertTrue( false );
    } catch (Exception e) {
      assertTrue( e instanceof StringIndexOutOfBoundsException ) ;
    }

//@ assert false; // TEST FOR CONSISTENCY - 13
  }

  //@ ensures \result == null;
  String Null() { return null; }

  public void testSubsequence() {
    String s = "123456789";
    CharSequence c = s.subSequence(2,5);
    assertTrue ( c.length() == 3) ;
    assertTrue( c.charAt(0) == s.charAt(2));  // FIXME

    c = s.subSequence(0,9);
    //@ assert CharSequence.equal(s.charArray, c.charArray);
    assertTrue( c.charAt(0) == s.charAt(0));   // FIXME

    try {
      c = s.subSequence(-1,5);
      assertTrue( false);
    } catch(Exception e) {
      assertTrue (e instanceof StringIndexOutOfBoundsException);
    }

    try {
      c = s.subSequence(0,-5);
      assertTrue( false);
    } catch(Exception e) {
      assertTrue (e instanceof StringIndexOutOfBoundsException);
    }

    try {
      c = s.subSequence(1,15);
      assertTrue( false);
    } catch(Exception e) {
      assertTrue (e instanceof StringIndexOutOfBoundsException);
    }

    try {
      c = s.subSequence(11,15);
      assertTrue( false);
    } catch(Exception e) {
      assertTrue (e instanceof StringIndexOutOfBoundsException);
    }
//@ assert false; // TEST FOR CONSISTENCY - 14
  }

  public void testEquals() {
     testEquals("s", new String("s"), "t");
  }

  //@ requires s1 != null && s2 != null && s3 != null;
  public void testEquals(String s1,String s2,String s3) {
    //@ assume s1.equals(s2);
    //@ assert s2.equals(s1);
    //@ assert s1.equals(s1);
    //@ assert s1.equals(s3) <==> s2.equals(s3);
    //@ assert (s1.equals(s3) && s1.length() > 2) ==> s1.charAt(0) == s3.charAt(3);
//@ assert false; // TEST FOR CONSISTENCY - 15
  }
  
  public void testCopyValueOf() {
    char[] data = new char[]{'1','2','3','4','5','6','7','8','9'};

    String s = String.copyValueOf(data,9,0);
    assertTrue( s.length() == 0);
    assertTrue( s.equals("") );

    s = String.copyValueOf(data);
    assertTrue( s.length() == 9);
    String ss = String.copyValueOf(data,0,9);
    assertTrue( ss.length() == 9);
    assertTrue( s.equals(ss));

    assertTrue (s.charAt(2) == data[2]);

    try {
       s = String.copyValueOf(null);
       assertTrue ( false );
    } catch (Exception e) {
       assertTrue (e instanceof NullPointerException);
    }

    try {
       s = String.copyValueOf(null,0,0);
       assertTrue ( false );
    } catch (Exception e) {
       assertTrue (e instanceof NullPointerException);
    }

    try {
       s = String.copyValueOf(data,-1,0);
       assertTrue ( false );
    } catch (Exception e) {
       assertTrue (e instanceof StringIndexOutOfBoundsException);
    }

    try {
       s = String.copyValueOf(data,0,-1);
       assertTrue ( false );
    } catch (Exception e) {
       assertTrue (e instanceof StringIndexOutOfBoundsException);
    }

    try {
       s = String.copyValueOf(data,0,10);
       assertTrue ( false );
    } catch (Exception e) {
       assertTrue (e instanceof StringIndexOutOfBoundsException);
    }

    try {
       s = String.copyValueOf(data,-1,0);
       assertTrue ( false );
    } catch (Exception e) {
       assertTrue (e instanceof StringIndexOutOfBoundsException);
    }

    try {
       s = String.copyValueOf(data,-1,0);
       assertTrue ( false );
    } catch (Exception e) {
       assertTrue (e instanceof StringIndexOutOfBoundsException);
    }
//@ assert false; // TEST FOR CONSISTENCY - 15

  }
  public void testIntern() {
    String s = "abs";
    assertTrue( s == s.intern());
    //@ assert String.isInterned(s);
    s = new String("abs");
    assertTrue( s != s.intern());
    //@ assert !String.isInterned(s);
    s = s.intern();
    assertTrue( s == s.intern());
    //@ assert String.isInterned(s);

//@ assert false; // TEST FOR CONSISTENCY - 16
  }


  public void testCharAt() {
    String s = "abscd";
    String ss = new String(s);
    //@ assume ss.length() == 5;
    assertTrue ( s.charAt(1) == ss.charAt(1));      // FIXME

    try {
      s.charAt(-1);
      assertTrue( false );
    } catch(Exception e) {
      assertTrue ( e instanceof StringIndexOutOfBoundsException );
    }

    try {
      s.charAt(5);
      assertTrue( false );
    } catch(Exception e) {
      assertTrue ( e instanceof StringIndexOutOfBoundsException );
    }

//@ assert false; // TEST FOR CONSISTENCY - 17
  }

   /* Need tests for
	real charsets; compare use of default charset with no charset
        charAt needs work to do proofs

	getChars(begin,end,dst,destBegin)
	getBytes(charsetName)
	getBytes()

	equalsIgnoreCase(String)

	compareTo(String)

	compareTo(Object)

	compareToIgnoreCase(String)
	regionMatches
	regionMatches
	startsWith
	startsWith
	endsWith
	endsWith
	indexOf *2
	lastIndexOf *2 
	indexOf *2 (String)
	lastIndexOf *2 (String)

	replace
	matches
	replaceFirst
	replaceAll
	split *2
	toLowerCase*2
	toUpperCase*2
	trim
	** valueOf(Object)
	** valueOf(char[]) *2
	** valueOf(boolean)/char/int/long/float/double

	*/

}
