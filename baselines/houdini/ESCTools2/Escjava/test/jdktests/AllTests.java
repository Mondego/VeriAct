
import java.io.*;
import junit.framework.Test;
import junit.framework.TestSuite;

/*
 * This file is part of the Esc/Java2 project. Copyright 2004 David R.
 * Cok
 * 
 * Created on Feb 13, 2005
 */

/**
 * TODO Fill in class description
 * 
 * @author David R. Cok
 */
public class AllTests {

  public static void main(String[] args) {
    junit.textui.TestRunner.run(AllTests.suite());
  }

  public static Test suite() {
    TestSuite suite = new TestSuite("Test for default package");
    //$JUnit-BEGIN$
/*
    File[] ff = new File(".").listFiles();
    for (int i=0; i< ff.length; ++i) {
        String s = ff[i].getName();
        if (!s.endsWith(".java") || !s.startsWith("Test")) continue;
        s = s.substring(0,s.length()-5);
        try {
	    suite.addTestSuite(Class.forName(s));
        } catch (Exception e) {
            System.err.println("Failed on " + s);
        }
    }
*/
    try {
    BufferedReader r = new BufferedReader(new FileReader("listfiles"));
    String s;
    while ( (s=r.readLine()) != null ) {
        s = s.trim();
        s = s.substring(0,s.length()-5);
        try {
	    suite.addTestSuite(Class.forName(s));
        } catch (Exception e) {
            System.err.println("Failed on " + s);
        }
    }
    } catch (IOException e) {
        System.out.println("IO Error occurred: " + e);
    }
    //$JUnit-END$
    return suite;
  }
}
