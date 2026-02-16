package instrumenter;

/**
 ** This class provides the functions for checking. <p>
 **/

public class Check {

  static int[] annotationTrueCount;
  static int[] annotationFalseCount;

  final static String outPrefix = "InStRuMeNtEr: ";
  
  static {
    annotationTrueCount = new int[houdini.Info.annotationLocs.length];
    annotationFalseCount = new int[houdini.Info.annotationLocs.length];
  }
  
  public synchronized static void printError(String srcloc, int k, String msg) {
    System.err.print(outPrefix + houdini.Info.annotationLocs[k]);
    System.err.println(" " + srcloc + ": Warning: " + msg);
  }

  public synchronized static boolean checkAssertion(String srcloc, int k, boolean p) {
    if (p) {
      annotationTrueCount[k] += 1;
    }
    else {
      if (annotationFalseCount[k] == 0) {
	printError(srcloc, k, "assertion violated at run-time");
      }
      annotationFalseCount[k] += 1;
    }
    return p;
  }
  
  public synchronized static Object checkNonNull(String srcloc, int k, Object p) {
    if (p == null) {
      if (annotationFalseCount[k] == 0) {
	printError(srcloc, k, "non_null violated at run-time");
      }
      annotationFalseCount[k] += 1;
    }
    else {
      annotationTrueCount[k] += 1;
    }
    return p;
  }

}
