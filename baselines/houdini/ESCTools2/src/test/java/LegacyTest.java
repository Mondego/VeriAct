import junit.framework.*;
import junitutils.*;

public class LegacyTest extends junitutils.TestFilesTestSuite {

    static public junit.framework.Test suite() {
    
    	String defaultSettings = "-nowarn Deadlock -verboseTrace -testMode";
    	String listOfFiles = "src/test/legacy.txt";
    	String testSuiteName = "Legacy-Test";

        junit.framework.TestSuite suite =
	    new TestFilesTestSuite(
		testSuiteName,
		listOfFiles,
		defaultSettings,
		escjava.Main.class) {
		    public int expectedIntegerStatus(String f, String o) {
			if (javafe.util.ErrorSet.errors > 0) return 2;

			return 0;
		    }
                    public String expectedStatusReport(String fileToTest,
                        int ecode, String expectedOutput) {
                        int ret = expectedIntegerStatus(fileToTest,expectedOutput);
                        if ((ecode > 0) == (ret > 0)) return null;
                        return super.expectedStatusReport(fileToTest,ecode,expectedOutput);
                    }
		};
	return suite;
    }
}
