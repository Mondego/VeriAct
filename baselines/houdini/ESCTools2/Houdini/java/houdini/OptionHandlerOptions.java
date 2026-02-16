package houdini;

import java.io.PrintStream;
import javafe.*;
import houdini.util.Debug;

public class OptionHandlerOptions extends escjava.Options
{
    /** port for communication */
    public int port = 17777;
    
    /** host of server */
    public String host = null;

    /** directory where vc files live */
    String vcdir = "./";

    /** unversial background predicate to use */
    String univBackPredFile = "./UnivBackPred.ax";

    /** log output to file */
    public boolean logToFile = false;

    /** output stream for logging (default is System.out) */
    PrintStream logFile = System.out;

    /**
     ** Process next tool option. <p>
     **
     ** See <code>Tool.processOption</code> for the complete
     ** specification of this routine.<p>
     **
     ** This routine handles the standard front-end options, storing the
     ** resulting information in the preceding instance variables and
     ** <code>Info.on</code>.<p>
     **/
    public int processOption(String option, String[] args, int offset) throws javafe.util.UsageError {
	if (option.equals("-debug")) {
	    Debug.debug = true;
	    return offset;
	} else if (option.equals("-nodebug")) {
	    Debug.debug = false;
	    return offset;
	} else if (option.equals("-logToFile")) {
	    this.logToFile = true;
	    return offset;
	} else if (option.equals("-port")) {
	    if (offset>=args.length) {
	        usage("OptionHandler");
                throw (new javafe.util.UsageError("invalid port option"));
	    }
	    port = Integer.parseInt(args[offset]);
	    return offset+1;
	} else if (option.equals("-ubp")) {
	    if (offset>=args.length) {
	        usage("OptionHandler");
                throw (new javafe.util.UsageError("invalid ubp option"));
	    }
	    univBackPredFile =args[offset];
	    return offset+1;
	} else if (option.equals("-host")) {
	    if (offset>=args.length) {
	        usage("OptionHandler");
                throw (new javafe.util.UsageError("invalid host option"));
	    }
            host = args[offset];
	    return offset+1;
	} else if (option.equals("-vcdir")) {
	    if (offset>=args.length) {
	        usage("OptionHandler");
                throw (new javafe.util.UsageError("invalid vcdir option"));
	    }
	    vcdir=args[offset];
	    if (!vcdir.endsWith("/") && !vcdir.endsWith("\\")) {
	        vcdir += "/";
	    }
	    return offset+1;
	} else if (option.equals("-log")) {
	    if (offset>=args.length) {
	        usage("OptionHandler");
                throw (new javafe.util.UsageError("invalid log option"));
	    }
	    //Log.logToStdout(args[offset], Log.LL_HIGH);
	    return offset+1;
	} 
        return super.processOption(option, args, offset);
    }

    public String showOptions(boolean all) {
        StringBuffer sb = new StringBuffer (super.showOptions(all));
        String[][] data = {{"-debug", "TODO"},
            {"-nodebug","TODO"},
            {"-log <key>","TODO"},
            {"-port <port>","TODO"},
            {"-host <name>","TODO"},
            {"-ubp <universal back pred file>","TODO"},
            {"-logToFile","TODO"}};
        sb.append(data);
        return sb.toString();
    }
   
   
}
