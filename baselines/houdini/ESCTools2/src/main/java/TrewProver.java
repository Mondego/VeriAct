package escjava.sortedProver.trew;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Properties;

import javafe.util.ErrorSet;
import javafe.util.Info;
import escjava.sortedProver.CounterExampleResponse;
import escjava.sortedProver.SortedProverCallback;
import escjava.sortedProver.SortedProverResponse;
import escjava.sortedProver.NodeBuilder.SPred;
import escjava.sortedProver.auflia.AufliaProver;
import escjava.Main;

/*@ non_null_by_default @*/
public class TrewProver extends AufliaProver
{
	private Process trew;

	private String readShortLine()
	{
		String line =readLine();
		if (line.length() > 100)
			line = line.substring(0, 100) + "...";
		return line;
	}
	
	private String readLine()
	{
		try {
			InputStream s, out = trew.getInputStream(), err = trew.getErrorStream();
			if (out.available() < err.available())
				s = err;
			else
				s = out;			
			
			int c = s.read();
			while (c == ' ' || c == '\n' || c == '\r')
				c = s.read();
			
			if (c < 0) {
				if (out.available() > 0 || err.available() > 0)
					return readLine();
				return "EOF";
			}
			
			StringBuffer sb = new StringBuffer();
			sb.append((char)c);
			for (;;) {
				c = s.read();
				if (c == '\n' || c == '\r' || c < 0) break;
				sb.append((char)c);
			}
			
			String line = sb.toString();
			if (Info.on)
				Info.out("fx7: " + line);
			return line;
		} catch (IOException x) {
			ErrorSet.fatal(x.toString());
			return null; // unreached
		}
	}
	
	public SortedProverResponse isValid(SPred formula, SortedProverCallback callback, Properties properties)
	{
		String filename = encodeProblemName(properties);
		filename += ".smt";
		
		String resp = checkProof(formula, filename);
		if (resp != null) {
			ErrorSet.error(resp);
			return new SortedProverResponse(SortedProverResponse.FAIL);
		} else {
			return new SortedProverResponse(SortedProverResponse.YES);			
		}
	}

	protected String checkProof(SPred formula, String filename)
	{
		saveQuery(filename, formula);
		
		if (! (new File(filename + ".rw")).canRead())
			return "proof file for " + filename + " not found";	
		
		String args[] = new String[] { "trew", "-t", "prelude.rw", "-q", filename, "-p",
								       filename + ".rw" };
		long time = Main.currentTime();	
		try {			
			trew = Runtime.getRuntime().exec(args,null);
		} catch (IOException x) {
			ErrorSet.fatal(x.toString());
		}		
		
		Info.out("invoking trew on " + filename);
		
		String line = readShortLine();
		StringBuffer buf = new StringBuffer(line);		
		while (! line.equals("EOF")) {
			line = readShortLine();
			buf.append('\n').append(line);
		}
		
		int exit = 1;
		
		try {
			exit =trew.waitFor(); 
		} catch (InterruptedException e) { }

		System.out.println("proof checking time: " + Main.timeUsed(time));
			
		
		if (exit == 0) {
			return null;
		} else {
			return "trew verify failed:\n" + buf;
		}
	}
}
