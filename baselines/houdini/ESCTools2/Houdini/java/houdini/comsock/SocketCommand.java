/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.comsock;

import java.io.*;
import java.net.*;
import java.util.*;

import houdini.util.*;

/**
 * This class receives commands from a socket and then the execute some code. 
 * @see SocketSender
 */
public class SocketCommand extends Thread {
   
    protected Hashtable commands = new Hashtable();
    protected int port;
    protected Socket socket = null;
    protected DataInputStream in;
    protected DataOutputStream out;    
    protected boolean connected = false;
    
    protected Command exitCommand = null;
    
    public boolean isConnected() {
        return this.connected;
    }
    
    /**
     * Pass in the port on which to listen. 
     */
    public SocketCommand(int port) {
        super("SocketCommand(" + port + ")");
        this.port = port;
    }
    
    /**
     * Create a new entry in the dispatch table. 
     * @param s command name
     * @param c command code to run on receipt
     */
    public void addCommand(String s, Command c) {
        this.commands.put(s, c);
    }

    /** create only one server for each port */
    private static Vector servers = new Vector();
    
    static private synchronized ServerSocket getServer(int port) throws IOException {
        ServerSocket ss;
        for (int i = 0; i < servers.size(); i++) {
            ss = (ServerSocket)servers.elementAt(i);
            if (ss.getLocalPort() == port) return ss;
        }
        ss = new ServerSocket(port);
        servers.addElement(ss);
        return ss;
    }
    
    /**
     * Set up the connection by listening on the port. 
     */
    public void connect() {
        try {
            ServerSocket server = getServer(port);
            socket = server.accept();
            in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));            
        } catch (Exception e) {
            Assert.fail(e);
        }
        this.connected = true;
    }

    /**
     * Close the connection.
     */
    public synchronized void close() throws IOException {
	if (!this.connected) return;
	this.connected = false;
	in.close();
	out.close();
	socket.close();
    }
    
    /**
     * read one line from the socket.
     */
    protected String getMessage() throws IOException {
	String s = in.readLine();
	if (s != null) {
	    s = s.replace('\4','\n');
	    if (Debug.debug) Log.log("socket", "msg: " + Utility.replaceString(s, "\3", "::", false));
	}
	return s;
    }

    /**
     * parse the string and return the arguments as an array. 
     */   
    protected String[] getArgs(String s) {
        return Utility.stringListToArray(s, "\3");
    }

    /**
     * parse the string and return the command name. 
     */   
    protected String getCommand(String s) {
        return getArgs(s)[0];
    }
    
    /**
     * look up the command and call the correct method. 
     */
    protected void processMessage(String com, String args[]) throws IOException {
        Command r = (Command)commands.get(com);
        Assert.notFalse(r != null, "unrecognized command: " + com);
        String s;
	s = r.doIt(args);
	if (Debug.debug) Log.log("socket", "ack: " + s);
	s = s.replace('\n', '\4') + "\n";
	out.writeBytes(s);
	out.flush();
    }

    /**
     * Main entry point for clients.  This call blocks
     * until the connection is established.
     */
    public void start() {
        this.connect();
        super.start();
    }

    /**
     * iterate reading a line from the socket and then processing in it. 
     */
    public void run() {
        String exitString = "";
        try {
            while (true) {
                String s = this.getMessage();
                if (s == null) break;
                String c = this.getCommand(s);
                String args[] = this.getArgs(s);                
                this.processMessage(c, args);
            }            
        } catch (java.io.IOException e) {
            exitString = e.toString();
        }
        try {
            close();
        } catch (java.io.IOException e) {
            exitString = e.toString();
        }
        String array[] = { exitString };
        if (this.exitCommand != null) this.exitCommand.doIt(array);
    }
    
    public void setExitCommand(Command c) {
        this.exitCommand = c;
    }
    
}
      
