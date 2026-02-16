/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.comsock;

import java.io.*;
import java.net.*;
import java.util.*;

import houdini.util.*;

/**
 * A class that connects to a receiving socket and then is able to send command messages
 * to it. 
 * <p> 
 * Each message has the form "command arg1 arg2 ..." and can either block
 * for a return message or not wait for an acknowledgement. 
 */
public class SocketSender  {

    protected boolean connected = false;

    protected int port;
    protected InetAddress host;
    protected Socket socket = null;
    protected DataInputStream in;
    protected DataOutputStream out;    

    

    public boolean isConnected() {
	return this.connected;
    }
    
    /**
     * Pass in the port to use. 
     */
    public SocketSender(int port) {
        this(null, port);
    }
    
    /**
     * Pass in the host/port to use. 
     */
    public SocketSender(String host,int port) {
        this.port = port;         
        try {
            this.host = host == null ? InetAddress.getLocalHost() :
                                       InetAddress.getByName(host);
        } catch (Exception e) {
            Assert.fail(e);
        }
    }
    
    /**
     * Close down the socket connection. 
     */
    public void close() throws IOException {
	this.connected = false;
        this.socket.close();
    }

    private void writeMessage(String s) throws IOException {
	if (Debug.debug) Log.log("socket", "sending on " + this.port + ":" + Utility.replaceString(s, "\3", "::", false));
	out.writeBytes(s.replace('\n', '\4'));
	out.writeBytes("\n");	
	out.flush();
    }

    private String readResponse() throws IOException {
        String res = in.readLine();
        while(res == null)
            res = in.readLine();
	    return res.replace('\4', '\n');
    }
    
    /**
     * Set up the connection to the host.  
     */
    public void connect() throws ConnectException {
        try {
            socket = new Socket(host, port);
            in = new DataInputStream(new BufferedInputStream(socket.getInputStream()));
            out = new DataOutputStream(new BufferedOutputStream(socket.getOutputStream()));            
        } catch (ConnectException e) {
	    throw e;
	} catch (IOException e) {
            Assert.fail(e);
        }
        connected = true;
    }
 
    /**
     * Send a message with no arguments. 
     * @param waitForAck determines whether or not to wait for an acknowledgement message. 
     * @return null if not waiting, or ack message if waiting.
     */
    public String sendMessage(String s, boolean waitForAck) throws IOException {
	try {
	    String ack = "";
	    Assert.notFalse(connected);        
	    
	    writeMessage(s);
	    if (waitForAck) {          
		ack = readResponse();
	    } 
	    return ack;
	} catch (IOException e) {
	    this.close();
	    throw e;
	}
    }

    /**
     * Send a message with 1 argument.
     * @param args the string argument.  
     * @param waitForAck determines whether or not to wait for an acknowledgement message. 
     * @return null if not waiting, or ack message if waiting.
     */
    public String sendMessage(String s, String arg, boolean waitForAck) throws IOException {
	try {
	    String ack = "";
	    Assert.notFalse(connected);        
	    s += "\3" + arg;
	    writeMessage(s);
	    if (waitForAck) {          
		ack = readResponse();
	    }
	    return ack;
	} catch (IOException e) {
	    this.close();
	    throw e;
	}
    }
    
    /**
     * Send a message with arguments.
     * @param args the array of string arguments.  
     * @param waitForAck determines whether or not to wait for an acknowledgement message. 
     * @return null if not waiting, or ack message if waiting.
     */
    public String sendMessage(String s, String args[], boolean waitForAck) throws IOException {
	try {
	    String ack = "";
	    Assert.notFalse(connected);        
	    if (args != null) {
		for (int i = 0; i < args.length; i++) {
		    s += "\3" + args[i];
		}
	    }
	    writeMessage(s);
	    if (waitForAck) {          
		ack = readResponse();
	    } 
	    return ack;
	} catch (IOException e) {
	    this.close();
	    throw e;
	}
    }


    /**
     * Bundle is a helper class that a client can use to bunch together a message
     * and then pass it as a single unit to this socket sender. 
     */
    static public class Bundle {
        public String command;
        public String args[];
        public boolean waitForAck;
        public Bundle(String c, String a[], boolean w) {
            this.command = c;
            this.args = a;
            this.waitForAck = w;
        }
    }


    /**
     * Send a message using the information in b.
     * @return null if not waiting, or ack message if waiting.
     */
    public String sendMessage(SocketSender.Bundle b) throws IOException {
        return this.sendMessage(b.command, b.args, b.waitForAck);
    }
    
}
      
