/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.comsock;

import java.io.*;
import java.net.*;

/**
 * part of a test program. 
 */
class Main {   
    static SocketCommand sc = new SocketCommand(9999);
    static SocketSender ss = new SocketSender(1111);
    static SocketCommand mic;

    public static void main(String st[]) throws java.net.UnknownHostException, java.io.IOException {        
        sc.addCommand("echo", new Command() {
            public String doIt(String args[]) {
                for (int i = 0; i < args.length; i++) {
                    System.out.print(args[i]);
                } 
                System.out.println("");
                return "ack";
            }
        });
        sc.addCommand("MicLevel", new Command() {
            public String doIt(String args[]) {
                int n = Integer.parseInt(args[1]);
                System.out.println("Mic Level = " + n);
                return "ack";
            }
        });        
        sc.start();
        while (true) {
            try {
                Thread.sleep(3000);
//                ss.connect();
            } catch (Exception e) {
                System.out.println(e);
            }
        }
    }
}
