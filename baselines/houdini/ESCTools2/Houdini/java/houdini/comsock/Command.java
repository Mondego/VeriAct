/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini.comsock;

/**
 * Commands are used to encapsulate the code executed upon receipt
 * of a message from the socket. 
 */
public abstract class Command {
    /**
     * args is list of args to message. args[0] is command, args[1] is first
     * real arg.
     */
    public abstract String doIt(String args[]);
}
