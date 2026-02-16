/*  Copyright 2004, David R. Cok 
    Originally generated as part of a GUI interface to the
    Esc/Java2 application.
*/

package escjava.gui;

import java.awt.Frame;
import java.awt.EventQueue;

/** The FrameShower is used to be sure that a Frame is shown 
    through a call that is made on the Event thread.  
    This is generally advised in order to avoid race conditions
    in the GUI painting itself.  Use this class by calling the
    static method to queue a new instance of an object to be
    shown.
 */
public class FrameShower implements Runnable {

    //@ spec_public
    final private /*@ non_null */ Frame frame;

    /** Creates a FrameShower object holding the given Frame.
     */
    //@ requires frame != null;
    //@ also ensures this.frame == frame;
    //@ pure
    protected FrameShower(Frame frame) {
	this.frame = frame;
    }

    /** Called by the EventQueue as the task to be
        accomplished - it shows/makes visible the 
        given frame.
     */
    // Cannot call this pure because it certainly changes the GUI.
    //@ also ensures \not_modified(frame);
    public void run() {
	frame.show();
    }

    // This is going to need some kind of modifies clause - it
    // modifies the event logic and the gui at least.
    //@ requires frame != null;
    static void show(Frame frame) {
        EventQueue.invokeLater(new FrameShower(frame));
    }
}


