package escjava.gui;

public class TaskQueue {

    private /*@ non_null */ java.util.LinkedList tasks = new java.util.LinkedList();

    synchronized public void addTask(/*@ non_null */ Object o) { 
	tasks.addLast(o); 
	notifyAll();
    }

    synchronized public /*@ non_null */ Object getTask() { 
	while (tasks.isEmpty()) {
	    try { wait(); } catch (Exception e) {}
	}
	return tasks.removeFirst(); 
    }

    synchronized public void clear() {
	tasks.clear();
    }
}
