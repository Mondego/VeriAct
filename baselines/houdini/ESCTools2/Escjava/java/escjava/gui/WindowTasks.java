package escjava.gui;

import java.util.LinkedList;

public class WindowTasks implements Runnable {

	public void run() {

	    while (true) {
		Object o = getTask();
		if (o == null) { Thread.currentThread().yield(); continue; }
		// Do something with task

	    }

	}

    static private /*@non_null*/ LinkedList tasks = new LinkedList();

	synchronized static public void addTask(Object o) {
		tasks.addLast(o);
	}

	synchronized static private Object getTask() {
		if (tasks.isEmpty()) return null;
		return tasks.removeFirst();
	}

}
