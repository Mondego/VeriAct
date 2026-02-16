/* Copyright 2000, 2001, Compaq Computer Corporation */


package houdini;

import java.io.*;
import houdini.util.Utility;
import houdini.util.Assert;
import houdini.util.ShutDown;
import java.util.*;

public class CheckPoint extends Thread {

    PrintStream outWorker;
    PrintStream outTimes;
    PrintStream outError;
    HoudiniServer server;
    long lastJobReport = 0;
    
    void reportWorker() {
	Vector workers = server.workers;
	StringBuffer sb = new StringBuffer(Utility.getDateString() + ": ");
	StringBuffer jobs = new StringBuffer();
	for (int i = 0; i < workers.size(); i++) {
	    if (i % 4 == 0) sb.append(" ");
	    WorkerState ws = (WorkerState)workers.elementAt(i);
	    switch (ws.getStatus()) {
	    case WorkerState.ALLOCATED:
		sb.append("A");
		break;
	    case WorkerState.CONNECTED:
		sb.append("C");
		break;
	    case WorkerState.BUSY:
		// minor race here:
		sb.append("B");
		Job j = ws.getJob();
		if (j != null) {
		    jobs.append(j.toShortString() + " ");
		} 
		break;
	    case WorkerState.DEAD:
		sb.append("D");
		break;
	    default:
		sb.append("?");
		break;
	    }
	}
	outWorker.println(sb + "    " + jobs + "<BR>");
	outWorker.flush();
    }

    public void run() {
	while (true) {
	    try { Thread.sleep(1000); } catch (Exception e) {}
	    reportWorker();
	    reportNet();
	    reportJob();
	}
    }

    public void reportJob() {
	if (System.currentTimeMillis() - lastJobReport < 5000) {
	    return;
	}
	lastJobReport = System.currentTimeMillis();
	StringBuffer sb = new StringBuffer(10000);
	try {
	    PrintStream outJob = 
		Utility.getPrintStream(server.options().vcdir + "job.html");
	    sb.append("<HTML><TITLE>Houdini Job Log- " + Utility.getDateString() + "</TITLE><BODY>\n");
	    sb.append("WORKLIST: <BR>\n").append(server.list.toHTML()).append("<BR><BR>");
	    sb.append("JOBS: <BR>\n");
	    Vector jobs = server.list.getSortedJobs();
	    for (int i = 0; i < jobs.size(); i++) {
		Job j = (Job)jobs.elementAt(i);
		sb.append(j.historyToShortString() + "<BR>\n");
	    }
	    sb.append("</BODY>");
	    outJob.println(sb);
	    outJob.close();
	} catch (IOException e) {
	    Assert.notify(e);
	    return;
	}

    }

    public void reportNet() {
	try {
	    PrintStream out = 
		Utility.getPrintStream(server.options().vcdir + "net.html");
	    out.println("<HTML><TITLE>Houdini Network Status- " + Utility.getDateString() + "</TITLE><BODY>\n");
	    for (int i = 0; i < server.workers.size(); i++) {
		WorkerState ws = (WorkerState)server.workers.elementAt(i);
		out.println(ws.toString()+"<BR>");
	    }
	    out.close();
	} catch (IOException e) {
	    Assert.notify(e);
	    return;
	}

    }


    public void makeRoot() {
	try {
	    PrintStream out = 
		Utility.getPrintStream(server.options().vcdir + "reports.html");
	    out.println("<HTML><TITLE>Houdini Reports- " + Utility.getDateString() + "</TITLE><BODY>");
	    out.println("<A HREF=\"job.html\">Job Summary<A><BR>");
	    out.println("<A HREF=\"worker.html\">Worker Summary<A><BR>");
	    out.println("<A HREF=\"net.html\">Network Status<A><BR>");
	    out.println("<A HREF=\"times.html\">Time Summary<A><BR>");
	    out.println("<A HREF=\"server.log\">Server Log<A><BR>");
	    out.println("<A HREF=\"error.html\">Error Log<A><BR>");
	    out.println("<A HREF=\"refuted.txt\">Refuted List<A><BR>");
	    out.println("<A HREF=\"files.txt\">File List<A><BR>");
	    for (int i = 0; i < server.workers.size(); i++) {
		WorkerState ws = (WorkerState)server.workers.elementAt(i);
		out.println("<A HREF=\"client."+i+".log\">Client " + i + " Log (" + ws.clientComputer + ")<A><BR>");
	    }
	    out.println("</BODY>");
	    out.close();
	} catch (IOException e) {
	    Assert.notify(e);
	    return;
	}

    }


    public void reportTime(int worker, String s) {
	outTimes.println("[" + Utility.getDateString() + "][id="+worker+"]\t" + "\t" + s + "<br>\n");
	outTimes.flush();
    }

    public void reportError(int worker, Job j, String s) {
	String job;
        if (j == null)
            job = " (no job)";
        else
            job = j.toString();

	outError.println("[" + Utility.getDateString() + "][id="+worker+"] ERROR! Job =" + job + " " + s + "<br>\n");
	outError.flush();
    }


    void finish() {
	this.stop();
	outWorker.println("</BODY>");
	outWorker.close();
	outTimes.println("</BODY>");
	outTimes.close();
	outError.println("</BODY>");
	outError.close();
    }

    CheckPoint(HoudiniServer server) {
	this.server = server;
	String s = Utility.getDateString();
	makeRoot();
	try {
	    outWorker = 
		Utility.getPrintStream(server.options().vcdir + "worker.html");
	    outTimes = 
		Utility.getPrintStream(server.options().vcdir + "times.html");
	    outError = 
		Utility.getPrintStream(server.options().vcdir + "error.html");
       	} catch (IOException e) {
	    Assert.fail(e);
	}
	outWorker.println("<HTML><TITLE>Houdini Worker Log- " + s + "</TITLE><BODY>");
	outTimes.println("<HTML><TITLE>Houdini Time Log- " + s + "</TITLE><BODY>");
	outError.println("<HTML><TITLE>Houdini Error Log- " + s + "</TITLE><BODY>");
	
	this.start();
	ShutDown.runOnExit(new Runnable() {
	    public void run() { finish(); }
	});
    }
}
