// Test some misues of monitored declarations

public class MonitoredErrors {

	//@ monitored_by o;
	static int i;

	Object o = new Object();

	//@ monitors_for i = o;

	int j;

	//@ monitors_for o = i;
}
