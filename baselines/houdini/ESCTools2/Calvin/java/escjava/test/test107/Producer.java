/* Copyright Hewlett-Packard, 2002 */

/** This class implements a producer
 **/

class Producer extends Thread {

    ProdConsSync myPCS;

    Producer(ProdConsSync pcs) {
	this.myPCS = pcs;
    }

    public void run() {
	int d = 0;
	while(true) {
	    myPCS.put(d); //@ nowarn Null
	    d++;
	}
    }

}
