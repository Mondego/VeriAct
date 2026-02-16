/* Copyright Hewlett-Packard, 2002 */

/** This class implements a consumer
 **/

class Consumer extends Thread {

    ProdConsSync myPCS;

    Consumer(ProdConsSync pcs) {
	this.myPCS = pcs;
    }

    public void run() {
	int d = 0;
	while(true) {
	    d = myPCS.get(); //@ nowarn Null
	    //@ assert d >= 0
	}
    }

}
