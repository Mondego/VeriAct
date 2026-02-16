// Tests recursive use of maps

public class ModChecksMaps {

	//@ non_null
	public MCMList list = new MCMList();

	//@ modifies list.links;
	public void mm() {
		//@ assume list.next != null;
		//@ assume list.next.next != null;
		//@ assume list.next != list.next.next;
		//@ assume list != list.next.next;
		//@ assume list != list.next;
		list.next.next.next = new MCMList();  // OK - except beyond unrolling
		list.next.next = new MCMList();       // OK
		list.next = new MCMList();            // OK
	}

	//@ modifies list.values;
	public void m() {
		list.value = 0;                      // OK
		//@ assume list.next != null;
		list.next.value = 1;                 // OK
		//@ assume list.next.next != null;
		list.next.next.value = 2;            // OK -- beyond unrolling?
	}

	//@ modifies list.values;
	public void ma() {
		list.array[0] = 0;                    // OK
		//@ assume list.next != null;
		list.next.array[0] = 0;               // OK
		//@ assume list.next.next != null;
		list.next.next.array[0] = 0;          // OK - beyond unrolling
		list.array[1] = 0;                    // WARNING
	}

	//@ modifies list.values;
	public void mloop() {
		MCMList it = list;
		while (it != null) {
			it.value ++;                  // OK
			it = it.next;
		}
	}
}


class MCMList {

	//@ public model Object values;
	//@ public model Object links;

	public int value; //@ in values;

	//@ non_null
        public int[] array = new int[20]; //@ in values;
			    //@ maps array[0] \into values;
	//@ public invariant array.length > 10;

	public MCMList next; //@ in links;
		//@ maps next.links \into links;
		//@ maps next.values \into values;

	//@ modifies \nothing;
	public MCMList();
}
