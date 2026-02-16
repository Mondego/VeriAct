// Tests recursive use of maps
//#FLAGS: -mapsUnrollCount 5
public class ModChecksMapsA {

	//@ non_null
	public MCMList list = new MCMList();

	//@ modifies list.links;
	public void mm() {
		//@ assume list.next != null;
		//@ assume list.next.next != null;
		//@ assume list.next != list.next.next;
		//@ assume list != list.next.next;
		//@ assume list != list.next;
		list.next.next.next = new MCMList();
		list.next.next = new MCMList();
		list.next = new MCMList();
	}

	//@ modifies list.values;
	public void m() {
		list.value = 0;
		//@ assume list.next != null;
		list.next.value = 1;
		//@ assume list.next.next != null;
		list.next.next.value = 2;
	}

	//@ modifies list.values;
	public void ma() {
		list.array[0] = 0;
		//@ assume list.next != null;
		list.next.array[0] = 0;
		//@ assume list.next.next != null;
		list.next.next.array[0] = 0;
		list.array[1] = 0;  // WARNING
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
