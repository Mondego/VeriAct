public class Static {

	int i;
	static int si;

	//@ invariant i >0;
	//@ invariant si > 0;
	//@ static invariant i > 0; // ERROR
	//@ static invariant si > 0;

	//@ constraint i == \old(i);
	//@ constraint si == \old(si);
	//@ static constraint i == \old(i); // ERROR
	//@ static constraint si == \old(si);
	
	//@ model int m;
	//@ static model int sm;
	//@ represents m = i;
	//@ represents sm = i;  // ERROR
	//@ static represents m = si; // ERROR
	//@ static represents sm = si;

	//@ initially i == 0;
	//@ initially si == 0;
	//@ static initially i == 0; // ERROR
	//@ static initially si == 0;

	//@ monitors_for i = this;
	//@ private monitors_for i = this; 
	//@ monitors_for si = this; // ERROR - NOT IN JML
	//@ static monitors_for i = this;  // ERROR
	//@ static monitors_for si = Static.class; 

}

