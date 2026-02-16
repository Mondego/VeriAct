// Testing the parsing of the two uses of \max

public class Max {

	//@ requires \max(\lockset) <= this;
	//@ requires ( \max(\lockset) <= this );
	//@ requires ( \max int i; 0<i && i <5; i+1) == 5;
	synchronized public int m();
}
