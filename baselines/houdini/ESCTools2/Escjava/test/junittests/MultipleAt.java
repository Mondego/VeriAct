// Tests that multiple @ signs are allowed.
//#FLAGS: -parsePlus -classpath .
public class MultipleAt {

	//@ ghost public int i;

	/*@@@@@ requires i == 0;
	  @@@@@ ensures i == 0;
	  @@@@@*/
	public void m() {}

	/*+@@@@@ requires i == 0;
	  @@@@@ ensures i == 0;
	  @@@@@+*/
	public void q() {}

	//@@@@@ ghost int j;
	//+@@@@@ghost int k;

}
