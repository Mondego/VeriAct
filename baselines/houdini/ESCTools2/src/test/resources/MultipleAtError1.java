// Tests that multiple @ signs are allowed.
// jml fails on the multiple ats that terminate the comment
//#FLAGS: -parsePlus -classpath .
public class MultipleAtError1 {

	//@ ghost public int i;

	/*+@@@@@ requires i == 0;
	  @@@@@ ensures i == 0;  @@@@+*/
	public void p() {}

}
