// This is a test case from Bug#244
//#FLAGS: -Q

public class HelperWithBody
{
	public final /*@non_null*/ Object[][] grid;
	public final /*@non_null*/ Object[]   rows;

	public HelperWithBody()
	{
		grid=new Object[2][2];
		rows=initializeRows();
	}

    /*@ public normal_behavior 
      @   requires true;
      @   assignable \everything;
      @   ensures true;
      @   diverges false;
      @*/
    /*@spec_public helper*/private /*@non_null*/Object[] initializeRows() 
	{
		Object[] retVal=new Object[grid.length];
		for (int i = 0; i < grid.length; i++) 
			retVal[i]=initializeRow(grid[i]);
		return retVal;
	}
    /*@ public normal_behavior 
      @   requires true;
      @   assignable \nothing;
      @   ensures true;
      @   diverges false;
      @*/
    public Object[] initializeRow(Object[] o)
    {return o;}
}
