public class SwapInArray {
        
	public void swap(int x, int y,  int array[]) {
	   int temp;     
      temp = array[x];
      array[x] = array[y];
      array[y] = temp;
   }
}
