public class BubbleSort { 
    
    int[] bubbleSort(int arr[]) { 
	SwapInArray s = new SwapInArray();
        int n = arr.length;

        for (int i = 0; i < n-1; i++) {	

            for (int j = 0; j < n-i-1; j++) {
                if (arr[j+1] < arr[j]) {  
		            s.swap(j, j + 1, arr); 
                } 
	        }
	    } 
	    return arr;
    } 
}