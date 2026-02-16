class FindInArray {
    private  int key;
    private  int arr[];

    FindInArray(int inputArr[])
    {
	int size = inputArr.length;
	arr = new int[size];
	arr = inputArr.clone();
    } 
 
    FindInArray(int inputArr[], int key)
    {
	int size = inputArr.length;
	arr = new int[size];
	arr = inputArr.clone();
        setKey(key);
    } 

    void setKey(int key) 
    {
	this.key = key;
    }

     int getKey() 
    {
	return this.key;
    }

     int getArr(int i) 
    {
	return this.arr[i];
    }

     int size() 
    {
   	return arr.length;
    }

    
     int findLast() {
	int index = size() - 1;
	
	
	while (0 <= index) {
		if (getArr(index) == getKey())
			return index;
		index--;
	}
	return -1;
    }

    
     int findFirst() {	
	
	
	for (int index = 0; index < size(); index++) {
		if (getArr(index) == getKey())
			return index;
	}
	return -1;
    }
    
    
     boolean isMoreThanOneKey() {
	int first = findFirst();
	int last = findLast();
	return (first != last);
    }
}
