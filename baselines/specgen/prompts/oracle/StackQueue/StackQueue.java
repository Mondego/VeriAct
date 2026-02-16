public class StackQueue {
	public class Stack { 
	 	public static final int MAX = 100;   // Maximum size of Stack 
	
	   	 private /*@ spec_public @*/ int top;
	   	 //@ public invariant -1 <= top && top < MAX;

	    	private /*@ spec_public @*/ int arr[] = new int[MAX];
	    	//@ public invariant arr.length == MAX; 
	
	    	//@ ensures top == -1;
	   	/*@ pure @*/ Stack() 
	   	{ 
	        	top = -1; 
	    	} 

	    	//@ public normal_behavior
	    	//@   ensures top == \result;
	   	public /*@ pure @*/ int getTop() 
	    	{
	        	return top;
	   	}

	   	//@ public normal_behavior
	    	//@   ensures \result <==> top < 0;
	    	//@   ensures_redundantly !\result ==> 0 <= top;
	    	public /*@ pure @*/ boolean isEmpty() 
	    	{ 
	        	return (getTop() < 0); 
	    	} 

	    	//@ public normal_behavior
	    	//@ 	ensures \result <==> top == MAX - 1;
	    	//@   	ensures_redundantly !\result ==> top < MAX - 1;
	    	public /*@ pure @*/ boolean isFull() 
	    	{ 
	        	return (top == MAX - 1); 
	    	} 

	    	/*@ public normal_behavior
	      	  @ 	requires !isFull();
	          @ 	assignable top, arr[*];
	          @ 	ensures 0 <= top && top < MAX;
	          @ 	ensures arr[top] == x;
	          @ 	ensures top == \old (top + 1);
	          @ 	ensures \forall int i; 0 <= i && i <= arr.length && i != top; arr[i] == \old (arr[i]);
	          @ 	ensures_redundantly 0 <= top && top < MAX; 

	          @ also
		
	          @ public exceptional_behavior
	          @	requires isFull();
	          @ 	assignable \nothing;
	          @     signals_only IllegalArgumentException; 
	      	@*/
	    	public void push(int x) 
	    	{ 
	       	 	if (!isFull()) {
	           		 arr[++top] = x; 
	       		 } else {
	            		throw new IllegalArgumentException();
	       		 }
	    	} 
	
	   	 /*@ public normal_behavior
	     	   @ 	requires !isEmpty();
	     	   @ 	assignable top;
	      	   @ 	ensures top == \old (top - 1);
	      	   @ 	ensures \result == \old (arr[top]);
	      	   @ 	ensures \forall int i; 0 <= i && i <= arr.length; arr[i] == \old (arr[i]);

     	      	   @ also

      	           @ public exceptional_behavior
     	      	   @ 	requires isEmpty();
      	      	   @ 	assignable \nothing;
      	      	   @    signals_only IllegalArgumentException; 	
      	    	@*/
   	    	public int pop() 
    	    	{ 
       	 		if (!isEmpty()) {
            			return arr[top--]; 
        		 } else {
            			throw new IllegalArgumentException();
       			 }
    	    	}
    
    	   	/*@ public normal_behavior
     	     	  @ 	requires !isEmpty();
      	     	  @ 	ensures \result == arr[top];

      	     	  @ also

      	     	  @ public exceptional_behavior
      	     	  @ 	requires isEmpty();
      	     	  @     signals_only IllegalArgumentException; 	
      	  	@*/
    	   	public /*@ pure @*/ int peek() 
    	   	{ 
        		if (!isEmpty())
            			return arr[top]; 
        		else
            			throw new IllegalArgumentException();
   	   	} 

    	  	//@ public normal_behavior
    	   	//@ 	ensures 0 <= \result && \result <= top ==> arr[\result] == key;
    	   	//@ 	ensures \result == -1 ==> \forall int i; 0 <= i && i <= top; arr[i] != key; 
    	   	public /*@ pure @*/ int search(int key) 
    	   	{
        		int index = top;
        		//@ maintaining -1 <= index && index <= top; 
        		//@ maintaining (\forall int i; index < i && i <= top; arr[i] != key);
        		while (0 <= index) {
        		    if (getElem(index) == key) {
        		        return index;
        		    }
        		    index--;
        		}
        		return -1;
    	   	}	

    	   	//@ public normal_behavior
    	   	//@ 	ensures \result ==> \exists int i; 0 <= i && i <= top; arr[i] == key;
    	   	//@ 	ensures !\result ==> \forall int i; 0 <= i && i <= top; arr[i] != key;
    	   	public /*@ pure @*/ boolean isContain(int key)
    	   	{	
        		int index = top;
        		//@ maintaining -1 <= index && index <= top; 
        		//@ maintaining (\forall int i; index < i && i <= top; arr[i] != key);
        		while (0 <= index) {
        		    if (key == getElem(index)) {
        		        return true;
        		    }
        		    index--;
        		}
        		return false;
    	   	}	

    	   	//@ public normal_behavior
    	   	//@ 	ensures \result == top + 1;
    	   	public /*@ pure @*/ int size() 
    	   	{ 
        		return getTop() + 1;
    	   	}

    	   	//@ public normal_behavior
    	   	//@ 	requires 0 <= i && i <= top;
    	   	//@ 	ensures \result == arr[i];
    	   	public /*@ pure @*/ int getElem(int i) 
    	   	{ 
        		return arr[i]; 
    	   	}
	};

	public class Queue { 
    		public static final int MAX = 100;   // Maximum size of Queue 
    		/*@ spec_public @*/ private int front, rear; 
    		/*@ spec_public @*/ private final int queue[] = new int[MAX];
    		//@ public invariant 0 <= rear && rear <= MAX;
    		//@ public invariant front == 0;
    		//@ public invariant queue.length == MAX;

    		//@ ensures front == 0 && rear == 0;
    		public /*@ pure @*/ Queue() 
    		{ 
       			 front = rear = 0;  
   		} 

    		/*@ public normal_behavior
     		  @ 	requires 0 <= rear && rear < MAX;
     		  @ 	requires !isFull();
      	   	  @ 	assignable queue[*], rear;
      		  @ 	ensures \old (rear) == rear - 1;
      		  @ 	ensures queue[rear-1] == data;

      		  @ also

      		  @      requires isFull();
		  @	 assignable \nothing;
      		  @      signals_only IllegalArgumentException; @*/
   		public void enter(int data) 
    		{ 
       			if (!isFull()) { 
            			queue[rear] = data; 
            			rear++; 
        		} else { 
            			throw new IllegalArgumentException(); 
        		} 
    		} 

    		/*@ public normal_behavior
      		  @ 	requires !isEmpty();
      		  @ 	old int tempQ[] = queue.clone();
     	 	  @ 	assignable queue[*], rear;
      		  @	ensures rear == \old (rear - 1);
      		  @ 	ensures \result == \old (queue[front]);
      		  @ 	ensures (\forall int i; 0 <= i && i < rear; queue[i] == tempQ[i+1]);

      		  @ also

      		  @ public exceptional_behavior
     		  @ 	requires isEmpty();
		  @	assignable \nothing;
      		  @     signals_only IllegalArgumentException; @*/
    		public int delete() 
    		{ 
        		if (!isEmpty()) { 
            			int poll = queue[front];
            			//@ ghost int temp[] = queue.clone();
            			int i = 0;
            			/*@ maintaining (\forall int j; 0 <= j && j < i; queue[j] == temp[j+1]);
              		  	@ maintaining 0 <= i && i < rear;
              			@*/
            			while (i < rear - 1) {
                			//@ assume queue[i+1] == temp[i+1];
                			queue[i] = queue[i + 1];
                			i++;
            			}
            			rear--;
            			return poll; 
        		} else {
            			throw new IllegalArgumentException();
        		}
    		} 
		
    		/*@ public normal_behavior
      		  @ 	requires !isEmpty();
      		  @ 	ensures \result == queue[front];

      		  @ also

      		  @ public exceptional_behavior
      		  @ 	requires isEmpty();
      		  @     signals_only IllegalArgumentException; 	
      		@*/
    		public /*@ pure @*/ int peek() 
    		{ 
        		if (!isEmpty()) { 
            			return queue[front]; 
        		} else {
            			throw new IllegalArgumentException(); 
        		}
    		} 

   	 	//@ public normal_behavior
    		//@   ensures \result ==> \exists int i; 0 <= i && i < rear; queue[i] == key;
    		//@   ensures !\result ==> \forall int i; 0 <= i && i < rear; queue[i] != key;
    		public /*@ pure @*/ boolean isContain(int key)
    		{
        		int index = 0;
        		//@ maintaining 0 <= index && index <= rear;
        		//@ maintaining \forall int i; 0 <= i && i < index; queue[i] != key;
        		while (index < rear) {
        		 	if (key == queue[index]) {
        		 	       return true;
            			}
            		index++;
    		    	}
        	return false;
    		}

    		//@ public normal_behavior
    		//@   ensures 0 <= \result && \result < rear ==> queue[\result] == key;
    		//@   ensures \result == -1 ==> \forall int i; 0 <= i && i < rear; queue[i] != key;
    		public /*@ pure @*/ int search(int key)
    		{
        		int index = 0;
        		//@ maintaining 0 <= index && index <= rear;
        		//@ maintaining \forall int i; 0 <= i && i < index; queue[i] != key;
        		while (index < rear) {
            			if (key == queue[index]) {
                			return index;
            			}
            		index++;
        		}
        	return -1;
    		}

    		//@ public normal_behavior
   		//@   ensures \result <==>  rear == front;
   		public /*@ pure @*/ boolean isEmpty() 
   		{
   		     	return (getRear() == getFront());
    		}

    		//@ public normal_behavior
    		//@   ensures \result <==>  MAX == rear;
    		public /*@ pure @*/ boolean isFull() 
    		{
    		    	if (MAX == getRear()) 
    		        	return true;
    		    	else
    		        	return false;
    		}

    		//@ public normal_behavior
    		//@   ensures rear == \result;
    		public /*@ pure @*/ int size()
    		{
    		    	return rear;
    		}

  		//@ public normal_behavior
  		//@   ensures \result == front;
   		public /*@ pure @*/ int getFront() 
    		{	
       			return front; 
   		}

    		//@ public normal_behavior
    		//@   ensures \result == rear;
    		public /*@ pure @*/ int getRear() 
    		{ 
        		return rear; 
    		}

    		//@ public normal_behavior
    		//@   requires 0 <= i && i < rear;
    		//@   ensures \result == queue[i];
    		public /*@ pure @*/ int getElem(int i) 
    		{ 
        		return queue[i]; 
    		}
	}; 

    	/*@ requires 1 <= stack.top;
      	  @ requires Integer.MIN_VALUE <= stack.getElem(stack.top) + stack.getElem(stack.top - 1);
     	  @ requires stack.getElem(stack.top) + stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
      	  @ assignable stack.top, stack.arr[*];
      	  @ ensures \result == \old (stack.getElem(stack.top) + stack.getElem(stack.top - 1));
      	  @ ensures stack.size() == \old (stack.size() - 1);
    	@*/
    	public int stackPlus(Stack stack) 
    	{
		stack.push(stack.pop() + stack.pop());
		return stack.peek();
    	}

    	/*@ requires 1 <= stack.getTop();
      	  @ requires Integer.MIN_VALUE <= stack.getElem(stack.top) - stack.getElem(stack.top - 1);
          @ requires stack.getElem(stack.top) - stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
          @ assignable stack.top, stack.arr[*];
          @ ensures \result == \old (stack.getElem(stack.top) - stack.getElem(stack.top - 1));
          @ ensures stack.size() == \old (stack.size() - 1);
    	@*/
    	public int stackMinus(Stack stack) 
    	{
		stack.push(stack.pop() - stack.pop());
		return stack.peek();
    	}

    	/*@ requires 1 <= stack.top;
      	  @ requires stack.getElem(stack.top - 1) != 0;
      	  @ requires Integer.MIN_VALUE <= stack.getElem(stack.top) / stack.getElem(stack.top - 1);
      	  @ requires stack.getElem(stack.top) / stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
      	  @ assignable stack.top, stack.arr[*];
      	  @ ensures \result == \old (stack.getElem(stack.top) / stack.getElem(stack.top - 1));
      	  @ ensures stack.size() == \old (stack.size() - 1);
    	@*/
    	public int stackDivision(Stack stack) 
    	{
		stack.push(stack.pop() / stack.pop());
		return stack.peek();
    	}

    	/*@ requires 1 <= stack.top && stack.getElem(stack.top - 1) != 0;
      	  @ requires Integer.MIN_VALUE <= stack.getElem(stack.top) % stack.getElem(stack.top - 1);
      	  @ requires stack.getElem(stack.top) % stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
      	  @ assignable stack.top, stack.arr[*];
      	  @ ensures \result == \old (stack.getElem(stack.top) % stack.getElem(stack.top - 1));
      	  @ ensures stack.size() == \old (stack.size() - 1);
    	@*/
    	public int stackModulus(Stack stack) 
    	{
		stack.push(stack.pop() % stack.pop());
 		return stack.peek();
    	}

    	/*@ requires 2 <= Q.rear;
          @ requires Integer.MIN_VALUE <= Q.getElem(Q.front) + Q.getElem(Q.front + 1);
      	  @ requires Q.getElem(Q.front) + Q.getElem(Q.front + 1)  <= Integer.MAX_VALUE;
      	  @ assignable Q.rear, Q.queue[*];
      	  @ ensures \result == \old (Q.getElem(Q.front) + Q.getElem(Q.front + 1));
      	  @ ensures Q.size() == \old (Q.size() - 1);
    	@*/
    	public int QPlus(Queue Q) 
    	{
		Q.enter(Q.delete() + Q.delete());
		return Q.getElem(Q.getRear() - 1);
    	}

    	/*@ requires 2 <= Q.rear;
      	  @ requires Integer.MIN_VALUE <= Q.getElem(Q.front) - Q.getElem(Q.front + 1);
      	  @ requires Q.getElem(Q.front) - Q.getElem(Q.front + 1)  <= Integer.MAX_VALUE;
          @ assignable Q.rear, Q.queue[*];
          @ ensures \result == \old (Q.getElem(Q.front) - Q.getElem(Q.front + 1));
          @ ensures Q.size() == \old (Q.size() - 1);
    	@*/
    	public int QMinus(Queue Q) 
    	{
		Q.enter(Q.delete() - Q.delete());
		return Q.getElem(Q.getRear() - 1);
    	}

    	/*@ requires 2 <= Q.rear && Q.getElem(Q.front + 1) != 0;
      	  @ requires Integer.MIN_VALUE <= Q.getElem(Q.front) / Q.getElem(Q.front + 1);
      	  @ requires Q.getElem(Q.front) / Q.getElem(Q.front + 1)  <= Integer.MAX_VALUE;
      	  @ assignable Q.rear, Q.queue[*];
      	  @ ensures \result == \old (Q.getElem(Q.front) / Q.getElem(Q.front + 1));
      	  @ ensures Q.size() == \old (Q.size() - 1);
    	@*/
    	public int QDivision(Queue Q) 
    	{
		Q.enter(Q.delete() / Q.delete());
		return Q.getElem(Q.getRear() - 1);
    	}

    	/*@ requires 2 <= Q.rear && Q.getElem(Q.front + 1) != 0;
      	  @ requires Integer.MIN_VALUE <= Q.getElem(Q.front) % Q.getElem(Q.front + 1);
      	  @ requires Q.getElem(Q.front) % Q.getElem(Q.front + 1)  <= Integer.MAX_VALUE;
          @ assignable Q.rear, Q.queue[*];
          @ ensures \result == \old (Q.getElem(Q.front) % Q.getElem(Q.front + 1));
          @ ensures Q.size() == \old (Q.size() - 1);
        @*/
        public int QModulus(Queue Q) 
    	{
		Q.enter(Q.delete() % Q.delete());
		return Q.getElem(Q.getRear() - 1);
    	}	
	

    	/*@ requires 1 <= Q.rear && 1 <= stack.top;
      	  @ requires Q.peek() + stack.peek() <= Integer.MAX_VALUE;
          @ requires Integer.MIN_VALUE <= Q.peek() + stack.peek();
          @ ensures \result == Q.peek() + stack.peek();
    	@*/

    	public /*@ pure @*/ int plusQStack(Queue Q, Stack stack) 
    	{
		return Q.peek() + stack.peek();
    	}

    	/*@ requires 1 <= Q.rear && 1 <= stack.top;
      	  @ requires Q.peek() - stack.peek() <= Integer.MAX_VALUE;
      	  @ requires Integer.MIN_VALUE <= Q.peek() - stack.peek();
      	  @ ensures \result == Q.peek() - stack.peek();
    	@*/
    	public /*@ pure @*/ int minusQStack(Queue Q, Stack stack) 
    	{
		return Q.peek() - stack.peek();
    	}

    	/*@ requires 1 <= Q.rear && 1 <= stack.top;
      	  @ requires stack.peek() != 0;
          @ requires Q.peek() / stack.peek() <= Integer.MAX_VALUE;
          @ requires Integer.MIN_VALUE <= Q.peek() / stack.peek();
          @ ensures \result == Q.peek() / stack.peek();
    	@*/
    	public /*@ pure @*/ int qDivideStack(Queue Q, Stack stack) 
    	{
		return Q.peek() / stack.peek();
    	}

    	/*@ requires 1 <= Q.rear && 1 <= stack.top;
      	  @ requires Q.peek() != 0;
      	  @ requires stack.peek() / Q.peek() <= Integer.MAX_VALUE;
      	  @ requires Integer.MIN_VALUE <= stack.peek() / Q.peek();
      	  @ ensures \result == stack.peek() / Q.peek();
    	@*/
    	public /*@ pure @*/ int stackDivideQ(Queue Q, Stack stack) 
    	{
		return  stack.peek()/Q.peek();
    	}	

       /*@ requires 1 <= Q.rear && 1 <= stack.top;
         @ requires stack.peek() != 0;
         @ requires Q.peek() % stack.peek() <= Integer.MAX_VALUE;
         @ requires Integer.MIN_VALUE <= Q.peek() % stack.peek();
         @ ensures \result == Q.peek() % stack.peek();
       @*/
    	public /*@ pure @*/ int qModulusStack(Queue Q, Stack stack) 
    	{
		return Q.peek() % stack.peek();
    	}

    	/*@ requires 1 <= Q.rear && 1 <= stack.top;
      	  @ requires Q.peek() != 0;
      	  @ requires stack.peek() % Q.peek() <= Integer.MAX_VALUE;
      	  @ requires Integer.MIN_VALUE <= stack.peek() % Q.peek();
      	  @ ensures \result == stack.peek() % Q.peek();
    	@*/
    	public /*@ pure @*/ int stackModulusQ(Queue Q, Stack stack) 
    	{
		return  stack.peek() % Q.peek();
    	}
	
	/*@ requires 0 <= op && op < 9;
    	  @ {|
	  @ 	requires op == 0 && !stack.isFull();
	  @	assignable stack.top, stack.arr[*];
	  @ 	ensures stack.peek() == input && stack.top == \old (stack.top + 1) && \result == 0;
	  @ also
	  @	requires op == 1 && !stack.isEmpty();
	  @	assignable stack.top, stack.arr[stack.top+1];
	  @	ensures \result == \old (stack.getElem(stack.top));
	  @	ensures stack.top == \old (stack.top - 1);
	  @ also
	  @	requires op == 2;
	  @	ensures 0 <= \result && \result <= stack.top ==> input == stack.getElem(\result);
	  @	ensures \result == -1 ==>  \forall int i; 0 <= i && i <= stack.top; stack.arr[i] != input;
	  @ also
	  @	requires op == 3;
	  @	ensures \result == 1 ==> \exists int i; 0 <= i && i <= stack.top; stack.arr[i] == input;
	  @	ensures \result == 0 ==> \forall int i; 0 <= i && i <= stack.top; stack.arr[i] != input;
	  @ also
	  @ 	requires op == 4 && 1 <= stack.top;
	  @ 	requires Integer.MIN_VALUE <= stack.getElem(stack.top) + stack.getElem(stack.top - 1);
	  @ 	requires stack.getElem(stack.top) + stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
	  @	assignable stack.top, stack.arr[*];
	  @	ensures \result == \old (stack.getElem(stack.top) + stack.getElem(stack.top - 1));
	  @	ensures stack.top == \old (stack.top - 1);
	  @ also
	  @ 	requires op == 5 && 1 <= stack.top;
	  @ 	requires Integer.MIN_VALUE <= stack.getElem(stack.top) - stack.getElem(stack.top - 1);
	  @ 	requires stack.getElem(stack.top) - stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
	  @	assignable stack.top, stack.arr[*];
	  @	ensures \result == \old (stack.getElem(stack.top) - stack.getElem(stack.top - 1));
	  @	ensures stack.top == \old (stack.top - 1);
	  @ also
	  @ 	requires op == 6 && 1 <= stack.top;
	  @ 	requires stack.getElem(stack.top - 1) != 0;
	  @ 	requires Integer.MIN_VALUE <= stack.getElem(stack.top) / stack.getElem(stack.top - 1);
	  @ 	requires stack.getElem(stack.top) / stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
	  @	assignable stack.top, stack.arr[*];
	  @	ensures \result == \old (stack.getElem(stack.top) / stack.getElem(stack.top - 1));
	  @	ensures stack.top == \old (stack.top - 1);		
	  @ also
	  @ 	requires op == 7 && 1 <= stack.top && stack.getElem(stack.top - 1) != 0;
	  @ 	requires Integer.MIN_VALUE <= stack.getElem(stack.top) % stack.getElem(stack.top - 1);
	  @ 	requires stack.getElem(stack.top) % stack.getElem(stack.top - 1)  <= Integer.MAX_VALUE;
	  @ 	assignable stack.top, stack.arr[*];
	  @ 	ensures \result == \old (stack.getElem(stack.top) % stack.getElem(stack.top - 1));
	  @ 	ensures stack.size() == \old (stack.size() - 1);
	  @ also
	  @ 	requires op == 8;
	  @	ensures \result == stack.top + 1;
        @ |}	
    	@*/
    	public int driverStack(Stack stack, int op, int input) 
    	{
		int output = 0;
		switch (op) {
          		case 0:
            		stack.push(input);
			break;

			case 1:
            		output = stack.pop();
			break;

			case 2:
            		output = stack.search(input);
			break;

			case 3:
            		output = (stack.isContain(input)) ? 1 : 0;
			break;

			case 4:
            		output = stackPlus(stack);
			break;

			case 5:
            		output = stackMinus(stack);
               		 break;

			case 6:
            		output = stackDivision(stack);
               		break;

			case 7:
			output = stackModulus(stack);
			break;

			default:
            		output = stack.size();
			break;
	 	}
		return output;
    	}

    	 /*@ requires 0 <= op && op < 9;
       	   @ {|
	   @ 	requires op == 0 && 0 <= q.rear && q.rear < q.MAX;
	   @ 	requires !q.isFull(); 
	   @ 	assignable q.queue[*], q.rear;
	   @	ensures \old (q.rear) == q.rear - 1;
	   @	ensures q.queue[q.rear-1] == input && \result == 0;	
	   @ also
	   @ 	requires op == 1 && !q.isEmpty();
           @ 	old int tempQ[] = q.queue.clone();
	   @ 	assignable q.queue[*], q.rear;
	   @ 	ensures q.rear == \old (q.rear - 1);
	   @ 	ensures \result == \old (q.queue[q.front]);
	   @ 	ensures (\forall int i; 0 <= i && i < q.rear; q.queue[i] == tempQ[i+1]);
	   @ also
	   @	requires op == 2;
	   @ 	ensures 0 <= \result && \result < q.rear ==> q.queue[\result] == input;
	   @ 	ensures \result == -1 ==> \forall int i; 0 <= i && i < q.rear; q.queue[i] != input;
	   @ also
	   @ 	requires op == 3;
	   @ 	ensures \result == 1 ==> \exists int i; 0 <= i && i < q.rear; q.queue[i] == input;
	   @ 	ensures \result == 0 ==> \forall int i; 0 <= i && i < q.rear; q.queue[i] != input;
	   @ also
	   @ 	requires op == 4 && 2 <= q.rear;
	   @ 	requires Integer.MIN_VALUE <= q.getElem(q.front) + q.getElem(q.front + 1);
	   @ 	requires q.getElem(q.front) + q.getElem(q.front + 1)  <= Integer.MAX_VALUE;
	   @ 	assignable q.rear, q.queue[*];
	   @ 	ensures \result == \old (q.getElem(q.front) + q.getElem(q.front + 1));
	   @ 	ensures q.size() == \old (q.size() - 1);
	   @ also
	   @	requires op == 5 && 2 <= q.getRear();
	   @ 	requires Integer.MIN_VALUE <= q.getElem(q.front) - q.getElem(q.front + 1);
	   @ 	requires q.getElem(q.front) - q.getElem(q.front + 1)  <= Integer.MAX_VALUE;
	   @ 	assignable q.rear, q.queue[*];
	   @ 	ensures \result == \old (q.getElem(q.front) - q.getElem(q.front + 1));
	   @ 	ensures q.size() == \old (q.size() - 1);
	   @ also
	   @	requires op == 6 && 2 <= q.rear && q.getElem(q.front + 1) != 0;
	   @ 	requires Integer.MIN_VALUE <= q.getElem(q.front) / q.getElem(q.front + 1);
	   @ 	requires q.getElem(q.front) / q.getElem(q.front + 1)  <= Integer.MAX_VALUE;
	   @ 	assignable q.rear, q.queue[*];
	   @ 	ensures \result == \old (q.getElem(q.front) / q.getElem(q.front + 1));
	   @ 	ensures q.size() == \old (q.size() - 1);
	   @ also
	   @	requires op == 7 && 2 <= q.rear && q.getElem(q.front + 1) != 0;
	   @ 	requires Integer.MIN_VALUE <= q.getElem(q.front) % q.getElem(q.front + 1);
	   @ 	requires q.getElem(q.front) % q.getElem(q.front + 1)  <= Integer.MAX_VALUE;
	   @ 	assignable q.rear, q.queue[*];
	   @ 	ensures \result == \old (q.getElem(q.front) % q.getElem(q.front + 1));
	   @ 	ensures q.size() == \old (q.size() - 1);
	   @ also
	   @ 	requires op == 8;
	   @	ensures \result == q.rear;	
         @ |}
   	 @*/
   	 public int driverQueue(Queue q, int op, int input)
    	 {
		int output = 0;
		switch (op) {
			case 0:
           		q.enter(input);
                	break;

			case 1:
            		output = q.delete();
                	break;

			case 2:
            		output = q.search(input);
                	break;

			case 3:
            		output = q.isContain(input) ? 1 : 0;
               		break;

			case 4:
            		output = QPlus(q);
                	break;

			case 5:
            		output = QMinus(q);
                	break;

			case 6:
            		output = QDivision(q);
                	break;

			case 7:
			output = QModulus(q);
			break;

			default:
            		output = q.size();
                	break;
		}
		return output;
    	}

    	/*@ requires 0 <= op && op < 6;
      	  @ {|
	  @	requires op == 0 && 1 <= q.rear && 1 <= stack.top;
	  @ 	requires q.peek() + stack.peek() <= Integer.MAX_VALUE;
	  @ 	requires Integer.MIN_VALUE <= q.peek() + stack.peek();
	  @ 	ensures \result == q.peek() + stack.peek();
	  @ also
	  @	requires op == 1 && 1 <= q.rear && 1 <= stack.top;
	  @ 	requires q.peek() - stack.peek() <= Integer.MAX_VALUE;
	  @ 	requires Integer.MIN_VALUE <= q.peek() - stack.peek();
	  @ 	ensures \result == q.peek() - stack.peek();
	  @ also
	  @	requires op == 2 && 1 <= q.rear && 1 <= stack.top;
	  @ 	requires stack.peek() != 0;
	  @ 	requires q.peek() / stack.peek() <= Integer.MAX_VALUE;
	  @ 	requires Integer.MIN_VALUE <= q.peek() / stack.peek();
	  @ 	ensures \result == q.peek() / stack.peek();
	  @ also
	  @	requires op == 3 && 1 <= q.rear && 1 <= stack.top;
	  @ 	requires q.peek() != 0;
	  @ 	requires stack.peek() / q.peek() <= Integer.MAX_VALUE;
	  @ 	requires Integer.MIN_VALUE <= stack.peek() / q.peek();
	  @ 	ensures \result == stack.peek() / q.peek();
	  @ also
	  @	requires op == 4 && 1 <= q.rear && 1 <= stack.top;
	  @ 	requires stack.peek() != 0;
	  @ 	requires q.peek() % stack.peek() <= Integer.MAX_VALUE;
	  @ 	requires Integer.MIN_VALUE <= q.peek() % stack.peek();
	  @ 	ensures \result == q.peek() % stack.peek();
	  @ also
	  @	requires op == 5 && 1 <= q.rear && 1 <= stack.top;
	  @ 	requires q.peek() != 0;
	  @ 	requires stack.peek() % q.peek() <= Integer.MAX_VALUE;
	  @ 	requires Integer.MIN_VALUE <= stack.peek() % q.peek();
	  @ 	ensures \result == stack.peek() % q.peek();
        @ |}
        @*/
    	public int driverQStack(Stack stack, Queue q, int op) 
    	{
		StackQueue sq = new StackQueue();
		int output = 0;
		switch (op) {
			case 0:
          		output = sq.plusQStack(q, stack);
                	break;

			case 1:
            		output = sq.minusQStack(q, stack);
               		break;

			case 2:
            		output = sq.qDivideStack(q, stack);
                	break;

			case 3:
			output = sq.stackDivideQ(q, stack);
			break;

			case 4:
			output = sq.qModulusStack(q, stack);
			break;

			default:
			output = sq.stackModulusQ(q, stack);
			break;
		}
		return output;
    	 }
}
