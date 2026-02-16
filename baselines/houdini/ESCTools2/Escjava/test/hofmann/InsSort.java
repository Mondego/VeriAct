class InsSort
{
    //@ public normal_behavior
    //@   requires args != null;
    //@   requires args.length >= 2;
    public static void main(String[] args) {
	List.init(2 * (args.length - 1));

	List l = List.nil();

	for(int i = 1; i < args.length; i++) {
	    l = List.cons(Integer.valueOf(args[i]).intValue(), l);
	}

	printlist(sort(l));

    }

    public static void printlist(List l) {
	if (List.isnil(l))
	    System.out.println();
	else {
	    System.out.print(List.head(l) + " ");
	    printlist(List.tail(l));
	}
    }

    public static List insert(int i, List l) {
	if (List.isnil(l))
	    return List.cons(i, List.nil());
	else {
	    int h = List.head(l);
	    List t = List.tail(l);
	    List.free(l);
	    if (h <= i) {
		return List.cons(h, insert(i, t));
	    } else {
		return List.cons(i, List.cons(h, t));
	    }
	}
    }

    public static List sort(List l) {
	if (List.isnil(l))
	    return List.nil();
	else {
	    int h = List.head(l);
	    List t = List.tail(l);
	    return insert(h, sort(t));
	}
    }
}