public class IsSuffix {

    public boolean isSuffix (String pat, String txt) {
	if(pat.length() == 0)
	    return true;
	else if (txt.length() == 0)
	    return false;
	
        int i = pat.length() - 1;
        int j = txt.length() - 1;

        while(i >= 0) {
            if(j < 0 || pat.charAt(i) != txt.charAt(i))
                return false;
            i = i - 1;
            j = j - 1;
        }

        return true;
    }

}
