package escjava.translate;

import escjava.ast.TagConstants;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javafe.util.Assert;
import javafe.util.Location;

// For testing
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


/**
 * Simple structure used to parse label encodings (as they are sent to the
 * theorem prover).
 */
public class LabelData implements Comparable {
    private LabelData() {
    }

    private int msgTag;       // id as recognized by TagConstants
    private String msgShort;  // tag as it appears in the label

    private int useLoc;  // usage location
    
    private int declLoc; // declaration location

    private int auxLoc; // auxiliary info location

    private int auxId; // auxiliary info type (-1 == none)

    public int getMsgTag() {
        return msgTag;
    }
    
    public String getMsgShort() {
        return msgShort;
    }
    
    public int getUseLoc() {
        return useLoc;
    }
    
    public boolean hasUseLoc() {
        return useLoc != Location.NULL;
    }

    public int getDeclLoc() {
        return declLoc;
    }
    
    public boolean hasDeclLoc() {
        return declLoc != Location.NULL;
    }

    public int getAuxLoc() {
        return auxLoc;
    }
    
    public boolean hasAuxLoc() {
        return auxLoc != Location.NULL;
    }

    public int getAuxId() {
        return auxId;
    }

    // Describes the syntax of the labels.
    static private Pattern pattern = 
        Pattern.compile("\\s*(.+?)(/(\\d+))?(:([\\d\\.]+)(:([\\d\\.]+))?)?(@(([\\d\\.]+)))?\\s*");

    static private final int ID_GRP = 1;
    static private final int AUXID_GRP = 3;
    static private final int DECLLOC_GRP = 5;
    static private final int AUXLOC_GRP = 7;
    static private final int USELOC_GRP = 9;
    static private final int GROUPS = 10;

    /** Constructs a [LabelData] out of the string representation. */
    // requires a string that matches [pattern]
    static public/* @non_null */LabelData parse(/* @non_null */String s) {
        // initialize with defaults
        LabelData r = new LabelData();
        r.useLoc = r.declLoc = r.auxLoc = Location.NULL;
        r.auxId = -1;

        // split in groups
        Matcher m = pattern.matcher(s);
        if (!m.matches()) {
            //@ assert false;
            //System.out.println("Failed to match."); // DBG
            Assert.notFalse(false, "Label with unknown format.");
        }
        Assert.notFalse(m.groupCount() == GROUPS);

        // parse strings into integers
        String g;
        /*DBG
        System.out.println("groups: " + m.groupCount()); 
        for (int i = 0; i < m.groupCount(); ++i) {
            g = m.group(i); 
            if (g != null) System.out.println("grp " + i + ": " + g); 
        }
        */
        g = m.group(USELOC_GRP);
        if (g != null) r.useLoc = UniqName.suffixToLoc(g);
        g = m.group(DECLLOC_GRP);
        if (g != null) r.declLoc = UniqName.suffixToLoc(g);
        g = m.group(AUXLOC_GRP);
        if (g != null) r.auxLoc = UniqName.suffixToLoc(g);
        g = m.group(AUXID_GRP);
        if (g != null) r.auxId = Integer.parseInt(g);

        // get the tag
        r.msgShort = m.group(ID_GRP);
        r.msgTag = TagConstants.checkFromString(r.msgShort);
        Assert.notFalse(r.msgShort.equals(TagConstants.toString(r.msgTag)), "Tag ID<->label mismatch.");
        
        return r;
    }

    public/*@ non_null */String toString() {
        String r = msgShort;
        char openC = '[', closeC = ']';
        
        if (auxId != -1) r += "/" + auxId;
        if (declLoc != Location.NULL) r += ":" +  openC + Location.toString(declLoc) + closeC;
        if (auxLoc != Location.NULL) r += ":"  +  openC + Location.toString(auxLoc) + closeC;
        if (useLoc != Location.NULL) r += "@"  +  openC + Location.toString(useLoc) + closeC;
        return r;
    }

    public int compareTo(/*@ non_null @*/Object o) {
        LabelData ld = (LabelData) o;

        // prone to overflow!
        if (msgTag != ld.msgTag)
            return msgTag - ld.msgTag;
        if (declLoc != ld.declLoc)
            return declLoc - ld.declLoc;
        if (auxLoc != ld.auxLoc)
            return auxLoc - ld.auxLoc;
        return auxId - ld.auxId;
    }

    // --- Test code ---
    static public void main(String[] args) throws IOException {
        String line;
        InputStreamReader isr = new InputStreamReader(System.in);
        BufferedReader br = new BufferedReader(isr);
        while (true) {
            line = br.readLine();
            if (line == null || line.length() == 0)
                break;
            LabelData ld = LabelData.parse(line);
            /*System.out.println(ld.getMsgTag());
            System.out.println(ld.getDeclLoc());
            System.out.println(ld.getAuxLoc());
            System.out.println(ld.getAuxId());
            System.out.println(ld);*/
        }
    }
}
