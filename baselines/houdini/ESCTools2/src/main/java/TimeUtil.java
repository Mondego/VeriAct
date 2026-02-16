package escjava.dfa.daganalysis;

import java.util.HashMap;

public class TimeUtil {
    static private HashMap startT = new HashMap(); // start times
    static private HashMap lastT = new HashMap(); // last time
    static private HashMap totalT = new HashMap();
    
    static void start(String s) {
        startT.put(s, new Long(System.currentTimeMillis()));
    }
    
    static void stop(String s) {
        long b = ((Long)startT.get(s)).longValue();
        long l = System.currentTimeMillis() - b;
        Long c = (Long)totalT.get(s);
        long d = l;
        if (c != null) d += c.longValue();
        lastT.put(s, new Long(l));
        totalT.put(s, new Long(d));
    }
    
    static void last(String s) {
        long b = ((Long)lastT.get(s)).longValue();
        System.err.println(s + " " + b);
    }

    static void total(String s) {
        long b = ((Long)totalT.get(s)).longValue();
        System.err.println("total_" + s + " " + b);
    }
    
    static void stoprep(String s) {
        stop(s);
        last(s);
    }
}
