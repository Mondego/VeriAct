package java.lang.annotation;

public interface Annotation 
{
    Annotation annotationType();
    boolean equals(Object o);
    int hashCode();
    String toString();
}
