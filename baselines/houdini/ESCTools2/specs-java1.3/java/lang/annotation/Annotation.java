package java.lang.annotation;

public interface Annotation 
{
    Class<? extends Annotation> annotationType();
    boolean equals(Object o);
    int hashCode();
    String toString();
}
