package java.lang.reflect;

import java.lang.annotation.Annotation;

public interface AnnotatedElement
{
    Annotation getAnnotation(Class annotationClass);

    Annotation[] getAnnotations();

    Annotation[] getDeclaredAnnotations();

    boolean isAnnotationPresent(Annotation annotationClass);
}
