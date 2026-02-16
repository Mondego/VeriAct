#include "DemonstrationProver.h"
#include <stdio.h>

/*
 * Class:     DemonstrationProver
 * Method:    demo_start_prover
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1start_1prover
  (JNIEnv *env, jobject obj) {
  // return start_solver();
  printf("Prover started in JNI interface.\n");
  return 0;
}

/*
 * Class:     DemonstrationProver
 * Method:    demo_set_prover_resource_flags
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1set_1prover_1resource_1flags
  (JNIEnv *env, jobject obj, jstring properties) {
  const char *properties_string = (*env)->GetStringUTFChars(env, properties, 0);
  // use properties_string in native call
  (*env)->ReleaseStringUTFChars(env, properties, properties_string);
}

/*
 * Class:     DemonstrationProver
 * Method:    demo_signature
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1signature
  (JNIEnv *env, jobject obj, jstring signature) {
  const char *signature_string = (*env)->GetStringUTFChars(env, signature, 0);
  // use signature_string in native call
  (*env)->ReleaseStringUTFChars(env, signature, signature_string);
}

/*
 * Class:     DemonstrationProver
 * Method:    demo_declare_axiom
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1declare_1axiom
  (JNIEnv *env, jobject obj, jstring axiom) {
  const char *axiom_string = (*env)->GetStringUTFChars(env, axiom, 0);
  // use axiom_string in native call
  (*env)->ReleaseStringUTFChars(env, axiom, axiom_string);
}

/*
 * Class:     DemonstrationProver
 * Method:    demo_make_assumption
 * Signature: (Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1make_1assumption
  (JNIEnv *env, jobject obj, jstring formula) {
  const char *formula_string = (*env)->GetStringUTFChars(env, formula, 0);
  // use formula_string in native call
  (*env)->ReleaseStringUTFChars(env, formula, formula_string);
}

/*
 * Class:     DemonstrationProver
 * Method:    demo_retract_assumption
 * Signature: (I)I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1retract_1assumption
  (JNIEnv *env, jobject obj, jint count) {
}

/*
 * Class:     DemonstrationProver
 * Method:    demo_is_valid
 * Signature: (Ljava/lang/String;Ljava/lang/String;)I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1is_1valid
  (JNIEnv *env, jobject obj, jstring formula, jstring properties) {
  const char *formula_string = (*env)->GetStringUTFChars(env, formula, 0);
  const char *properties_string = (*env)->GetStringUTFChars(env, properties, 0);
  // use formula_string in native call
  // use properties_string in native call
  (*env)->ReleaseStringUTFChars(env, formula, formula_string);
  (*env)->ReleaseStringUTFChars(env, properties, properties_string);
}

/*
 * Class:     DemonstrationProver
 * Method:    demo_stop_prover
 * Signature: ()I
 */
JNIEXPORT jint JNICALL Java_DemonstrationProver_demo_1stop_1prover
  (JNIEnv *env, jobject obj) {
  // return stop_solver();
  printf("Prover stopped in JNI interface.\n");
  return 0;
}

int main(int argc, char **argv) {
  Java_DemonstrationProver_demo_1start_1prover(NULL, NULL);
  Java_DemonstrationProver_demo_1stop_1prover(NULL, NULL);
}
