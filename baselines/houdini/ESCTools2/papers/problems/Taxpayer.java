/* 
  This assignment illustrates how contraints written in a formal
  language can help in removing error in code. 

  (NB we use contraints in the common sense as used in UML or ORM; 
   JML also has a notion of constriants, that is somewhat different, 
   but we won't use the word in the sense of JML.)

  The assignment concerns a class "Taxpayer" that is used for taxpayers.

 */
class Taxpayer {

 /* FIELDS */

 /* isFemale is true iff the person is female */
 boolean isFemale;

 /* isMale is true iff the person is male */
 boolean isMale;

 Taxpayer father, mother; // These fields won't really be used until
                          // the next part of the exercise.

 /* Age in years */ 
 int age; 

 boolean isMarried; 

 /* Reference to spouce if person is married, null otherwise */
 Taxpayer spouse; 

 /* Constant default income tax allowance (belastingvrije som) */
 static final int DEFAULT_ALLOWANCE = 5000;

 /* Constant income tax allowance for Old Age Pensioners over 65 */
 static final  int ALLOWANCE_OAP = 7000;

 /* Income tax allowance (belastingvrije som) */
 int tax_allowance; 

 /* Income per year */
 int income; 

 /* CONSTRUCTOR */

 Taxpayer(boolean jongetje, Taxpayer ma, Taxpayer pa) {
   age = 0;
   isMarried = false;
   this.isMale = jongetje;
   this.isFemale = !jongetje;
   mother = ma;
   father = pa;
   spouse = null;
   income = 0;
   tax_allowance = DEFAULT_ALLOWANCE;
   /* The line below makes explicit the assumption that a new Taxpayer is not 
    * married to anyone yet. A bit silly of course, but we need this to keep 
    * ESC/Java2 happy. */
   //@ assume (\forall Taxpayer p; p.spouse != this); 
 } 

 /* METHODS */

 /* Marry to new_spouse */
 //@ requires new_spouse != null;
 void marry(Taxpayer new_spouse) {
  spouse = new_spouse;
  isMarried = true;
 }
 
 /* Divorce from current spouse */
 void divorce() {
  spouse.spouse = null;
  spouse = null;
  isMarried = false;
 }

 /* Transfer change of the tax allowance from this person to his/her spouse */
 void transferAllowance(int change) {
  tax_allowance = tax_allowance - change;
  spouse.tax_allowance = spouse.tax_allowance + change;
 }

 /* Taxpayer has a birthday and the age increases by one */
 void haveBirthday() {
  age++;
 }

}
