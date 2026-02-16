//From: helen@polytech.unice.fr
//Date: 1 May, 2008 11:35:13 IST
//To: Joseph Kiniry <kiniry@acm.org>
//Cc: David Cok <cok@frontiernet.net>
//
//On 1 May, 2008, at 0:54, David Cok wrote:
//
//I looked into Hélène's question a bit more.  I've attached her
//bsearch program with annotations that enable escjava to verify it
//(using either -loop 6 or -loopSafe)
//
//A few points:
//1) I added the requires tab != null; that we all pointed out.
//2) I split the ensures into two ensures - this is not necessary but
//now the warning messages point to the conjunct that cannot be proven.
//3) I changed the requires clause that specifies the array is ordered
//from saying that tab[i] <= tab[i+1] to saying that i<=j ==> tab[i]
//<= tab[j]
//These are equivalent, but it takes induction to prove that latter
//from the former, and Simplify does not do induction. It is the
//latter that is needed  in the loop.
//4) I added loop invariants.  Escjava does not infer loop invariants,
//so we need to supply them.  It is interesting that people generally
//think of this algorithm as narrowing the range (between gauche and
//droite) that contains x.  Actually at no point is x necessarily
//between tab[gauche] and tab[droite].  The loop invariants actually
//make clear that what the algorithm is doing is expanding the range
//of tab entries that are known to be not equal to x - until there is
//either none left or one is found.
//5) For good measure, though not needed in this case, I added a
//variant condition.  The natural measure that decreases in each loop
//is droite-gauche.  However it does not decrease in the last loop
//iteration if x is found - hence the hack of subtracting the value of
//result.
//
//Specifying this algorithm would make an interesting student
//exercise, though I didn't search the literature - it has likely been
//done already.

class BSearch {

 /*@ requires tab != null;
   @ //requires (\forall int i; (i >= 0 && i < tab.length -1); tab[i] <= tab[i+1]);
   @ requires (\forall int i,j; (i >= 0 && i < tab.length ) && j >= i && j < tab.length; tab[i] <= tab[j]);
   @ ensures ((\result == -1) ==> (\forall int i; (i >= 0 && i < tab.length); tab[i] != x));
   @ ensures ((\result != -1) ==> (tab[\result] == x));
   @*/
 int binarySearch (int[] tab, int x) {
     int result = -1;
     int milieu = 0;
     int gauche = 0;
     int droite = tab.length -1;
     //@ maintaining (\forall int i; i>=0 && i < gauche; tab[i] !=x);
     //@ maintaining (\forall int i; i>droite && i < tab.length;tab[i] != x);
     //@ maintaining result != -1 ==> tab[result] == x;
     //@ maintaining gauche >= 0;
             //@ maintaining droite <= tab.length-1;
             //@ decreases droite-gauche-result;
     while (result == -1 && gauche <= droite) {
         milieu = (gauche + droite) / 2;
         if (tab[milieu] == x) {
             result = milieu;
         } else {
             if (tab[milieu] > x) {
                  droite = milieu - 1;
              } else {
                  gauche = milieu + 1;
              }
         }
         //@ assert result != -1 ==> (result == milieu && tab[result] == x);
     }
     return result;
 }
}

