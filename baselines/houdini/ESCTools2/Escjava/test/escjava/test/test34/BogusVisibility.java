class BogusVisibility {
  private int priv;
  int pack;
  protected int prot;
  public int publ;
  /*@ spec_public */ private int spub0;
  /*@ spec_public */ int spub1;
  /*@ spec_public */ protected int spub2;
  /*@ spec_public */ public int spub3;
  
  private static int privSt;
  static int packSt;
  protected static int protSt;
  public static int publSt;
  /*@ spec_public */ private static int spub0St;
  /*@ spec_public */ static int spub1St;
  /*@ spec_public */ protected static int spub2St;
  /*@ spec_public */ public static int spub3St;
  
  private Object privMu;
  Object packMu;
  protected Object protMu;
  public Object publMu;
  /*@ spec_public */ private Object spub0Mu;
  /*@ spec_public */ Object spub1Mu;
  /*@ spec_public */ protected Object spub2Mu;
  /*@ spec_public */ public Object spub3Mu;
  
  // -----  readable_if
  
  /*@ readable_if priv == 0; */ private int defifPriv0;
  /*@ readable_if pack == 0; */ private int defifPriv1;
  /*@ readable_if prot == 0; */ private int defifPriv2;
  /*@ readable_if publ == 0; */ private int defifPriv3;
  /*@ readable_if spub0 == 0; */ private int defifPriv4;
  /*@ readable_if spub1 == 0; */ private int defifPriv5;
  /*@ readable_if spub2 == 0; */ private int defifPriv6;
  /*@ readable_if spub3 == 0; */ private int defifPriv7;
  
  /*@ readable_if priv == 0; */ int defifPack0;
  /*@ readable_if pack == 0; */ int defifPack1;
  /*@ readable_if prot == 0; */ int defifPack2;
  /*@ readable_if publ == 0; */ int defifPack3;
  /*@ readable_if spub0 == 0; */ int defifPack4;
  /*@ readable_if spub1 == 0; */ int defifPack5;
  /*@ readable_if spub2 == 0; */ int defifPack6;
  /*@ readable_if spub3 == 0; */ int defifPack7;
  
  /*@ readable_if priv == 0; */ protected int defifProt0;
  /*@ readable_if pack == 0; */ protected int defifProt1;
  /*@ readable_if prot == 0; */ protected int defifProt2;
  /*@ readable_if publ == 0; */ protected int defifProt3;
  /*@ readable_if spub0 == 0; */ protected int defifProt4;
  /*@ readable_if spub1 == 0; */ protected int defifProt5;
  /*@ readable_if spub2 == 0; */ protected int defifProt6;
  /*@ readable_if spub3 == 0; */ protected int defifProt7;
  
  /*@ readable_if priv == 0; */ public int defifPubl0;
  /*@ readable_if pack == 0; */ public int defifPubl1;
  /*@ readable_if prot == 0; */ public int defifPubl2;
  /*@ readable_if publ == 0; */ public int defifPubl3;
  /*@ readable_if spub0 == 0; */ public int defifPubl4;
  /*@ readable_if spub1 == 0; */ public int defifPubl5;
  /*@ readable_if spub2 == 0; */ public int defifPubl6;
  /*@ readable_if spub3 == 0; */ public int defifPubl7;

  /*@ readable_if defifSelf == 0; */ protected int defifSelf;
  
  // -----  monitored_by

  /*@ monitored_by privMu, packMu, protMu, publMu */
  /*@ monitored_by spub0Mu, spub1Mu, spub2Mu, spub3Mu */
  private int monbyPriv;
  
  /*@ monitored_by privMu, packMu, protMu, publMu */
  /*@ monitored_by spub0Mu, spub1Mu, spub2Mu, spub3Mu */
  int monbyPack;
  
  /*@ monitored_by privMu, packMu, protMu, publMu */
  /*@ monitored_by spub0Mu, spub1Mu, spub2Mu, spub3Mu */
  protected int monbyProt;
  
  /*@ monitored_by privMu, packMu, protMu, publMu */
  /*@ monitored_by spub0Mu, spub1Mu, spub2Mu, spub3Mu */
  public int monbyPubl;

  /*@ monitored_by monbySelf */
  protected Object monbySelf;
  
  // ----- requires

  //@ requires priv == pack && prot == publ;
  //@ requires spub0 == spub1 && spub2 == spub3;
  private void reqPriv() {}

  //@ requires priv == pack && prot == publ;
  //@ requires spub0 == spub1 && spub2 == spub3;
  void reqPack() {}

  //@ requires priv == pack && prot == publ;
  //@ requires spub0 == spub1 && spub2 == spub3;
  protected void reqProt() {}

  //@ requires priv == pack && prot == publ;
  //@ requires spub0 == spub1 && spub2 == spub3;
  public void reqPubl() {}

  //@ requires reqProtSelf;
  protected void reqProtSelf() {}

  // ----- ensures

  //@ ensures priv == pack && prot == publ;
  //@ ensures spub0 == spub1 && spub2 == spub3;
  private void ens0() {}

  //@ ensures priv == pack && prot == publ;
  //@ ensures spub0 == spub1 && spub2 == spub3;
  void ens1() {}

  //@ ensures priv == pack && prot == publ;
  //@ ensures spub0 == spub1 && spub2 == spub3;
  protected void ens2() {}

  //@ ensures priv == pack && prot == publ;
  //@ ensures spub0 == spub1 && spub2 == spub3;
  public void ens3() {}

  //@ ensures priv == pack && prot == publ;
  //@ ensures spub0 == spub1 && spub2 == spub3;
  private final void ens4() {}

  //@ ensures privSt == packSt && protSt == publSt;
  //@ ensures spub0St == spub1St && spub2St == spub3St;
  private static void ens5() {}

  //@ ensures privSt == packSt && protSt == publSt;
  //@ ensures spub0St == spub1St && spub2St == spub3St;
  static void ens6() {}
}

class BogusVisibilitySub extends BogusVisibility {
  private int privSub;
  private static int privSubSt;
  /*@ spec_public */ private int spub4;
  /*@ spec_public */ private static int spub4St;
  
  // ----- also_ensures

  //@ also_ensures privSub == pack && prot == publ;
  //@ also_ensures spub4 == spub1 && spub2 == spub3;
  //@ also_ensures spub0 == spub4;
  private void ens0() {}

  //@ also ensures privSub == pack && prot == publ;
  //@ ensures spub4 == spub1 && spub2 == spub3;
  //@ ensures spub0 == spub4;
  private void ens0also() {}

  //@ also_ensures privSub == pack && prot == publ;
  //@ also_ensures spub4 == spub1 && spub2 == spub3;
  void ens1() {}

  //@ also ensures privSub == pack && prot == publ;
  //@ ensures spub4 == spub1 && spub2 == spub3;
  void ens1also() {}

  //@ also_ensures privSub == pack && prot == publ;
  //@ also_ensures spub4 == spub1 && spub2 == spub3;
  protected void ens2() {}

  //@ also ensures privSub == pack && prot == publ;
  //@ ensures spub4 == spub1 && spub2 == spub3;
  protected void ens2also() {}

  //@ also_ensures privSub == pack && prot == publ;
  //@ also_ensures spub4 == spub1 && spub2 == spub3;
  public void ens3() {}

  //@ also ensures privSub == pack && prot == publ;
  //@ ensures spub4 == spub1 && spub2 == spub3;
  public void ens3also() {}

  //@ also_ensures privSub == pack && prot == publ;
  //@ also_ensures spub4 == spub1 && spub2 == spub3;
  private final void ens4() {}

  //@ also ensures privSub == pack && prot == publ;
  //@ ensures spub4 == spub1 && spub2 == spub3;
  private final void ens4also() {}

  //@ also_ensures privSubSt == packSt && protSt == publSt;
  //@ also_ensures spub4St == spub1St && spub2St == spub3St;
  private static void ens5() {}

  //@ also ensures privSubSt == packSt && protSt == publSt;
  //@ ensures spub4St == spub1St && spub2St == spub3St;
  private static void ens5also() {}

  //@ also_ensures privSubSt == packSt && protSt == publSt;
  //@ also_ensures spub4St == spub1St && spub2St == spub3St;
  static void ens6() {}

  //@ also ensures privSubSt == packSt && protSt == publSt;
  //@ ensures spub4St == spub1St && spub2St == spub3St;
  static void ens6also() {}
}

class BogusVisibility2 {

    void doit(BogusVisibility x) {
	int y = x.spub0;    // Error: spec_public only applies to annotations
    }
}
