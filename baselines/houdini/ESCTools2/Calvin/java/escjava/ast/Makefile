# Copyright Hewlett-Packard, 2002

all:
	@echo "No all target exists!"

full: source
	$(JAVAC) -d ${CLASSDIRECTORY} *.java

#####

#
# Dependence on Javafe astgen programs:
#

ASTGEN = $(JAVAFE_ROOT)/astgen/astgen
ASTFILELIST = $(JAVAFE_ROOT)/astgen/astfilelist

$(ASTFILELIST) $(ASTGEN):
	(cd $(JAVAFE_ROOT)/astgen; ${MAKE})

#
# Generating Java files:
#

VECTEMPLATE = $(JAVAFE_ROOT)/java/javafe/util/_TYPE_Vec.j

TypeVec = \
    GuardedCmdVec.java GenericVarDeclVec.java ConditionVec.java \
    LocalVarDeclVec.java ExprModifierPragmaVec.java ExprDeclPragmaVec.java \
    ExprStmtPragmaVec.java FieldDeclVec.java VarExprModifierPragmaVec.java \
    DefPredVec.java  LocalVarDeclVec.java PerformsStmtVec.java

GenericVarDeclVec.java: $(VECTEMPLATE)
	sed s/_TYPE_/GenericVarDecl/g $(VECTEMPLATE) | \
        sed s/_PKG_/escjava.ast/g              | \
        sed s/_COMPONENTPKG_/javafe.ast/g > $@

FieldDeclVec.java: $(VECTEMPLATE)
	sed s/_TYPE_/FieldDecl/g $(VECTEMPLATE) | \
        sed s/_PKG_/escjava.ast/g              | \
        sed s/_COMPONENTPKG_/javafe.ast/g > $@

%Vec.java: $(VECTEMPLATE)
	sed s/_TYPE_/$*/g $(VECTEMPLATE) | \
        sed s/_PKG_/escjava.ast/g              | \
        sed s/_COMPONENTPKG_/escjava.ast/g > $@

source: GeneratedTags.java $(TypeVec) $(ASTGEN)

GeneratedTags.java: $(ASTGEN) hierarchy.j
	$(ASTGEN) < hierarchy.j
	grep -v "//@" AnOverview.java > AnOverview.tmp
	mv AnOverview.tmp AnOverview.java
	grep -v "//@" GCExpr.java > GCExpr.tmp
	mv GCExpr.tmp GCExpr.java
	grep -v "//@" GuardedCmd.java > GuardedCmd.tmp
	mv GuardedCmd.tmp GuardedCmd.java

LocalVarDeclVec.java: $(VECTEMPLATE)
	sed s/_TYPE_/LocalVarDecl/g $(VECTEMPLATE) | \
        sed s/_PKG_/escjava.ast/g              | \
        sed s/_COMPONENTPKG_/javafe.ast/g > $@
#####

clean: $(ASTFILELIST)
	$(ASTFILELIST) < hierarchy.j | xargs rm -f 
	rm -f *~ ${TypeVec} ASTGEN* zASTGEN* *.tmp
	rm -rf *.class ${CLASSDIRECTORY}/escjava/ast/*.class

runtest:
	@echo "No tests to run."
