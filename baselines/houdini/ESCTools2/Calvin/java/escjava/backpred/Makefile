# Copyright Hewlett-Packard, 2002

all:
	@echo "No all target exists!"

full: source
	$(JAVAC) -d ${CLASSDIRECTORY} *.java

source:
	rm -f DefaultUnivBackPred.java
	@echo 'package escjava.backpred;' > DefaultUnivBackPred.java
	@echo 'class DefaultUnivBackPred {' >> DefaultUnivBackPred.java
	@echo '  static String s =' >> DefaultUnivBackPred.java
	sed -e 's/"/\\"/g' \
	  -e 's/^/    "/' \
	  -e 's/$$/\\n" +/' \
	  UnivBackPred.ax >> DefaultUnivBackPred.java
	@echo '    "";' >> DefaultUnivBackPred.java
	@echo '}' >> DefaultUnivBackPred.java

# The following file is no longer used
ShortBackPred.ax: UnivBackPred.ax
	cat UnivBackPred.ax | lim collectDefpreds.lim > ShortBackPred.ax
	cat UnivBackPred.ax | lim gatherBGPushes.lim >> ShortBackPred.ax

clean: 
	rm -f *~
	rm -rf *.class ${CLASSDIRECTORY}/escjava/backpred/*.class
	rm -f DefaultUnivBackPred.java

runtest:
	@echo "No tests to run."
