javafiles =	$(wildcard *.java)

javakeyfiles =	$(wildcard *Key.java)

classfiles =	$(foreach javafile,$(javafiles),\
		$(addprefix classes/problems/,$(subst .java,.class,$(javafile))))

jmlcclassfiles =	$(foreach javafile,$(javafiles),\
			$(addprefix jmlc/problems/,$(subst .java,.class,$(javafile))))

keyclasses =	$(foreach javakeyfile,$(javakeyfiles),\
		$(subst .java,,$(javakeyfile)))

classes/problems/%.class: %.java
	mkdir -p classes; javac -d classes -g -source 1.4 $<

jmlc/problems/%.class: %.java
	mkdir -p jmlc; jmlc -d jmlc -g -source 1.4 $<

.PHONY:	all test \
	build javac jmlc \
	run java jmlrac \
	verify \
	clean

all:	build verify

test:
	@echo $(javafiles)
	@echo $(classfiles)
	@echo $(jmlcclassfiles)

build:	javac jmlc

javac:
	javac -d classes -g -source 1.4 *.java

jmlc:
	jmlc -d jmlc -g -source 1.4 *.java

run:	java jmlrac

java:
	for i in $(keyclasses); \
	do \
	  CLASSPATH=classes; java -ea problems.$$i; \
	done

jmlrac:
	for i in $(keyclasses); \
	do \
	  CLASSPATH=jmlc; jmlrac problems.$$i; \
	done

verify:
#	CLASSPATH=..; escjava2 -jmlAssertions JavaAssertionsKey.java
#	CLASSPATH=..; escjava2 -source 1.4 -specs ~/JML2/:~/JML2/specs/ FrameConditionsKey.java
#	CLASSPATH=..; escjava2 -source 1.4 FrameConditionsKey.java
	CLASSPATH=..; escjava2 -source 1.4 ContractsAbstractListKey.java
#	CLASSPATH=..; escjava2 -source 1.4 Contracts*Key.java

clean:
	rm -f `find . -name "*.class"`

