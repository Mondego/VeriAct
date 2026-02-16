# Copyright 2000, 2001, Compaq Computer Corporation

export TOP = ..
include ${TOP}/Makefile.defs

# Makefile.defs provides:
#  JAVAC
#  JAVAC_FLAGS

export CLASSPATH=./java:${ESCJAVA_ROOT}/classfiles:${JAVAFE_ROOT}/classfiles:${UTILS_ROOT}:.

CLASSDIRECTORY=./classfiles

all: clean depend houdini

build: houdini

zip: 
	rm -f ${CLASSDIRECTORY}/houdini.zip

source:
	cd java/houdini; ${MAKE} source

clean: cleanclasses
	cd java/houdini; ${MAKE} clean

cleanclasses:
	cd ${CLASSDIRECTORY}; find . -name \*.class -exec \rm -f {} \;


#HOUDINIJAVATARGETS =	houdini.SimplifyTest \
#			houdini.HoudiniServer \
#			houdini.HoudiniClient \
#			houdini.CheckPoint \
#			houdini.SExpPrinter \
#			houdini.FinalESCRun

#depend: source
#	cd java; CLASSPATH=${LCLASSPATH}  \
#		javadepend -g -d ${CLASSDIRECTORY} ${HOUDINIJAVATARGETS}  \
#		> ../sources

#houdini: source		# depend
#	cd java;  \
#		javamake -d ${CLASSDIRECTORY} `cat ../sources`

depend:

	#tcsh -c "source setup ;  \ 
houdini:
	${JAVAC} ${JAVAC_FLAGS} -d ${CLASSDIRECTORY} -source 1.2 ./java/houdini/CheckPoint.java \
	./java/houdini/FinalESCRun.java \
	./java/houdini/GuardManager.java \
	./java/houdini/HoudiniClient.java \
	./java/houdini/HoudiniServer.java \
	./java/houdini/Job.java \
	./java/houdini/SExpOptimizer.java \
	./java/houdini/SExpParser.java \
	./java/houdini/SExpPrinter.java \
	./java/houdini/SimplifyTest.java \
	./java/houdini/WorkList.java \
	./java/houdini/WorkerState.java \
 \
	./java/houdini/AnnLinks.java \
	./java/houdini/AnnotationGuesser.java \
	./java/houdini/AnnotationVisitor.java \
	./java/houdini/Annotator.java \
	./java/houdini/CommonAnnotationGuesser.java \
	./java/houdini/CopyLoaded.java \
	./java/houdini/InferThrows.java \
	./java/houdini/LibraryAnnotationGuesser.java \
	./java/houdini/Main.java \
	./java/houdini/MoreReachableAnnotationGuesser.java \
	./java/houdini/ReqFalseAnnotationGuesser.java \
	./java/houdini/StandardAnnotationGuesser.java \
	./java/houdini/SwingGuesser.java \
 \
	./java/houdini/OptionHandlerOptions.java \
 \
	./java/houdini/AnnotationInserter.java \
	./java/houdini/FileCollection.java \
	./java/houdini/FileInfo.java \
	./java/houdini/WorkItem.java \
	./java/houdini/Instr.java \
	./java/houdini/NowarnInstr.java \
	./java/houdini/PragmaInstr.java \
 \
	./java/houdini/comsock/Command.java \
	./java/houdini/comsock/Main.java \
	./java/houdini/comsock/SocketCommand.java \
	./java/houdini/comsock/SocketSender.java \
	./java/houdini/util/Assert.java \
	./java/houdini/util/Debug.java \
	./java/houdini/util/Log.java \
	./java/houdini/util/OrderedVector.java \
	./java/houdini/util/PriorityQueue.java \
	./java/houdini/util/Queue.java \
	./java/houdini/util/ShutDown.java \
	./java/houdini/util/SplitPrintStream.java \
	./java/houdini/util/Tree.java \
	./java/houdini/util/Utility.java

