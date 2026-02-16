build:
	${MAKE} -C Escjava2-ImplementationNotes all
	${MAKE} -C Escjava2-Logics all
	${MAKE} -C Escjava2-Provers all
	${MAKE} -C Escjava2-Extending all

all:	clean build

clean: clean-norel
	${MAKE} -C Escjava2-ImplementationNotes clean
	${MAKE} -C Escjava2-Logics clean
	${MAKE} -C Escjava2-Provers clean
	${MAKE} -C Escjava2-Extending clean

clean-norel:
	rm -rf api/*

test:
	@true
