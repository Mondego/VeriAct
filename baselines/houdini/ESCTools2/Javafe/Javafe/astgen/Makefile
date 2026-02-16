# Copyright 2000, 2001, Compaq Computer Corporation

LEX =		flex
CFLAGS = 	-g
ifdef COMSPEC
  CC = gcc
else
  CC = cc
endif

all: astgen astfilelist

astgen: lex.yy.o astutil.o astactions.o astoutput.o
	${CC} -o astgen lex.yy.o astutil.o astactions.o astoutput.o

astfilelist: lex.yy.o astutil.o astfilelistactions.o
	${CC} -o astfilelist lex.yy.o astutil.o astfilelistactions.o

lex.yy.c: astgen.l
	${LEX} astgen.l

lex.yy.o: lex.yy.c astactions.h

astutil.o: astutil.h astutil.c

astactions.o: astutil.h astactions.h astoutput.h astactions.c

astfilelistactions.o: astutil.h astactions.h astoutput.h astfilelistactions.c

astoutput.o: astutil.h astoutput.h astoutput.c

clean: tidy
	rm -f astgen astfilelist

tidy:
	rm -f lex.yy.* *.o *~
