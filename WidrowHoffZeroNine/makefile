CC = gcc
CFLAGS = -Wall -Iinclude
LDFLAGS = -lm
RM = rm -rf
SRC = $(wildcard $(srcdir)*.c)
HEAD = $(wildcard $(srcdir)*.h)
OBJ = $(subst $(srcdir), $(bindir),$(SRC:.c=.o))
PROG = $(bindir)neuralnet
srcdir = ./src/
docdir = ./doc/
bindir = ./bin/
savedir = ./save/
CP = cp

all: $(PROG)

$(PROG): $(OBJ)
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

./bin/%.o : ./src/%.c
	$(CC) $(CFLAGS)  -c $^ -o $@ $(LDFLAGS)
	
.PHONY: clean
clean :
	$(RM) $(OBJ) core

.PHONY: save
save :
	$(CP) $(SRC) $(savedir) && $(CP) $(HEAD) $(savedir)

.PHONY: mrproper
mrproper :
	$(RM) -f $(bindir)* $(docdir)html/ $(docdir)latex/

.PHONY: resetgraphs
resetgrph :
	./resetDat.sh
	
.PHONY: graphs
graphs :
	./runGraphs.sh

.PHONY: doxy
doxy:
	doxygen ./doc/Doxyfile && firefox ./doc/html/index.html