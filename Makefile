# Makefile

IDIR = include
CC = gcc
CFLAGS = -I$(IDIR) -O -lmkl_rt -O3 -lpthread -lm -Wall -g -ffast-math

SDIR = src
ODIR = obj

_SOURCES = PERDIF.c rec_IO.c csr_handling.c rec_mem.c perdif_train.c perdif_predict.c rec_graph_gen.c rec_metrics.c perdif_mthreads.c perdif_fit.c
SOURCES = $(patsubst %,$(SDIR)/%,$(_SOURCES))

_OBJECTS = $(_SOURCES:.c=.o)
OBJECTS = $(patsubst %,$(ODIR)/%,$(_OBJECTS))

_DEPS =  rec_defs.h  rec_IO.h csr_handling.h rec_mem.h perdif_train.h perdif_predict.h rec_graph_gen.h rec_metrics.h perdif_mthreads.h perdif_fit.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

EXE = PERDIF 

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(EXE): $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
