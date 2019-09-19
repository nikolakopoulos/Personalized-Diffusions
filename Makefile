# Makefile

IDIR = include
CC = gcc
# CFLAGS = -I$(IDIR) -O -lblas -O3 -lpthread -lm -Wall -g -ffast-math -fopenmp
CFLAGS = -I$(IDIR) -O -lmkl_rt -O3 -lpthread -lm -Wall -g -ffast-math -fopenmp

SDIR = src
ODIR = obj

_SOURCES = perdif_learn.c rec_IO.c csr_handling.c rec_mem.c perdif_train.c perdif_predict.c rec_graph_gen.c rec_metrics.c perdif_mthreads.c perdif_fit.c
SOURCES = $(patsubst %,$(SDIR)/%,$(_SOURCES))

_SOURCES2 = perdif_mselect.c rec_IO.c csr_handling.c rec_mem.c perdif_train.c perdif_predict.c rec_graph_gen.c rec_metrics.c perdif_mthreads.c perdif_fit.c
SOURCES2 = $(patsubst %,$(SDIR)/%,$(_SOURCES2))

_OBJECTS = $(_SOURCES:.c=.o)
OBJECTS = $(patsubst %,$(ODIR)/%,$(_OBJECTS))

_OBJECTS2 = $(_SOURCES2:.c=.o)
OBJECTS2 = $(patsubst %,$(ODIR)/%,$(_OBJECTS2))

_DEPS =  rec_defs.h  rec_IO.h csr_handling.h rec_mem.h perdif_train.h perdif_predict.h rec_graph_gen.h rec_metrics.h perdif_mthreads.h perdif_fit.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

all: perdif_learn perdif_mselect

$(ODIR)/%.o: $(SDIR)/%.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

perdif_learn: $(OBJECTS)
	$(CC) -o $@ $^ $(CFLAGS)

perdif_mselect:$(OBJECTS2)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o
