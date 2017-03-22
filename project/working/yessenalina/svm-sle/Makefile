# Makefile for struct SVM-struct API, 03.10.06

#Call 'make' using the following line to make CYGWIN produce stand-alone Windows executables
#		make 'SFLAGS=-mno-cygwin'

#Use the following to compile under unix or cygwin
CC = gcc
LD = g++
CXX = g++

CFLAGS =   $(SFLAGS) -O3 -fomit-frame-pointer -ffast-math -Wall 
LDFLAGS =  $(SFLAGS) -O3 -lm -Wall
#CFLAGS =  $(SFLAGS) -pg -Wall
#LDFLAGS = $(SFLAGS) -pg -lm -Wall 

all: svm_sle_learn svm_sle_classify

.PHONY: clean
clean: svm_light_clean svm_struct_clean
	rm -f *.o *.tcov *.d core gmon.out *.stackdump svm_sle_classify svm_sle_learn

#-----------------------#
#----   SVM-light   ----#
#-----------------------#
svm_light_hideo_noexe: 
	cd svm_light; make svm_learn_hideo_noexe

svm_light_clean: 
	cd svm_light; make clean

#----------------------#
#----  STRUCT SVM  ----#
#----------------------#

svm_struct_noexe: 
	cd svm_struct; make svm_struct_noexe

svm_struct_clean: 
	cd svm_struct; make clean


#-------------------------#
#----  SVM struct API  ----#
#-------------------------#

svm_sle_classify: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_classify.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o helper.o latent_size_custom.o
	$(LD) $(LDFLAGS) svm_struct_api.o svm_struct/svm_struct_classify.o svm_light/svm_common.o svm_struct/svm_struct_common.o helper.o latent_size_custom.o -o svm_sle_classify $(LIBS)

svm_sle_learn: svm_light_hideo_noexe svm_struct_noexe svm_struct_api.o svm_struct/svm_struct_learn.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o helper.o latent_size_custom.o
	$(LD) $(LDFLAGS) svm_struct/svm_struct_learn.o svm_struct_api.o svm_light/svm_hideo.o svm_light/svm_learn.o svm_light/svm_common.o svm_struct/svm_struct_common.o svm_struct/svm_struct_main.o helper.o latent_size_custom.o -o svm_sle_learn $(LIBS)

svm_struct_api.o: svm_struct_api.c svm_struct_api.h svm_struct_api_types.h svm_struct/svm_struct_common.h helper.h
	$(CC) -c $(CFLAGS) svm_struct_api.c -o svm_struct_api.o

helper.o: helper.cc helper.h
	$(CXX) -c $(CFLAGS) helper.cc -o helper.o

latent_size_custom.o: latent_size_custom.cc helper.h
	$(CXX) -c $(CFLAGS) latent_size_custom.cc -o latent_size_custom.o

