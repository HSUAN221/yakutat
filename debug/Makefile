# Makefile 

# Nom du compilateur
CC = mpicxx

# Options de compilation: optimisation, debug etc...
OPT = -O3 -fopenmp -std=c++17 -march=native -Wall
Linking = 
INC = -I ../ -I debug_tools/
# -ipo -xCORE-AVX2 -align array32byte 
# -mcmodel=large
# -heap-arrays 64 
# --coverage -Wall

EXE = debug_hsuan

OBJDIR = Object_Files
SRCDIR = Source_Files

# Defining the objects (OBJS) variables  P.S. If u want to add cpp file, u can modify here.
OBJS =  $(addprefix $(OBJDIR)/,\
    debug_hsuan.o \
	)
# Linking object files
exe :  $(OBJS) 
	@$(CC) -o $(EXE) \
    $(OBJS) \
    $(OPT) $(Linking)


# echo something
	@echo "   ***************** successful *****************   "                                                                                      
	@echo "    |  Author:  Zi-Hsuan Wei                        "                       
	@echo "    |  Good luck !!!                                "  

# Defining the flags of objects
$(OBJDIR)/%.o: %.cpp
	@$(CC) $(OPT) $(INC) $(Linking) -c $< -o $@


# Removing object files
clean :
	@/bin/rm -f Output/*.dat
	@/bin/rm -f Output/*.x
	@/bin/rm -f Output/*.q

cleanall : 
	@/bin/rm -f $(OBJS) $(EXE)  *.mod
	@/bin/rm -f Output/*.dat
	@/bin/rm -f Output/*.x
	@/bin/rm -f Output/*.q
    
config :
	if [ ! -d obj ] ; then mkdir Output ; fi


