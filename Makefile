# Makefile

# デフォルトのターゲット
all: gradient_descent.exe plot

gradient_descent.exe: gradient_descent.o optimizer.o
	g++ -o gradient_descent.exe gradient_descent.o optimizer.o -std=c++11

gradient_descent.o: gradient_descent.cpp optimizer.o
	g++ -c gradient_descent.cpp -std=c++11

optimizer.o: optimizer.cpp
	g++ -c optimizer.cpp -std=c++11

gradient_descent.dat: gradient_descent.exe
	./gradient_descent.exe

plot: gradient_descent.gif

gradient_descent.gif: gradient_descent.plt gradient_descent.dat
	gnuplot gradient_descent.plt

# クリーンターゲット
clean:
	rm -f *.o *.exe

.PHONY: all clean