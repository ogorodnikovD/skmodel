prog_omp: prog.cpp
	xlC_r -O1 -qsmp=omp -qarch=pwr8 prog.cpp -o prog_omp

upload:
	scp prog.cpp Makefile omp_*.lsf polus:~/skmodel/

local:
	g++ -O1 -Wno-unknown-pragmas prog.cpp -o prog
	./prog 20 20
	./prog 40 40
	./prog 80 80

run400:
	bsub < omp_2_400x600.lsf
	bsub < omp_4_400x600.lsf
	bsub < omp_8_400x600.lsf
	bsub < omp_16_400x600.lsf

run800:
	bsub < omp_4_800x1200.lsf
	bsub < omp_8_800x1200.lsf
	bsub < omp_16_800x1200.lsf
	bsub < omp_32_800x1200.lsf

results:
	grep -E "threads|iterations|total" omp_*.out

clean:
	rm -f *.out *.err

.PHONY: upload local run400 run800 results clean
