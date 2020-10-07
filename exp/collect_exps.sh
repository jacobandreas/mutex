#!/bin/sh

std(){
    awk '{sum+=$1; sumsq+=$1*$1}END{print "± " sqrt(sumsq/NR - (sum/NR)**2)}'
}

mean(){
    awk 'BEGIN { sum=0 } { sum+=$1 } END {print sum / NR}'
}
min(){
    sort -n | head -1
}
max(){
    sort -n | tail -1
}

stdmean(){
    echo -n "TEST:"
    mu=$(printf "$1" | mean | tr -d '\n')
    sigma=$(printf "$1" | std | tr -d '\n')
    maximum=$(printf "$1" | max | tr -d '\n')
    minimum=$(printf "$1" | min | tr -d '\n')
    echo -n "$mu ($sigma) (max: $maximum , min: $minimum )"
    echo
}

for lr in 0.0005 0.001 0.002; do
    for Nsample in 50 80 150; do
	for dim in 100 200 400; do
	    for lamda in 0.05 0.1 0.2 0.5; do
		for kll in 0.1 0.2 0.5 1.0; do
		    for ent in 0.0 0.00001 0.0001 0.001; do
			#expname=lamda_${lamda}_kll_${kll}_ent_${ent}
			expname=lr_${lr}_Nsample_${Nsample}_dim_${dim}_lamda_${lamda}_kll_${kll}_ent_${ent}
			if [ -d "$expname" ]; then
			    cd $expname
			    numbers1=$(grep -oh 'test evaluation/acc 0.*' * | awk '{print $3}' FS=" ")
			    echo -n "${expname}"
			    stdmean "$numbers1"
			    cd ..
			else
			    continue
			fi
		    done
		done
	    done
	done
    done	 
done


