#!/bin/sh
# for lamda in 0.05 0.1 0.2 0.5; do
#   for kll in 0.1 0.2 0.5 1.0; do
#       for ent in 0.0 0.00001 0.0001 0.001; do
# 	  screen -X -S lamda_${lamda}_kll_${kll}_ent_${ent} quit
#       done
#   done
# done

takejob='salloc --gres=gpu:volta:1 --time=24:00:00 --constraint=xeon-g6 --cpus-per-task=5 --qos=high  srun'
for lr in 0.0005 0.001 0.002; do
    for Nsample in 50 80 150; do
	for dim in 100 200 400; do
	    for lamda in 0.05; do
		for kll in 1.0; do
		    for ent in 0.001; do
			expname=lr_${lr}_Nsample_${Nsample}_dim_${dim}_lamda_${lamda}_kll_${kll}_ent_${ent}
			mkdir -p $expname
			cd $expname
			[ -d "run.sh" ] && rm "run.sh"
			cat > "run.sh" <<EOF
#!/bin/sh		
home="../../"
for i in \`seq 0 9\`
do
python \$home/main.py \\
         --seed \$i \\
         --n_epochs 50 \\
         --n_batch 1 \\
         --dim ${dim} \\
	 --lr ${lr} \\
         --regularize \\
         --Nsample ${Nsample} \\
         --lamda ${lamda} \\
         --kl_lamda ${kll} \\
	 --ent ${ent} \\
         --temp 1.0 \\
         --full_data \\
         --dropout 0.05 > eval.\$i.out 2> eval.\$i.err
done
EOF
	  chmod u+x run.sh
	  screen -S ${expname} -d -m bash -c  "$takejob ./run.sh"
	  cd ..
		    done
		done
	    done
	done
    done
done


