GCMCSteps=(1000 5000 10000 20000 40000 50000 70000 80000 90000 100000 200000)

for steps in "${GCMCSteps[@]}"
do
    qsub -v STEPS=$steps MCSwell_10k_frames.qsub
done
