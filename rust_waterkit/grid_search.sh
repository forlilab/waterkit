GCMCSteps=(10000 50000 100000 200000 400000 500000 800000)

for steps in "${GCMCSteps[@]}"
do
    qsub -v STEPS=$steps MCSwell_10k_frames.qsub
done
