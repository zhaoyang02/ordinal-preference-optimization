#dpo
#sbatch --job-name=all_pairs dpo.slurm dpo 0.1 8 all_pairs 8
#sbatch --job-name=list_mle dpo.slurm dpo 0.1 8 list_mle 8

#command for approx_ndcg_1 
#sbatch --job-name=approx approx_ndcg_1.slurm approx_ndcg_1 25.0 8 8

#command for lipo
#sbatch --job-name=lipo lipo.slurm lipo 0.1 8 8

#hinge
#sbatch --job-name=hinge hinge.slurm hinge 0.1 8 8

#command for neural_ndcg
#sbatch --job-name=neural neural_ndcg.slurm neural_ndcg 0.1 1.0 8 8 none