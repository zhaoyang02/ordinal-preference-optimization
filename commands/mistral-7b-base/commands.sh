#dpo
#sbatch --job-name=all_pairs dpo.slurm dpo 0.1 8 all_pairs 8
#sbatch --job-name=worst dpo.slurm dpo 0.1 8 best_with_worst 8
#sbatch --job-name=others dpo.slurm dpo 0.1 8 best_with_others 8

#SLiC
#sbatch --job-name=hinge hinge.slurm hinge 0.05 8 8

#LiPO
#sbatch --job-name=lipo lipo.slurm lipo 0.1 8 8

# ListMLE
#sbatch --job-name=list_mle dpo.slurm dpo 0.1 8 list_mle 8

#ApproxNDCG
#sbatch --job-name=approx approx_ndcg_1.slurm approx_ndcg_1 25.0 8 8

#NeuralNDCG
#sbatch --job-name=neural neural_ndcg.slurm neural_ndcg 0.1 1.0 8 8 none