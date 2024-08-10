#DPO
#sbatch --job-name=others dpo.slurm dpo 0.1 8 best_with_others 32
#sbatch --job-name=worst dpo.slurm dpo 0.1 8 best_with_worst 32
#sbatch --job-name=all_pairs dpo.slurm dpo 0.1 8 all_pairs 32

#SLiC
#sbatch --job-name=hinge hinge.slurm hinge 0.1 8 32

#LambdaRank
#sbatch --job-name=lipo lipo.slurm lipo 0.1 8 32

#ListMLE
#sbatch --job-name=list_mle dpo.slurm dpo 0.1 8 list_mle 32

#ApproxNDCG
#sbatch --job-name=approx approx_ndcg.slurm approx_ndcg 25.0 8 32

#NeuralNDCG
#sbatch --job-name=neural neural_ndcg.slurm neural_ndcg 0.1 1.0 8 32