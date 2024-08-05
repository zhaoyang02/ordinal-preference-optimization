#DPO
#sbatch --job-name=others_0.1 dpo.slurm dpo 0.1 8 best_with_others 32
#sbatch --job-name=worst_0.1 dpo.slurm dpo 0.1 8 best_with_worst 32
#sbatch --job-name=all_pairs_0.1 dpo.slurm dpo 0.1 8 all_pairs 32

#SLiC
#sbatch --job-name=hinge_0.1 hinge.slurm hinge 0.1 8 32

#LambdaRank
#sbatch --job-name=lipo_0.1 lipo.slurm lipo 0.1 8 32

#ListMLE
#sbatch --job-name=list_mle_0.1 dpo.slurm dpo 0.1 8 list_mle 32

#ApproxNDCG
#sbatch --job-name=approx_50.0 approx_ndcg.slurm approx_ndcg 50.0 8 32

#NeuralNDCG
#sbatch --job-name=neural_0.1 neural_ndcg.slurm neural_ndcg 0.1 1.0 8 32 none

#Ablation study for NeuralNDCG
#sbatch --job-name=neural_top4 neural_ndcg.slurm neural_ndcg 0.1 1.0 8 32 top-4
#sbatch --job-name=neural_nopower neural_ndcg.slurm neural_ndcg 0.1 1.0 8 32 no_power
#sbatch --job-name=neural_scale neural_ndcg.slurm neural_ndcg 0.1 1.0 8 32 scale