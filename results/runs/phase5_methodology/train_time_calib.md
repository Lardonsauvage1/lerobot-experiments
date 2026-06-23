# Calibrage du temps d'entraînement (M1) — auto-extensible

*Régénéré par `experiments/can/41_calibrate_train_time.py`. Se précise à chaque nouveau run (plus de runs d'une même archi → p10 médian plus resserré). `updt_s` p10 = s/step calcul pur.*

## Par signature d'architecture
| archi (backbone · enc · U-Net · batch) | params | n runs | s/step (p10 médian) | écart | s/exemple |
|---|---|---|---|---|---|
| resnet18 shared U[32,64,128] B32 | 12.8 M | 27 | 0.08 s | ±0.39 | 2.4 ms |
| resnet18 shared U[16,32,64] B32 | 11.8 M | 1 | 0.44 s | — | 13.8 ms |
| resnet18 shared U[8,16,32] B32 | 11.5 M | 1 | 0.44 s | — | 13.8 ms |
| resnet18 shared U[64,128,256] B32 | 5.0 M | 9 | 0.47 s | ±0.36 | 14.8 ms |
| resnet18 shared U[128,256,512] B32 | 28.7 M | 1 | 0.58 s | — | 18.0 ms |
| resnet34 sep U[128,256,512] B16 | 60.9 M | 2 | 0.94 s | ±0.00 | 59.0 ms |
| resnet18 shared U[256,512,1024] B32 | 76.6 M | 1 | 1.08 s | — | 33.9 ms |
| resnet18 sep U[64,128,256] B32 | 27.8 M | 6 | 1.13 s | ±0.96 | 35.4 ms |
| resnet18 sep U[128,256,512] B32 | 40.7 M | 2 | 1.23 s | ±0.02 | 38.5 ms |
| resnet34 shared U[32,64,128] B32 | 23.1 M | 2 | 1.26 s | ±0.03 | 39.4 ms |
| resnet34 sep U[64,128,256] B32 | 48.0 M | 3 | 1.66 s | ±0.17 | 51.7 ms |
| resnet18 shared U[512,1024,2048] B32 | 263.7 M | 5 | 3.58 s | ±2.95 | 112.0 ms |

## Prédiction
`t_total ≈ N_steps × (s/step p10) × 1,15` (overhead val-loss + checkpoints). Exemple : 40k steps à 1,1 s/step ≈ 12,7 h.

## Détail par run
| log | archi | batch | params | s/step p10 |
|---|---|---|---|---|
| run_camaug_50.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_camaug_10.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_mini_constant_continue.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_mini_cosine.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_mini_cosine_150k.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_mini_cosine_continue.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_mini_cosine_continue2.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_camaug_30.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_mini_constant_continue2.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.07 s |
| run_mini_cosine_200k.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_constant.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_constant_finevar.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_constant_resume.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_cosine_300k.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_constant_250k.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_constant_200k.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_constant_150k.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_cont_200000.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_camaug_150.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_mini_cosine_250k.log | resnet18 shared U[32,64,128] | 32 | 1.8 M | 0.08 s |
| run_61_minicnn.log | resnet18 shared U[32,64,128] | 32 | 1.7 M | 0.11 s |
| run_63_dataeff_n100.log | resnet18 shared U[32,64,128] | 32 | 1.7 M | 0.11 s |
| run_63_dataeff_n10.log | resnet18 shared U[32,64,128] | 32 | 1.7 M | 0.11 s |
| run_63_dataeff_n20.log | resnet18 shared U[32,64,128] | 32 | 1.7 M | 0.11 s |
| run_63_dataeff_n50.log | resnet18 shared U[32,64,128] | 32 | 1.7 M | 0.12 s |
| run_01_baseline.log | resnet18 shared U[64,128,256] | 32 | 5.0 M | 0.12 s |
| run_69_grid.log | resnet18 shared U[64,128,256] | 32 | 5.0 M | 0.13 s |
| run_51_unet_d16_32_64.log | resnet18 shared U[16,32,64] | 32 | 11.8 M | 0.44 s |
| run_51_unet_d8_16_32.log | resnet18 shared U[8,16,32] | 32 | 11.5 M | 0.44 s |
| run_51_unet_d32_64_128.log | resnet18 shared U[32,64,128] | 32 | 12.8 M | 0.44 s |
| run_76_proprio_dataeff.log | resnet18 shared U[64,128,256] | 32 | 16.1 M | 0.46 s |
| run_51_unet_d64_128_256.log | resnet18 shared U[64,128,256] | 32 | 16.2 M | 0.46 s |
| run_15_proprio_wristonly.log | resnet18 shared U[64,128,256] | 32 | 16.1 M | 0.47 s |
| run_73_proprio.log | resnet18 shared U[64,128,256] | 32 | 16.1 M | 0.48 s |
| run_02_proprio.log | resnet18 shared U[64,128,256] | 32 | 16.1 M | 0.48 s |
| run_29_resume_25_02.log | resnet18 shared U[64,128,256] | 32 | 16.1 M | 0.55 s |
| run_51_unet_d128_256_512.log | resnet18 shared U[128,256,512] | 32 | 28.7 M | 0.58 s |
| run_camaug_150_resnet18.log | resnet18 shared U[32,64,128] | 32 | 13.0 M | 0.85 s |
| run_06_proprio_wrist.log | resnet18 shared U[64,128,256] | 32 | 16.6 M | 0.85 s |
| run_08_camaug_r34_bigunet.log | resnet34 sep U[128,256,512] | 16 | 60.9 M | 0.94 s |
| run_31_r34_bigunet.log | resnet34 sep U[128,256,512] | 16 | 60.9 M | 0.95 s |
| run_21_proprio_birdview_crop.log | resnet18 sep U[64,128,256] | 32 | 27.8 M | 0.99 s |
| run_18_proprio_birdview_aug.log | resnet18 sep U[64,128,256] | 32 | 27.8 M | 1.03 s |
| run_51_unet_d256_512_1024.log | resnet18 shared U[256,512,1024] | 32 | 76.6 M | 1.08 s |
| run_08_proprio_wrist_sep.log | resnet18 sep U[64,128,256] | 32 | 27.8 M | 1.13 s |
| run_16_proprio_birdview.log | resnet18 sep U[64,128,256] | 32 | 27.8 M | 1.13 s |
| run_13_proprio_wrist_sep_ext.log | resnet18 sep U[64,128,256] | 32 | 27.8 M | 1.15 s |
| run_14_proprio_wrist_sep_big.log | resnet18 sep U[128,256,512] | 32 | 40.7 M | 1.22 s |
| run_camaug_150_resnet34.log | resnet34 shared U[32,64,128] | 32 | 23.1 M | 1.23 s |
| run_25_proprio_birdview_big.log | resnet18 sep U[128,256,512] | 32 | 40.7 M | 1.25 s |
| run_camaug_150_resnet34_ext.log | resnet34 shared U[32,64,128] | 32 | 23.1 M | 1.30 s |
| run_27_resume_all.log | resnet34 sep U[64,128,256] | 32 | 48.0 M | 1.41 s |
| run_26_chain.log | resnet34 sep U[64,128,256] | 32 | 48.0 M | 1.66 s |
| run_01_train_dense.log | resnet34 sep U[64,128,256] | 32 | 48.0 M | 1.76 s |
| run_23_proprio_birdview_160.log | resnet18 sep U[64,128,256] | 32 | 27.8 M | 2.91 s |
| run_46_diffusion.log | resnet18 shared U[512,1024,2048] | 32 | 263.7 M | 3.02 s |
| run_47_baseline_150.log | resnet18 shared U[512,1024,2048] | 32 | 263.7 M | 3.29 s |
| run_canonical_camaug_150.log | resnet18 shared U[512,1024,2048] | 32 | 266.8 M | 3.58 s |
| run_canonical_ext.log | resnet18 shared U[512,1024,2048] | 32 | 266.8 M | 3.87 s |
| run_canonical_ext2.log | resnet18 shared U[512,1024,2048] | 32 | 266.8 M | 8.92 s |
