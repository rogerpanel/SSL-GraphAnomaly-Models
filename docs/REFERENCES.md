# References

All references cited from the SSL-GraphAnomaly codebase, in BibTeX
format. Citation keys follow `author-year-shorttitle` where possible.

---

## Graph neural networks for intrusion detection

```bibtex
@inproceedings{lo2022egraphsage,
  author    = {Lo, Wai Weng and Layeghy, Siamak and Sarhan, Mohanad and
               Gallagher, Marcus and Portmann, Marius},
  title     = {{E-GraphSAGE}: A Graph Neural Network Based Intrusion Detection System},
  booktitle = {Proc. IEEE/IFIP Network Operations and Management Symposium (NOMS)},
  year      = {2022},
  doi       = {10.1109/NOMS54207.2022.9789878},
  note      = {Preprint at \url{https://arxiv.org/abs/2103.16329}}
}

@article{caville2022anomale,
  author  = {Caville, Evan and Lo, Wai Weng and Layeghy, Siamak and
             Portmann, Marius},
  title   = {{Anomal-E}: A Self-Supervised Network Intrusion Detection System
             Based on Graph Neural Networks},
  journal = {Knowledge-Based Systems},
  year    = {2022},
  volume  = {258},
  pages   = {110030},
  doi     = {10.1016/j.knosys.2022.110030}
}

@inproceedings{hamilton2017graphsage,
  author    = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  title     = {Inductive Representation Learning on Large Graphs},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2017}
}

@inproceedings{nguyen2023tsids,
  author    = {Nguyen, Thanh and Tran, Phuong and Pham, Hieu},
  title     = {{TS-IDS}: Temporal Self-Supervised Graph Learning for Network Intrusion Detection},
  booktitle = {Proc. IEEE Conf. on Communications and Network Security (CNS)},
  year      = {2023}
}
```

## Self-supervised graph learning

```bibtex
@inproceedings{velickovic2019dgi,
  author    = {Velickovic, Petar and Fedus, William and Hamilton, William L. and
               Lio, Pietro and Bengio, Yoshua and Hjelm, R. Devon},
  title     = {Deep Graph Infomax},
  booktitle = {Proc. International Conference on Learning Representations (ICLR)},
  year      = {2019},
  url       = {https://openreview.net/forum?id=rklz9iAcKQ}
}

@inproceedings{hou2022graphmae,
  author    = {Hou, Zhenyu and Liu, Xiao and Cen, Yukuo and Dong, Yuxiao and
               Yang, Hongxia and Wang, Chunjie and Tang, Jie},
  title     = {{GraphMAE}: Self-Supervised Masked Graph Autoencoders},
  booktitle = {Proc. ACM SIGKDD Conference},
  year      = {2022},
  doi       = {10.1145/3534678.3539321}
}
```

## Network intrusion detection baselines

```bibtex
@inproceedings{mirsky2018kitsune,
  author    = {Mirsky, Yisroel and Doitshman, Tomer and Elovici, Yuval and
               Shabtai, Asaf},
  title     = {{Kitsune}: An Ensemble of Autoencoders for Online Network Intrusion Detection},
  booktitle = {Proc. Network and Distributed System Security Symposium (NDSS)},
  year      = {2018},
  doi       = {10.14722/ndss.2018.23204}
}

@article{wu2022rtids,
  author  = {Wu, Zihan and Zhang, Hong and Wang, Penghai and Sun, Zhibo},
  title   = {{RTIDS}: A Robust Transformer-Based Approach for Intrusion Detection System},
  journal = {IEEE Access},
  year    = {2022},
  volume  = {10},
  pages   = {64375--64387}
}

@article{ferrag2024securitybert,
  author  = {Ferrag, Mohamed Amine and Ndhlovu, Mthulisi and Tihanyi, Norbert and
             Cordeiro, Lucas C. and Debbah, Merouane and Lestable, Thierry and
             Thandi, Narinderjit Singh},
  title   = {{SecurityBERT}: Revolutionizing Cyber Threat Detection
             with Lightweight {BERT}-Based Architecture},
  journal = {IEEE Access},
  year    = {2024},
  volume  = {12},
  pages   = {23733--23750},
  doi     = {10.1109/ACCESS.2024.3363469}
}
```

## Conformal prediction

```bibtex
@inproceedings{huang2023cfgnn,
  author    = {Huang, Kexin and Jin, Ying and Candes, Emmanuel J. and Leskovec, Jure},
  title     = {Uncertainty Quantification over Graph with Conformalized Graph Neural Networks},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2023}
}

@inproceedings{zargarbashi2023daps,
  author    = {Zargarbashi, Soroush H. and Antonelli, Simone and Bojchevski, Aleksandar},
  title     = {Conformal Prediction Sets for Graph Neural Networks},
  booktitle = {Proc. International Conference on Machine Learning (ICML)},
  year      = {2023}
}

@article{gibbs2021aci,
  author  = {Gibbs, Isaac and Candes, Emmanuel J.},
  title   = {Adaptive Conformal Inference Under Distribution Shift},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2021},
  note    = {\url{https://arxiv.org/abs/2106.00170}}
}

@article{bates2021rcps,
  author  = {Bates, Stephen and Angelopoulos, Anastasios N. and Lei, Lihua and
             Malik, Jitendra and Jordan, Michael I.},
  title   = {Distribution-Free, Risk-Controlling Prediction Sets},
  journal = {Journal of the ACM},
  year    = {2021},
  volume  = {68},
  number  = {6}
}

@book{vovk2022alrw,
  author    = {Vovk, Vladimir and Gammerman, Alex and Shafer, Glenn},
  title     = {Algorithmic Learning in a Random World},
  edition   = {2nd},
  publisher = {Springer},
  year      = {2022}
}

@article{angelopoulos2023gentle,
  author  = {Angelopoulos, Anastasios N. and Bates, Stephen},
  title   = {A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification},
  journal = {Foundations and Trends in Machine Learning},
  year    = {2023},
  note    = {\url{https://arxiv.org/abs/2107.07511}}
}

@inproceedings{romano2020aps,
  author    = {Romano, Yaniv and Sesia, Matteo and Candes, Emmanuel J.},
  title     = {Classification with Valid and Adaptive Coverage},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2020}
}

@inproceedings{angelopoulos2021raps,
  author    = {Angelopoulos, Anastasios N. and Bates, Stephen and
               Jordan, Michael I. and Malik, Jitendra},
  title     = {Uncertainty Sets for Image Classifiers Using Conformal Prediction},
  booktitle = {Proc. International Conference on Learning Representations (ICLR)},
  year      = {2021}
}

@article{barber2023beyond,
  author  = {Barber, Rina Foygel and Candes, Emmanuel J. and Ramdas, Aaditya and
             Tibshirani, Ryan J.},
  title   = {Conformal Prediction Beyond Exchangeability},
  journal = {Annals of Statistics},
  year    = {2023},
  volume  = {51},
  number  = {2},
  pages   = {816--845}
}

@article{tibshirani2019weighted,
  author  = {Tibshirani, Ryan J. and Barber, Rina Foygel and Candes, Emmanuel J. and
             Ramdas, Aaditya},
  title   = {Conformal Prediction Under Covariate Shift},
  journal = {Advances in Neural Information Processing Systems},
  year    = {2019}
}
```

## Datasets and standardisation

```bibtex
@article{sarhan2022netflow,
  author  = {Sarhan, Mohanad and Layeghy, Siamak and Portmann, Marius},
  title   = {Towards a Standard Feature Set for Network Intrusion Detection System Datasets},
  journal = {Journal of Big Data},
  year    = {2022},
  volume  = {9},
  number  = {1},
  pages   = {1--26},
  doi     = {10.1186/s40537-022-00553-y}
}

@inproceedings{sharafaldin2018cicids,
  author    = {Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A.},
  title     = {Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  booktitle = {Proc. International Conference on Information Systems Security and Privacy (ICISSP)},
  year      = {2018}
}

@article{neto2023ciciot,
  author  = {Neto, Euclides Carlos Pinto and Dadkhah, Sajjad and Ferreira, Raphael and
             Zohourian, Alireza and Lu, Rongxing and Ghorbani, Ali A.},
  title   = {{CICIoT2023}: A Real-Time Dataset and Benchmark for Large-Scale Attacks in {IoT} Environment},
  journal = {Sensors},
  year    = {2023},
  volume  = {23},
  number  = {13},
  pages   = {5941}
}

@inproceedings{moustafa2015unswnb15,
  author    = {Moustafa, Nour and Slay, Jill},
  title     = {{UNSW-NB15}: A Comprehensive Data Set for Network Intrusion Detection Systems},
  booktitle = {Proc. Military Communications and Information Systems Conference (MilCIS)},
  year      = {2015}
}

@article{ferrag2022edgeiiotset,
  author  = {Ferrag, Mohamed Amine and Friha, Othmane and Hamouda, Djallel and
             Maglaras, Leandros and Janicke, Helge},
  title   = {{Edge-IIoTset}: A New Comprehensive Realistic Cyber Security Dataset of
             {IoT} and {IIoT} Applications for Centralized and Federated Learning},
  journal = {IEEE Access},
  year    = {2022}
}
```

## Companion RobustIDPS.ai work (same authors, prior detectors)

```bibtex
@article{anaedevha2024robustadv,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich},
  title   = {{RobustAdv}: Adversarially Robust Network Intrusion Detection},
  journal = {RobustIDPS.ai Technical Reports},
  year    = {2024}
}

@article{anaedevha2024deviceid,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich},
  title   = {Encrypted-Traffic Device Identification for {IoT} Defence},
  journal = {RobustIDPS.ai Technical Reports},
  year    = {2024}
}

@article{anaedevha2025mambashield,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich and
             Borodachev, Yuri Vladimirovich},
  title   = {{MambaShield}: State-Space Models for Stream IDS},
  journal = {RobustIDPS.ai Technical Reports},
  year    = {2025}
}

@article{anaedevha2025stochmm,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich and
             Borodachev, Yuri Vladimirovich},
  title   = {{StochMM}: Stochastic Markov Models for Intrusion Detection},
  journal = {RobustIDPS.ai Technical Reports},
  year    = {2025}
}

@article{anaedevha2026hgp,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich and
             Borodachev, Yuri Vladimirovich},
  title   = {Hierarchical Gaussian Process Detectors for {IDS}},
  journal = {RobustIDPS.ai Technical Reports},
  year    = {2026}
}

@article{anaedevha2026neuralode,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich and
             Borodachev, Yuri Vladimirovich},
  title   = {Neural-{ODE} Anomaly Detection for Streaming Traffic},
  journal = {RobustIDPS.ai Technical Reports},
  year    = {2026}
}

@misc{robustidps2026platform,
  author  = {Anaedevha, Roger Nick and Trofimov, Alexander Gennadievich and
             Borodachev, Yuri Vladimirovich},
  title   = {{RobustIDPS.ai} v3 Platform Reproducibility Archive},
  year    = {2026},
  doi     = {10.5281/zenodo.19129512},
  note    = {\url{https://doi.org/10.5281/zenodo.19129512}}
}
```
