from pcoptim import LeastSquares, Logistic, L1Regularizer, L2Regularizer, ElasticNetRegularizer

# losses_desc = {
#     # lognormal datasets
#     "lognormal_opt_1_raw": (Logistic, "lognormal_opt_1", "raw", "l2", 1.0/1000),
#     "lognormal_opt_1_norm": (Logistic, "lognormal_opt_1", "norm", "l2", 1.0/1000),
#     "lognormal_opt_2_raw": (Logistic, "lognormal_opt_2", "raw", "l2", 1.0/1000),
#     "lognormal_opt_2_norm": (Logistic, "lognormal_opt_2", "norm", "l2", 1.0/1000),
#     "lognormal_opt_5_raw": (Logistic, "lognormal_opt_5", "raw", "l2", 1.0/1000),
#     "lognormal_opt_5_norm": (Logistic, "lognormal_opt_5", "norm", "l2", 1.0/1000),

#     # sparse lasso
#     "sparse_lasso": (LeastSquares, "lasso", "norm", "l1", 30),

#     # raw real datasets
#     "california_raw": (LeastSquares, "california", "raw", "l1", 1.0),
#     "leukemia_raw": (Logistic, "leukemia", "raw", "l1", 1000),
#     "colon_raw": (Logistic, "colon", "raw", "l1", 0.1),
#     "arrhythmia_raw": (Logistic, "arrhythmia", "raw", "none", "none"), # (Logistic, "arrhythmia", "raw", "l1", 1),
#     "dexter_raw": (Logistic, "dexter", "raw", "l1", 5), #todo
#     "qsar_raw": (LeastSquares, "qsar", "raw", "l1", 0.5),
#     "mtp_raw": (LeastSquares, "mtp", "raw", "none", "none"),
#     "mtp_l1_raw": (LeastSquares, "mtp", "raw", "l1", 0.005),
#     "hill_raw": (Logistic, "hill", "raw", "none", "none"),
#     "hill_l1_raw": (Logistic, "hill", "raw", "l1", 20),
#     "madelon_raw": (Logistic, "madelon", "raw", "none", "none"),
#     "madelon_l1_raw": (Logistic, "madelon", "raw", "l1", 1),
#     "topo21_raw": (LeastSquares, "topo21", "raw", "elastic-net", 0.5, 1),

#     # normalized real datasets
#     "leukemia_norm": (Logistic, "leukemia", "norm", "l1", 0.2),
#     "colon_norm": (Logistic, "colon", "norm", "l1", 0.1),
#     "arrhythmia_norm": (Logistic, "arrhythmia", "norm", "none", "none"), #(Logistic, "arrhythmia", "norm", "l1", 0.1),
#     "dexter_l1_norm": (Logistic, "dexter", "norm", "l1", 0.1), #todo
#     "dexter_l2_norm": (Logistic, "dexter", "norm", "l2", 1.0/600), #todo
#     "qsar_norm": (LeastSquares, "qsar", "norm", "l1", 0.2),
#     "california_norm": (LeastSquares, "california", "norm", "l1", 0.1),
#     "mtp_norm": (LeastSquares, "mtp", "norm", "none", "none"),
#     "mtp_l1_norm": (LeastSquares, "mtp", "norm", "l1", 0.01),
#     "hill_norm": (Logistic, "hill", "norm", "none", "none"),
#     "hill_l1_norm": (Logistic, "hill", "norm", "l1", 0.005),
#     "madelon_l1_norm": (Logistic, "madelon", "norm", "l1", 0.0001),
#     "madelon_norm": (Logistic, "madelon", "norm", "none", "none"),
#     "topo21_norm": (LeastSquares, "topo21", "norm", "elastic-net", 0.5, 1),
# }


# radius = {
#     "sparse_lasso": (503.22063491512756, 7),
#     "california_raw": (0.1734907014796655, 4),
#     "leukemia_raw": (0.0002047307758163219, 7),
#     "colon_raw": (1.7760177306485008, 11),
#     "dexter_raw": (0.012152417409733832, 4),
#     "qsar_raw": (6.185262750821032, 5),
#     "leukemia_norm": (0.9680771585046587, 8),
#     "colon_norm": (1.7793978746050207, 11),
#     "dexter_norm": (1.2404840880868415, 9),
#     "qsar_norm": (0.6279194762354101, 9),
#     "california_norm": (0.8229370321812559, 3),
#     "madelon_l1_raw": (0.010278067422504478, 10),
#     "madelon_l1_norm": (9.192836179130802, 1),
#     "hill_l1_raw": (0.0024757014274262093, 36),
#     "hill_l1_norm": (0.6569739482041056, 4),
#     "mtp_l1_raw": (0.6896753330937823, 5),
#     "mtp_l1_norm": (0.1959073460778213, 22),
# }


losses_desc = {
    # lognormal datasets
    # "lognormal_opt_1_raw": (Logistic, "lognormal_opt_1", "raw", "l2", 1.0/1000),
    "lognormal_opt_1_norm": (Logistic, "lognormal_opt_1", "norm", "l2", 1.0/1000),
    # "lognormal_opt_2_raw": (Logistic, "lognormal_opt_2", "raw", "l2", 1.0/1000),
    "lognormal_opt_2_norm": (Logistic, "lognormal_opt_2", "norm", "l2", 1.0/1000),
    # "lognormal_opt_5_raw": (Logistic, "lognormal_opt_5", "raw", "l2", 1.0/1000),
    # "lognormal_opt_5_norm": (Logistic, "lognormal_opt_5", "norm", "l2", 1.0/1000),

    # sparse lasso
    # "sparse_lasso": (LeastSquares, "lasso", "norm", "l1", 30),

    # raw real datasets
    # "california_raw": (LeastSquares, "california", "raw", "l1", 1.0),
    # "leukemia_raw": (Logistic, "leukemia", "raw", "l1", 1000),
    # "colon_raw": (Logistic, "colon", "raw", "l1", 0.1),
    # "arrhythmia_raw": (Logistic, "arrhythmia", "raw", "none", "none"), # (Logistic, "arrhythmia", "raw", "l1", 1),
    # "dexter_raw": (Logistic, "dexter", "raw", "l1", 5), #todo
    # "qsar_raw": (LeastSquares, "qsar", "raw", "l1", 0.5),
    # "mtp_raw": (LeastSquares, "mtp", "raw", "none", "none"),
    # "mtp_l1_raw": (LeastSquares, "mtp", "raw", "l1", 0.005),
    # "hill_raw": (Logistic, "hill", "raw", "none", "none"),
    # "hill_l1_raw": (Logistic, "hill", "raw", "l1", 20),
    # "madelon_raw": (Logistic, "madelon", "raw", "none", "none"),
    # "madelon_l1_raw": (Logistic, "madelon", "raw", "l1", 1),
    # "topo21_raw": (LeastSquares, "topo21", "raw", "elastic-net", 0.5, 1),

    # normalized real datasets
    #"leukemia_norm": (Logistic, "leukemia", "norm", "l1", 0.2),
    #"colon_norm": (Logistic, "colon", "norm", "l1", 0.1),
    # "arrhythmia_l1_norm": (Logistic, "arrhythmia", "norm", "l1", 0.1),
    # "arrhythmia_l1_norm_2": (Logistic, "arrhythmia", "norm", "l1", 0.01),
    # "arrhythmia_l2_norm": (Logistic, "arrhythmia", "norm", "l2", 1e-4),
    # "arrhythmia_l2_norm_2": (Logistic, "arrhythmia", "norm", "l2", 1),

    # "dexter_l1_norm": (Logistic, "dexter", "norm", "l1", 0.1), #todo
    # #"dexter_l2_norm": (Logistic, "dexter", "norm", "l2", 1.0/600), #todo

    # "qsar_l1_norm": (LeastSquares, "qsar", "norm", "l1", 0.4),
    # "qsar_l1_norm_2": (LeastSquares, "qsar", "norm", "l1", 0.05),

    # "california_l1_norm": (LeastSquares, "california", "norm", "l1", 0.1),

    # "mtp_l1_norm": (LeastSquares, "mtp", "norm", "l1", 0.05),
    # "mtp_l1_norm_2": (LeastSquares, "mtp", "norm", "l1", 0.001),
    # "mtp_l2_norm": (LeastSquares, "mtp", "norm", "l2", 5e-8),

    #"hill_norm": (Logistic, "hill", "norm", "none", "none"),
    #"hill_l1_norm": (Logistic, "hill", "norm", "l1", 0.005),
    # "madelon_l1_norm": (Logistic, "madelon", "norm", "l1", 0.05),
    "madelon_l2_norm": (Logistic, "madelon", "norm", "l2", 1.0),

    # "topo21_norm": (LeastSquares, "topo21", "norm", "l2", 1e-7),

    # "dexter_norm": (Logistic, "dexter", "norm", "l1", 0.1),
    # "dexter_raw": (Logistic, "dexter", "raw", "l1", 5)
}


datasets_size = {
    "lognormal_opt_1_norm": (1000, 100),
    "lognormal_opt_2_norm": (1000, 100),
    #"lognormal_opt_5_norm": (1000, 100),
#     "sparse_lasso": (1000, 1000),
#     "california_l1_norm": (20640, 8),
#     "arrhythmia_l1_norm": (452, 257),
#     "arrhythmia_l2_norm": (452, 257),
# #    "dexter_l1_norm": (600, 11035),
#     "qsar_l1_norm": (1976, 1022),
#     "mtp_l1_norm": (4450, 202),
#     "mtp_l2_norm": (4450, 202),
#     "dexter_norm": (600, 11035),
#     "dexter_raw": (600, 11035),
#     "madelon_l1_norm": (2600, 500),
    "madelon_l2_norm": (2600, 500),
#     "topo21_norm": (8885, 261),
}
