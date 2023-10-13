# Awesome Tensorial Neural Networks [![Awesome](https://awesome.re/badge.svg)](https://awesome.re)


An survey of tensorial neural networks (TNNs) in
- Network compression via TNNs
- Information fusion via TNNs
- Quantum Circuit Simulation via TNNs
- Training Strategy of TNNs
- Toolboxes of TNNs


This repository is consistent with our survey paper `Tensor Networks Meet Neural Networks: A Survey and Future Perspectives`. Please see https://arxiv.org/abs/2302.09019 for more details. And if you find this work helpful, we would appreciate it if you could cite this collection in the following form:


    @article{DBLP:journals/corr/abs-2302-09019,
      author       = {Maolin Wang and
                      Yu Pan and
                      Zenglin Xu and
                      Xiangli Yang and
                      Guangxi Li and
                      Andrzej Cichocki},
      title        = {Tensor Networks Meet Neural Networks: {A} Survey and Future Perspectives},
      journal      = {CoRR},
      volume       = {abs/2302.09019},
      year         = {2023}
    }


## Network compression via TNNs

### Tensorial Convolutional Neural Networks
| Paper                                                        | Remarks                                                      | Conference/Journal      | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------------- | ---- |
| Pan et al. "A Unified Weight Initialization Paradigm for Tensorial Convolutional Neural Networks". [[link](https://proceedings.mlr.press/v162/pan22b.html)] | Proposing a universal weight initialization paradigm, which generalizes Xavier and Kaiming methods and can be widely applicable to arbitrary TCNNs. | ICML                | 2022 |
| Ye Liu and Michael K. Ng. "Deep neural network compression by Tucker decomposition with nonlinear response". [[link](https://www.sciencedirect.com/science/article/abs/pii/S0950705122000326)] | Compressing deep neural network with low multilinear rank Tucker format. | Knowledge-Based Systems | 2022 |
| Liu et al. "TT-TSVD: A Multi-modal Tensor Train Decomposition with Its Application in Convolutional Neural Networks for Smart Healthcare".[[link]()] | A tensor train-tensor singular value decomposition (TT-TSVD) algorithm for data reduction and compression of the convolutional neural networks. | TOMM                    | 2022 |
| Ye et al. "Block-term tensor neural networks". [[link](https://www.sciencedirect.com/science/article/pii/S0893608020302045?casa_token=Alj_Cd4nbg4AAAAA:hDwXolzObNziEO3dvoCDDUTZPrOrlVB3sctqpbovfxNIx-z0kvJy11XCK7mPx7M6N_cEMd7rqw)] | Exploring the correlations in the weight matrices, and approximating the weight matrices with the low-rank Block-Term Tucker tensors. | Neural Networks         | 2020 |
| Kossaifi et al. "Tensor regression networks". [[link](https://dl.acm.org/doi/abs/10.5555/3455716.3455839)] | Introducing Tensor Contraction Layers (TCLs) that reduce the dimensionality. | JMLR                    | 2020 |
| Wu et al. "Hybrid tensor decomposition in neural network compression". [[link](https://www.sciencedirect.com/science/article/pii/S0893608020303294?casa_token=K5Je4lJhoqwAAAAA:GmS7dvtKnXpzrNYj4PaHyn6LaPY0YzS5PadxiCiBdUXzUIw_Gl1A6Oe-QuzUuzw9rc_jYhq6gw)] | Introducing the hierarchical Tucker (HT) to investigate its capability in neural network compression. | Neural Networks         | 2020 |
| Kossaifi et al. "Factorized higher-order cnns with an application to spatio-temporal emotion estimation". [[link](http://openaccess.thecvf.com/content_CVPR_2020/html/Kossaifi_Factorized_Higher-Order_CNNs_With_an_Application_to_Spatio-Temporal_Emotion_Estimation_CVPR_2020_paper.html)] | Proposing coined CP-HigherOrder Convolution (HO-CPConv), to spatio-temporal facial emotion analysis. | CVPR                    | 2020 |
| Phan et al. "Stable low-rank tensor decomposition for compression of convolutional neural network".[[link](https://arxiv.org/abs/2008.05441)] | A stable decomposition method CPD-EPC is proposed with a minimal sensitivity design for both CP convolutional layers and hybrid Tucker2-CP convolutional layers. | ECCV                    | 2020 |
| Wang et al. "Concatenated tensor networks for deep multi-task learning". [[link](https://link.springer.com/chapter/10.1007/978-3-030-63823-8_59)] | Introducing a novel Concatenated Tensor Network structure, in particular, Projected Entangled Pair States (PEPS) like structure, into multi-task deep models. | ICONIP                  | 2020 |
| Kossaifi et al. "T-net: Parametrizing fully convolutional nets with a single high-order tensor". [[link](http://openaccess.thecvf.com/content_CVPR_2019/html/Kossaifi_T-Net_Parametrizing_Fully_Convolutional_Nets_With_a_Single_High-Order_Tensor_CVPR_2019_paper.html)] | Proposing to fully parametrize Convolutional Neural Networks (CNNs) with a single highorder, low-rank tucker tensor format. | CVPR                    | 2019 |
| Hayashi et al. "Einconv: Exploring Unexplored Tensor Network Decompositions for Convolutional Neural Networks". [[link](https://proceedings.neurips.cc/paper/2019/file/2bd2e3373dce441c6c3bfadd1daa953e-Paper.pdf)] | Characterizing a decomposition class specific to CNNs by adopting a flexible graphical notation. | NeurIPS                 | 2019 |
| Wang et al. "Wide compression: Tensor ring nets". [[link](http://openaccess.thecvf.com/content_cvpr_2018/html/Wang_Wide_Compression_Tensor_CVPR_2018_paper.html)] | Significantly compressing both the fully connected layers and the convolutional layers of deep networks via Introducing Tensor Ring format. | CVPR                    | 2018 |
| Yang et al. "Deep multi-task representation learning: A tensor factorisation approach". [[link](https://arxiv.org/abs/1605.06391)] | Proposing deep multi-task Tucker models and Tensor Train modesl that learn cross-task sharing structure. | ICLR                    | 2017 |
| Garipov et al. "Ultimate tensorization: compressing convolutional and fc layers alike". [[link](https://arxiv.org/abs/1611.03214)] | Compressing convolutional layers via Tensor Train format.    | Arxiv preprint          | 2016 |
| Novikov et al. "Tensorizing neural networks". [[link](https://proceedings.neurips.cc/paper/5787-tensorizing-neural-networks)] | Converting the dense weight matrices of the fully-connected layers in CNNs to the Tensor Train format. | NeurIPS                 | 2015 |
| Lebedev et al. "Speeding-up convolutional neural networks using fine-tuned CP-decomposition". [[link](https://arxiv.org/pdf/1412.6553.pdf)] | Decomposing the 4D convolution kernel tensor via CP-decomposition. | ICLR                    | 2015 |
| Denton et al. "Exploiting Linear Structure Within Convolutional Networks for Efficient Evaluation". [[link](https://arxiv.org/pdf/1404.0736.pdf)] | Speeding up the test-time evaluation of large convolutional networks via CP-decomposition. | NeurIPS                 | 2014 |

### Tensorial Recurrent Neural Networks

| Paper                                                        | Remarks                                                      | Conference/Journal                            | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | --------------------------------------------- | ---- |
| Yin et al. "Towards extremely compact rnns for video recognition with fully decomposed hierarchical tucker structure". [[link](http://openaccess.thecvf.com/content/CVPR2021/html/Yin_Towards_Extremely_Compact_RNNs_for_Video_Recognition_With_Fully_Decomposed_CVPR_2021_paper.html)] | Proposing to develop extremely compact RNN models with fully decomposed hierarchical Tucker structure. | CVPR                                          | 2021 |
| Wang et al. "Kronecker CP decomposition with fast multiplication for compressing RNNs". [[link](https://ieeexplore.ieee.org/abstract/document/9540760/?casa_token=DK21LnwuNUIAAAAA:GQBvHt8r6r9LjA63GoAMed-Rp4ifGf38IEAfruOT9qEFMViqFpXPc29WPFiwRHSmZ4QFNO9XWQ)] | Compressing RNNs based on a novel Kronecker CANDECOMP/PARAFAC decomposition, which is derived from Kronecker tensor decomposition. | TNNLS                                         | 2021 |
| Kossaifi et al. "Tensor regression networks". [[link](https://dl.acm.org/doi/abs/10.5555/3455716.3455839)] | Introducing Tensor Contraction Layers (TCLs) that reduce the dimensionality. | JMLR                                          | 2020 |
| Ye et al. "Block-term tensor neural networks". [[link](https://www.sciencedirect.com/science/article/pii/S0893608020302045?casa_token=Alj_Cd4nbg4AAAAA:hDwXolzObNziEO3dvoCDDUTZPrOrlVB3sctqpbovfxNIx-z0kvJy11XCK7mPx7M6N_cEMd7rqw)] | Exploring the correlations in the weight matrices, and approximating the weight matrices with the low-rank Block-Term Tucker tensors. | Neural Networks                               | 2020 |
| Su et al. "Convolutional tensor-train LSTM for spatio-temporal learning". [[link](https://proceedings.neurips.cc/paper/2020/hash/9e1a36515d6704d7eb7a30d783400e5d-Abstract.html)] | Proposing a novel tensor-train module that performs prediction by combining convolutional features across time. | NeurIPS                                       | 2020 |
| Wu et al. "Hybrid tensor decomposition in neural network compression". [[link](https://www.sciencedirect.com/science/article/pii/S0893608020303294?casa_token=K5Je4lJhoqwAAAAA:GmS7dvtKnXpzrNYj4PaHyn6LaPY0YzS5PadxiCiBdUXzUIw_Gl1A6Oe-QuzUuzw9rc_jYhq6gw)] | Introducing the hierarchical Tucker (HT) to investigate its capability in neural network compression. | Neural Networks                               | 2020 |
| Tjandra et al. "Recurrent Neural Network Compression Based on Low-Rank Tensor Representation". [[link](https://www.jstage.jst.go.jp/article/transinf/E103.D/2/E103.D_2019EDP7040/_article/-char/ja/)] | Proposing to use Tensor Train formats to re-parameterize the Gated Recurrent Unit (GRU) RNN. | IEICE Transactions on Information and Systems | 2019 |
| Pan et al. "Compressing recurrent neural networks with tensor ring for action recognition". [[link](https://ojs.aaai.org/index.php/AAAI/article/view/4393)] | Proposing a novel compact LSTM model, named as TR-LSTM, by utilizing the low-rank tensor ring decomposition (TRD) to reformulate the input-to-hidden transformation. | AAAI                                          | 2019 |
| Jose et al. "Kronecker recurrent units". [[link](http://proceedings.mlr.press/v80/jose18a.html)] | Achieving a parameter efficiency in RNNs through a Kronecker factored recurrent matrix. | ICML                                          | 2018 |
| Ye et al. "Learning compact recurrent neural networks with block-term tensor decomposition". [[link](http://openaccess.thecvf.com/content_cvpr_2018/html/Ye_Learning_Compact_Recurrent_CVPR_2018_paper.html)] | Proposing to apply Block-Term tensor decomposition to reduce the parameters of RNNs and improves their training efficiency. | CVPR                                          | 2018 |
| Yang et al. "Tensor-train recurrent neural networks for video classification". [[link](http://proceedings.mlr.press/v70/yang17e)] | Factorizing the input-to-hidden weight matrix in RNNs using Tensor-Train decomposition. | ICML                                          | 2017 |
| Kossaifi et al. "Tensor Contraction Layers for Parsimonious Deep Nets". [[link](https://arxiv.org/abs/1706.00439)] | Proposing the Tensor Contraction Layer (TCL), the first attempt to incorporate tensor contractions as end-to-end trainable neural network layers. | CVPR-Workshop                                 | 2017 |

### Tensorial Transformer

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Vasilescu et al."Causal Deep Learning: Causal Capsules and Tensor Transformers". [[link](https://arxiv.org/pdf/2301.00314.pdf)] | Forward causal questions are addressed with a neural network architecture composed of causal capsules and a tucker format tensor transformer. | Arxiv preprint     | 2023 |
| Liu et al. "Tuformer: Data-driven Design of Transformers for Improved Generalization or Efficiency". [[link](https://openreview.net/forum?id=V0A5g83gdQ_)] | Proposing a novel design by allowing data-driven weights across heads via low rank tensor diagrams. | ICLR               | 2022 |
| Ren et al. "Exploring extreme parameter compression for pre-trained language models". [[link](https://arxiv.org/abs/2205.10036)] | Proposing to use Tucker formats to improve the effectiveness and efficiency during compression of Transformers. | ICLR               | 2022 |
| Li et al. "Hypoformer: Hybrid decomposition transformer for edge-friendly neural machine translation". [[link](https://preview.aclanthology.org/emnlp-22-ingestion/2022.emnlp-main.475/)] | Compressing and accelerating Transformer via a Hybrid TensorTrain (HTT) decomposition. | EMNLP              | 2022 |
| Liu et al. "Enabling lightweight fine-tuning for pre-trained language model compression based on matrix product operators". [[link](https://arxiv.org/abs/2106.02205)] | Proposing a novel fine-tuning strategy by only updating the parameters from the auxiliary tensors, and design an optimization algorithm for MPO-based approximation over stacked network architectures. | ACL/IJCNLP         | 2021 |
| Ma et al. "A tensorized transformer for language modeling". [[link](https://proceedings.neurips.cc/paper/8495-a-tensorized-transformer-for-language-modeling)] | Proposing a novel self-attention model (namely Multi-linear attention) with Block-Term Tensor Decomposition. | NeurIPS            | 2019 |

### Tensorial Graph Neural Networks

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Hua et al. "High-Order Pooling for Graph Neural Networks with Tensor Decomposition". [[link](https://openreview.net/pdf?id=N7-EIciq3R)] | Proposing the highly expressive Tensorized Graph Neural Network (tGNN) to model high-order non-linear node interactions. | NeurIPS            | 2022 |
| "Multi-view tensor graph neural networks through reinforced aggregation".[[link]()] | A Tucker format structure is applied to extract the graph structure features in the common feature space, was introduced to capture the potential high order correlation information in multi-view graph learning tasks | TKDE               | 2022 |
| Baghershahi et al. "Efficient Relation-aware Neighborhood Aggregation in Graph Neural Networks via Tensor Decomposition". [[link](https://arxiv.org/abs/2212.05581)] | Introducing a general knowledge graph encoder incorporating tensor decomposition in the aggregation function. | Arxiv preprint     | 2022 |
| Jia et al. "Dynamic spatiotemporal graph neural network with tensor network". [[link](https://arxiv.org/abs/2003.08729)] | Exploring the entangled correlations in spatial tensor graph and  temporal tensor graph  by Projected Entangled Pair States (PEPS). | Arxiv preprint     | 2020 |

### Tensorial Restricted Boltzmann Machine

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Ju et al. "Tensorizing Restricted Boltzmann Machine". [[link](https://dl.acm.org/doi/abs/10.1145/3321517)] | Proposing TT-RBM which both visible and hidden variables are in tensorial form and are connected by a parameter matrix in tensor train formats. | TKDD               | 2019 |
| Chen et al. "Matrix Product Operator Restricted Boltzmann Machines". [[link](https://arxiv.org/abs/1811.04608)] | Proposing the matrix product operator RBM that utilizes a tensor network generalization of Mv/TvRBM. | IJCNN              | 2019 |
| Wang et al. "Tensor ring restricted Boltzmann machines". [[link](https://ieeexplore.ieee.org/abstract/document/8852432/?casa_token=kZTMUrKchnoAAAAA:aU1uQZyZKHpRXnvnYpqaXLs_JLqPQKLGx67D1vCRB63yTo36Mk0OG5Ldx27s3ZCgpVP-BJqejw)] | Proposing a tensor-input RBM model, which employs the tensor-ring (TR) decomposition structure to naturally represent the high-order relationship. | IJCNN              | 2019 |
| Qi et al. "Matrix variate restricted Boltzmann machine". [[link](https://ieeexplore.ieee.org/abstract/document/7727225/?casa_token=YodFcxTprtYAAAAA:XBX8NxOCVqFkS9NVEVHpnxirp7vq6BdbRvr6ka3SCvg0Y3Oh12N8vs5T2XiI3fOqy4KNNQKnfA)] | Proposing a bilinear connection between matrix variate visible layer and matrix variate hidden layer. | IJCNN              | 2016 |
| Nguyen et al. "Tensor-variate restricted Boltzmann machines". [[link](https://ojs.aaai.org/index.php/AAAI/article/view/9553)] | Generalizing RBMs to capture the multiplicative interaction between data modes and the latent variables via CP decomposition. | AAAI               | 2015 |

## Information Fusion via TNNs

### Tensor Fusion Layer-Based Methods

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Hou et al. "Deep multimodal multilinear fusion with high-order polynomial pooling". [[link](https://proceedings.neurips.cc/paper/2019/hash/f56d8183992b6c54c92c16a8519a6e2b-Abstract.html)] | Proposing a polynomial tensor pooling (PTP) block for integrating multimodal features by considering high-order moments. | NeurIPS            | 2019 |
| Liu et al. "Efficient low-rank multimodal fusion with modality-specific factors". [[link](https://arxiv.org/abs/1806.00064)] | Proposing the low-rank method, which performs multimodal fusion using low-rank tensors to improve efficiency. | ACL                | 2018 |
| Zadeh et al. "Tensor fusion network for multimodal sentiment analysis". [[link](https://aclanthology.org/D17-1115/)] | Introducing a novel model, termed Tensor Fusion Network, which learns both intra-modality and inter-modality dynamics. | EMNLP              | 2017 |

### Multimodal Pooling-Based Methods

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Do et al. "Compact trilinear interaction for visual question answering". [[link](http://openaccess.thecvf.com/content_ICCV_2019/html/Do_Compact_Trilinear_Interaction_for_Visual_Question_Answering_ICCV_2019_paper.html)] | Introducing a multimodal tensor-based PARALIND decomposition which efficiently parameterizes trilinear teraction between inputs. | CVPR               | 2019 |
| Fukui et al. "Multimodal compact bilinear pooling for visual question answering and visual grounding". [[link](https://arxiv.org/abs/1606.01847)] | Proposing utilizing Multimodal Compact Bilinear pooling (MCB) to efficiently and expressively combine multimodal features. | EMNLP              | 2016 |
| Kim et al. "Hadamard product for low-rank bilinear pooling". [[link](https://arxiv.org/abs/1610.04325)] | Proposing low-rank bilinear pooling using Hadamard product for an efficient attention mechanism of multimodal learning. | Arxiv preprint     | 2016 |
| Ben-Younes et al. "Mutan: Multimodal tucker fusion for visual question answering". [[link](http://openaccess.thecvf.com/content_iccv_2017/html/Ben-younes_MUTAN_Multimodal_Tucker_ICCV_2017_paper.html)] | Proposing a multimodal tensor-based Tucker decomposition to efficiently parametrize bilinear interactions between visual and textual representations. | CVPR               | 2017 |

## Quantum Circuit Simulation on TNNs

### Classical Data's Quantum State Embedding

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Miller et al. "Tensor Networks for Probabilistic Sequence Modeling". [[link](https://arxiv.org/abs/2003.01039)] | Introducing a novel generative algorithm giving trained u-MPS the ability to efficiently sample from a wide variety of conditional distributions, each one defined by a regular expression. | AISTATS            | 2021 |
| Li et al. "CNM: An interpretable complex-valued network for matching". [[link](https://arxiv.org/abs/1904.05298)] | Unifing different linguistic units in a single complex-valued vector space. | NAACL              | 2019 |
| Zhang et al. "A quantum many-body wave function inspired language modeling approach". [[link](https://dl.acm.org/doi/abs/10.1145/3269206.3271723?casa_token=HM9Mc9HHEaoAAAAA:ZO_Ug1U3OUWy8RTD0AfLewE6fJSmDpyAQ7U_BXEQrqNVuNsf9XDgTwfBHuHmQbMP2A1-lC5FszLBeg)] | Considering word embeddings as a kind of global dependency information and integrated the quantum-inspired idea in a neural network architecture. | CIKM               | 2018 |
| Stoudenmire et al. "Supervised learning with tensor networks". [[link](https://proceedings.neurips.cc/paper/2016/hash/5314b9674c86e3f9d1ba25ef9bb32895-Abstract.html)] | Introducing a framework for applying quantum-inspired tensor networks to image classification. | NeurIPS            | 2016 |

### Quantum Embedded Data Processing

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Liu et al. "Tensor networks for unsupervised machine learning".[[link](https://journals.aps.org/pre/pdf/10.1103/PhysRevE.107.L012103?casa_token=890wuIcFVkkAAAAA%3A3iMJpRTj9z4bPdbaijGzHtNtzBjktJx9rWjovmB406DaYXZKsT_zAxYfpGseacdLI2ubED4_Iga-pN0)] | A tensor network model combined matrix product states from quantum many-body physics and autoregressive modeling from machine learning. | Physical Review E  | 2023 |
| Miller et al. "Tensor Networks for Probabilistic Sequence Modeling". [[link](https://arxiv.org/abs/2003.01039)] | Introducing a novel generative algorithm giving trained u-MPS the ability to efficiently sample from a wide variety of conditional distributions, each one defined by a regular expression. | AISTATS            | 2021 |
| Glasser et al. "Expressive power of tensor-network factorizations for probabilistic modeling". [[link](https://proceedings.neurips.cc/paper/2019/hash/b86e8d03fe992d1b0e19656875ee557c-Abstract.html)] | Introducing locally purified states (LPS), a new factorization inspired by techniques for the simulation of quantum systems, with provably better expressive power than all other representations considered. | NeurIPS            | 2019 |
| Cheng et al. "Tree tensor networks for generative modeling". [[link](https://dl.acm.org/doi/abs/10.1145/3269206.3271723?casa_token=HM9Mc9HHEaoAAAAA:ZO_Ug1U3OUWy8RTD0AfLewE6fJSmDpyAQ7U_BXEQrqNVuNsf9XDgTwfBHuHmQbMP2A1-lC5FszLBeg)] | Designing the tree tensor network to utilize the 2-dimensional prior of the natural images and develop sweeping learning and sampling algorithms. | Physical Review B  | 2019 |
| Han et al. "Unsupervised generative modeling using matrix product states". [[link](https://arxiv.org/abs/1904.05298)] | Proposing a generative model using matrix product states, which is a tensor network originally proposed for describing (particularly one-dimensional) entangled quantum states. | Physical Review X  | 2018 |
| Edwin Stoudenmire and David J. Schwab. "Supervised learning with tensor networks". [[link](https://proceedings.neurips.cc/paper/2016/hash/5314b9674c86e3f9d1ba25ef9bb32895-Abstract.html)] | Introducing a framework for applying quantum-inspired tensor networks to image classification. | NeurIPS            | 2016 |

### Convolutional Arithmetic Circuits 

| Paper                                                        | Remarks                                                      | Conference/Journal | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------ | ---- |
| Zhang et al. "A Generalized Language Model in Tensor Space". [[link](https://ojs.aaai.org/index.php/AAAI/article/view/4735/4613)] | Proposing a language model named Tensor Space Language Model (TSLM), by utilizing tensor networks and tensor decomposition. | AAAI               | 2019 |
| Levine et al. "Quantum Entanglement in Deep Learning Architectures". [[link](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=DmzoCRMAAAAJ&sortby=pubdate&citation_for_view=DmzoCRMAAAAJ:Zph67rFs4hoC)] | Identifying an inherent re-use of information in the network operation as a key trait which distinguishes them from standard Tensor Network based representations. | PRL                | 2019 |
| Zhang et al. "A quantum many-body wave function inspired language modeling approach". [[link](https://dl.acm.org/doi/abs/10.1145/3269206.3271723?casa_token=u2aWWBq46SsAAAAA:BrPcodbo0cGgZV1SNzFBWS5Qx6gHVi8SoSeMOayaa_N-2vNzhI3q8NwT4c4rzTnwsvljcXdLuqBf3Q)] | Proposing a Quantum Many-body Wave Function (QMWF) inspired language modeling approach. | CIKM               | 2018 |
| Levine et al. "Deep Learning and Quantum Entanglement: Fundamental Connections with Implications to Network Design". [[link](https://arxiv.org/abs/1704.01552)] | Showing an equivalence between the function realized by a deep convolutional arithmetic circuit (ConvAC) and a quantum many-body wave function. | ICLR               | 2018 |
| Cohen et al. "On the Expressive Power of Deep Learning: A Tensor Analysis". [[link](https://arxiv.org/abs/1509.05009)] | Showing that a shallow network corresponds to CP (rank-1) decomposition, whereas a deep network corresponds to Hierarchical Tucker decomposition. | COLT               | 2016 |
| Nadav Cohen and Amnon Shashua. "Convolutional Rectifier Networks as Generalized Tensor Decompositions". [[link](https://proceedings.mlr.press/v48/cohenb16.html)] | Describing a construction based on generalized tensor decompositions, that transforms convolutional arithmetic circuits into convolutional rectifier networks. | ICML               | 2016 |

## Training Strategy

### Stable Training

| Paper                                                        | Remarks                                                      | Conference/Journal  | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------- | ---- |
| Pan et al. "A Unified Weight Initialization Paradigm for Tensorial Convolutional Neural Networks". [[link](https://proceedings.mlr.press/v162/pan22b.html)] | Proposing a universal weight initialization paradigm, which generalizes Xavier and Kaiming methods and can be widely applicable to arbitrary TCNNs. | ICML                | 2022 |
| Panagakis et al. "Tensor methods in computer vision and deep learning". [[link](https://ieeexplore.ieee.org/abstract/document/9420085/?casa_token=sxgy4gD8rAEAAAAA:xJs6F0f9CwbkUSvpiWlIhE5GKu_01eOs-XmSH07N5zc2YHyotzMryxnpm1D0egcpCfE3ZfhnuQ)] | Proposing a mixed-precision strategy to trade off time cost and numerical stability. | Proceedings of IEEE | 2021 |

### Rank Selection

| Paper                                                        | Remarks                                                      | Conference/Journal                                   | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------------------------------- | ---- |
| Sobolev et al. "PARS: Proxy-Based Automatic Rank Selection for Neural Network Compression via Low-Rank Weight Approximation".[[link]()] | A proxy-based Bayesian optimization approach to find the best combination of ranks for neural network (NN) compression. | Mathematics                                          | 2022 |
| Sedighin et al. "Adaptive Rank Selection for Tensor Ring Decomposition".[[link](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9321501&casa_token=Kccslqd3PWgAAAAA:PVjEiZd5XzPT0-AW9YHTRqY9OLTs-QeZX6_K7Ubuk7h0pxp1huN5EsjvwOuw3sQOAdF40jd-vw&tag=1)] | An adaptive rank search framework for TR format in which TR ranks gradually increase in each iteration rather than being predetermined in advance. | IEEE Journal of Selected Topics in Signal Processing | 2021 |
| Li et al. "Heuristic rank selection with progressively searching tensor ring network". [[link](https://link.springer.com/article/10.1007/s40747-021-00308-x)] | Proposing a novel progressive genetic algorithm named progressively searching tensor ring network search (PSTRN), which has the ability to find optimal rank precisely and efficiently. | Complex & Intelligent Systems                        | 2021 |
| Cole Hawkins and Zheng Zhang. "Bayesian tensorized neural networks with automatic rank selection". [[link](https://www.sciencedirect.com/science/article/pii/S0925231221006950?casa_token=giYJF1h4jC8AAAAA:aavT0_ZN_pmPnDqbsJM6K1MYUiLbxI0oWXPyODLa3wyzXUUJWiwhNQ7alu2Cju201QhpRBk0gg)] | Proposing approaches for posterior density calculation and maximum a posteriori (MAP) estimation for the end-to-end training of our tensorized neural network. | Neurocomputing                                       | 2021 |
| Yin et al. "Towards efficient tensor decomposition-based dnn model compression with optimization framework". [[link](http://openaccess.thecvf.com/content/CVPR2021/html/Yin_Towards_Efficient_Tensor_Decomposition-Based_DNN_Model_Compression_With_Optimization_Framework_CVPR_2021_paper.html)] | Proposing a systematic framework for tensor decomposition-based model compression using Alternating Direction Method of Multipliers(ADMM). | CVPR                                                 | 2021 |
| Cheng et al. "A novel rank selection scheme in tensor ring decomposition based on reinforcement learning for deep neural networks". [[link](https://ieeexplore.ieee.org/abstract/document/9053292/?casa_token=h_XCI5YFy6EAAAAA:QDydsaV3VIhlmQzyO_MbHg269K_qy0lwObfZC7bbnJ2YmqBN8DgUQ4tlO2aTx8zfqMqR4guLOg)] | Proposing a novel rank selection scheme, which is inspired by reinforcement learning, to automatically select ranks in recently studied tensor ring decomposition in each convolutional layer. | ICASSP                                               | 2020 |
| Kim et al. "Compression of deep convolutional neural networks for fast and low power mobile applications". [[link](https://arxiv.org/abs/1511.06530)] | Deriving an approximate rank by employing the Bayesian matrix factorization (BMF) to an unfolding weight tensor. | ICLR                                                 | 2016 |
| Zhao et al. "Bayesian CP factorization of incomplete tensors with automatic rank determination". [[link](https://ieeexplore.ieee.org/abstract/document/7010937/?casa_token=kGYY-jf-OwYAAAAA:F6_6WuCgR9HxtAtYVIXJ2HfnJuoCcvFdVnccJK1ZU73J23EFgaBfh1jKrC5o8DGfryO9LHUCTA)] | Formulating CP factorization using a hierarchical probabilistic model and employ a fully Bayesian treatment. | TPAMI                                                | 2015 |

### Hardware Training

| Paper                                                        | Remarks                                                      | Conference/Journal                                           | Year |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ---- |
| Kao et al. "Hardware Acceleration in Large-Scale Tensor Decomposition for Neural Network Compression". [[link](https://ieeexplore.ieee.org/abstract/document/9859440/?casa_token=-GX0RyAssGoAAAAA:7ZW1LAw5_oO14VD13-IRCLGFClH8xeKC8PTzbHFPHyerGWmjmqtNn2Tq4YTH-7m8yE20Oad1YQ)] | Proposing an energy-efficient hardware accelerator that implements randomized CPD in large-scale tensors for neural network compression. | MWSCAS                                                       | 2022 |
| Qu et al. "Hardware-Enabled Efficient Data Processing with Tensor-Train Decomposition". [[link](https://ieeexplore.ieee.org/abstract/document/9351565/?casa_token=BZLiwk-m9L8AAAAA:1lQdMChb-Y3-D0s_n2GvMsZxpXv0wGsiWSIxbI1fQ8jjhR6YXzeJ2vAw6R0FwYG-scqK9LP3LA)] | Proposing an algorithm-hardware co-design with customized architecture, namely, TTD Engine to accelerate TTD. | IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems | 2021 |
| Deng et al. "TIE: Energy-efficient tensor train-based inference engine for deep neural network". [[link](https://dl.acm.org/doi/abs/10.1145/3307650.3322258?casa_token=WzxcffHLph8AAAAA:7n3ABF-ZbVwK9iyyri5nQMdlGzDPega8kGYG0Sde8L18g49X-umS9XW_RORYaxqCIC_AAPFC9ov9KQ)] | Developing a computation-efficient inference scheme for TT-format DNN. | ISCA                                                         | 2019 |
| Huang et al. "LTNN: An energy-efficient machine learning accelerator on 3D CMOS-RRAM for layer-wise tensorized neural network". [[link](https://ieeexplore.ieee.org/abstract/document/8226058/?casa_token=275Xgo_oYxYAAAAA:qMupRjne2MLHWwwZrQbDEK0U1dXYJC7_omIrO8EvZulYhXBmWJst8bR7_K1XdvwBSl-PutrJBA)] | Mapping TNNs  to a 3D CMOS-RRAM based accelerator with significant bandwidth boosting from vertical I/O connections. | SOCC                                                         | 2017 |

## 



## Toolboxes

### Basic Tensor Operation

| Name                                                         | Remarks                                                      | Backends                                                     |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| [Tensorly](http://tensorly.org/stable/index.html)            | TensorLy is open-source, actively maintained and easily extensible. TensorLy provides all the utilities to easily use tensor methods from core tensor operations and tensor algebra to tensor decomposition and regression. | Python (NumPy, PyTorch, TensorFlow, JAX, Apache MXNet and CuPy) |
| [TensorNetwork](https://github.com/google/TensorNetwork)     | TensorNetwork is an open source library for implementing tensor network algorithms. | Python (TensorFlow, JAX, PyTorch, and Numpy)                 |
| [Tensortools](https://github.com/neurostatslab/tensortools)  | TensorTools is a bare bones Python package for fitting and visualizing canonical polyadic (CP) tensor decompositions of higher-order data arrays. | Python (NumPy)                                               |
| [TnTorch](https://github.com/rballester/tntorch)             | TnTorch is a PyTorch-powered library for tensor modeling and learning that features transparent support for the the tensor train (TT) model, CANDECOMP/PARAFAC (CP), the Tucker model, and more. | Python (Pytorch)                                             |
| [TorchMPS](https://github.com/jemisjoky/TorchMPS)            | TorchMPS is a framework for working with matrix product state (also known as MPS or tensor train) models within Pytorch. | Python (Pytorch)                                             |
| [T3F](https://github.com/Bihaqo/t3f)                         | T3F supports GPU execution, batch processing, automatic differentiation, and versatile functionality for the Riemannian optimization framework. | Python (Tensorflow)                                          |
| [TensorD](https://github.com/Large-Scale-Tensor-Decomposition/tensorD) | TensorD provides basic decomposition methods, such as Tucker decomposition and CANDECOMP/PARAFAC (CP) decomposition, as well as new decomposition methods developed recently, for example, Pairwise Interaction Tensor Decomposition. | Python (Tensorflow)                                          |
| [ITensor](https://itensor.org/)                              | ITensor is a system for programming tensor network calculations with an interface modeled on tensor diagram notation, which allows users to focus on the connectivity of a tensor network without manually bookkeeping tensor indices. | C++/Julia                                                    |
| [TenDeC++](https://github.com/XiaoYangLiu-FinRL/TensorLet_in_C_C_PlusPlus) | TenDeC++ implements four popular tensor decomposition methods, CANDECOMP/PARAFAC (CP) decomposition, Tucker decomposition, t-SVD, and Tensor-Train (TT) decomposition. | C++                                                          |
| [TensorToolbox](https://www.tensortoolbox.org/)              | Tensor Toolbox provides a suite of tools for working with multidimensional or N-way arrays. | Matlab                                                       |
| [TT-Toolbox](https://github.com/oseledets/TT-Toolbox)        | he TT-Toolbox is a MATLAB implementation of basic operations with tensors in TT-format. | Matlab                                                       |
| [OSTD](https://github.com/andrewssobral/ostd)                | Online Stochastic Tensor Decomposition for Background Subtraction in Multispectral Video Sequences. | Matlab                                                       |
| [Scikit-TT](https://github.com/PGelss/scikit_tt)             | Scikit-TT provides a powerful TT class as well as different modules comprising solvers for algebraic problems, the automatic construction of tensor trains, and data-driven methods. | Python                                                       |

### Deep Model Implementation

| Name                                              | Remarks                                                      | Backends |
| ------------------------------------------------- | ------------------------------------------------------------ | -------- |
| [Tensorly-Torch](https://tensorly.org/torch/dev/) | TensorLy-Torch is a PyTorch only library that builds on top of [TensorLy](http://tensorly.org/dev) and provides out-of-the-box tensor layers. It comes with all batteries included and tries to make it as easy as possible to use tensor methods within your deep networks. | Python (Pytorch)  |
| [TedNet](https://github.com/tnbar/tednet)         | TedNet implements 5 kinds of tensor decomposition (i.e., CANDECOMP/PARAFAC (CP), Block-Term Tucker (BTT), Tucker-2, Tensor Train (TT) and Tensor Ring (TR) on traditional deep neural layers. | Python (Pytorch)  |

### Quantum Tensor Simulation

| Name                                                       | Remarks                                                      | Backends  |
| ---------------------------------------------------------- | ------------------------------------------------------------ | --------- |
| [TensorToolbox](https://www.tensortoolbox.org/)            | Tensor Toolbox provides a suite of tools for working with multidimensional or N-way arrays. | Matlab    |
| [ITensor](https://itensor.org/)                            | ITensor is a system for programming tensor network calculations with an interface modeled on tensor diagram notation, which allows users to focus on the connectivity of a tensor network without manually bookkeeping tensor indices. | C++/Julia |
| [Yao](http://quantum-journal.org/papers/q-2020-10-11-341/) | Yao is an extensible, efficient open-source framework for quantum algorithm design. | Python    |
| [lambeq](https://github.com/CQCL/lambeq)                   | Lambeq is a toolkit for quantum natural language processing. | Python    |
| [TeD-Q](https://github.com/amore-upf/ted-q)                | TeD-Q provides an additional layer of annotations to the existing dataset. | Python    |

### 

