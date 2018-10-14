lines="""
[Propagating LSTM: 3D Pose Estimation based on Joint Interdependency](http://openaccess.thecvf.com/content_ECCV_2018/papers/Kyoungoh_Lee_Propagating_LSTM_3D_ECCV_2018_paper.pdf) (Sep 2018)
[Hierarchical graphical-based human pose estimation via local multi-resolution convolutional neural network](https://aip.scitation.org/doi/full/10.1063/1.5024463) (Feb 2018)
[A Deep Learning Based Method For 3D Human Pose Estimation From 2D Fisheye Images](https://pdfs.semanticscholar.org/8ff8/840a418f9202a33fae08997afcd2da6b19f2.pdf) (Mar 2018)
[A Dual-Source Approach for 3D Human Pose Estimation from a Single Image] (https://arxiv.org/pdf/1705.02883.pdf) (May 2017)
[Human pose estimation method based on single depth image](http://digital-library.theiet.org/content/journals/10.1049/iet-cvi.2017.0536) (Sep 2018)
[Human Pose Retrieval for Image and Video collections](http://ir.inflibnet.ac.in:8080/jspui/handle/10603/168240) (Oct 2017)
###### A search engine for dancers
[Stacked dense-hourglass networks for human pose estimation](https://www.ideals.illinois.edu/handle/2142/101155) (Apr 2018)
[Deeply Learned Compositional Models for Human Pose Estimation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Wei_Tang_Deeply_Learned_Compositional_ECCV_2018_paper.pdf) (Sep 2018)
[Exploiting temporal information for 3D human pose estimation](http://openaccess.thecvf.com/content_ECCV_2018/html/Mir_Rayat_Imtiaz_Hossain_Exploiting_temporal_information_ECCV_2018_paper.html) (Sep 2018)
[3-D Reconstruction of Human Body Shape from a Single Commodity Depth Camera](https://ieeexplore.ieee.org/abstract/document/8371630) (Jun 2018)
[Human Pose As Calibration Pattern; 3D Human Pose Estimation With Multiple Unsynchronized and Uncalibrated Cameras](http://openaccess.thecvf.com/content_cvpr_2018_workshops/w34/html/Takahashi_Human_Pose_As_CVPR_2018_paper.html) (Jun 2018)
[Human Pose Estimation Based on Deep Neural Network](https://ieeexplore.ieee.org/abstract/document/8455245) (Jul 2018)
[A Review of Human Pose Estimation from Single Image](https://ieeexplore.ieee.org/abstract/document/8455796) (Jul 2018)
[Hockey Pose Estimation and Action Recognition using Convolutional Neural Networks to Ice Hockey][https://uwspace.uwaterloo.ca/handle/10012/13835] (Sep 2018)
[Learning Robust Features and Latent Representations for Single View 3D Pose Estimation of Humans and Objects]() (Sep 2018)
[Domain Transfer for 3D Pose Estimation from Color Images without Manual Annotations](https://arxiv.org/pdf/1810.03707v1) (Oct 2018)
[Context-Aware Deep Spatio-Temporal Network for Hand Pose Estimation from Depth Images](https://arxiv.org/pdf/1810.02994v1) (Oct 2018)
[A Unified Framework for Multi-View Multi-Class Object Pose Estimation](https://arxiv.org/pdf/1803.08103v2) (Mar 2018)
[Cascaded Pyramid Network for 3D Human Pose Estimation Challenge](https://arxiv.org/pdf/1810.01616v1) (Oct 2018)

"""
import datetime as date; [print(i) for i in sorted([i.rstrip() for i in lines.splitlines() if i and i[0]=='['],key=lambda x: date.datetime.strptime(x[-9:-1], "%b %Y"), reverse=True)]

"""
                       ;
import datetime as date  [print(i) for i in                                                                                                                                           ]
                                            sorted(                                                           , key=                                                   , reverse=True)
                                                   [i          for i in lines              if                ]      lambda x: date.datetime
                                                     .rstrip()               .splitlines()    i and i[0]=='['                              .strptime(        ,        )
                                                                                                                                                     x[-9:-1]  "%b %Y"
"""
#https://repl.it/repls/WirelessHappyBoard
