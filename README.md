# Human Pose Estimation from RGB Camera - The repo
In recent years, tremendous amount of progress is being made in the field of Human Pose Estimation from RGB Camera, which is an interdisciplinary field that fuses computer vision, deep/machine learning and anatomy. This repo is for my study notes and will be used as a place for triaging new research papers. 

## Get Involved
To make it a collaborative project, you may add content through pull requests or open an issue to let me know. 

## Table of Contents

I'll use the following icons and dimensions to differentiate the approaches:

- Time Dimension
	- :camera: Single-Shot 
	- :movie_camera: Video/Real-Time
- Number of People
	- :one: Single-Person
	- :1234: Multi-Person
- Spatial Dimensions
	- :door: 2D Models
	- :package: 3D Models
- Positioning
	- :running: 3rd person
	- :girl: Ego-centric

## Projects and papers

<a name="Fall 2018"/>

### Fall 2018

<b>:camera::one::package:[3D Human Pose Estimation Using Stochastic Optimization In Real Time](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8451427) (Oct 2018)</b>
###### Again and again, This is depth based, but the process of using iterations can be applied to pure RGB

<b>:camera::one::package:[Deep 3D Human Pose Estimation Under Partial Body Presence
](https://ieeexplore.ieee.org/document/8451031) (Oct 2018)</b>
###### My legs have been chopped off

<b>:camera::one::package:[Adversarial 3D Human Pose Estimation via Multimodal Depth Supervision](https://arxiv.org/pdf/1809.07921v1.pdf) (Sep 2018) </b>
###### Continuation of FBI work, also got multimodal netowrk now.

<b>:camera::one::package:[3D Ego-Pose Estimation via Imitation Learning](http://openaccess.thecvf.com/content_ECCV_2018/papers/Ye_Yuan_3D_Ego-Pose_Estimation_ECCV_2018_paper.pdf) (Sep 2018) </b>
###### Headcam, they use a very complicated ragdoll, also, just walking

<b>:camera::camera::one::package:[3D Human Pose Estimation with Siamese Equivariant Embedding](https://arxiv.org/pdf/1809.07217.pdf) (Sep 2018) </b>
###### Lets compare answers after doing our homework.

<b>:door:[Dense Pose Transfer](https://arxiv.org/pdf/1809.01995.pdf) (Sep 2018) </b>
###### Coloring book the shape of a person, machine uses imagination to fill in the details, then animates it's paper statue

<b>:camera::one::package:[Synthetic Occlusion Augmentation with Volumetric Heatmaps for the 2018 ECCV PoseTrack Challenge on 3D Human Pose Estimation](https://arxiv.org/pdf/1809.04987v1.pdf) (Sep 2018) </b>
###### They block their face and body with cheap photoshop techniques, then the machine has to "x-ray" through all that

<a name="Summer 2018"/>

### Summer 2018

<b>:camera::one::package:[Neural Body Fitting: Unifying Deep Learning and Model-Based Human Pose and Shape Estimation](https://arxiv.org/pdf/1808.05942.pdf) (Aug 2018) </b> [[CODE]](http://github.com/mohomran/neural_body_fitting)
###### Color Me Rad

<b>:camera::1234::package: [Single-Shot Multi-Person 3D Body Pose Estimation From Monocular RGB Input](https://arxiv.org/pdf/1712.03453.pdf) (Aug 2018)</b>
###### They use a ORPM, whatever that means. And they have some very obviouly green screened images.

<b>:camera::one::package: [Rethinking Pose in 3D: Multi-stage Refinement and Recovery for Markerless Motion Capture](https://arxiv.org/pdf/1808.01525v1.pdf) (Aug 2018)</b>

<b>:camera::one::package:[3D Human Pose Estimation with Relational Networks](https://arxiv.org/pdf/1805.08961v2.pdf) (Jul 2018) </b>
###### Back bone connected to the shoulder bone, shoulder bone connected to the neck bone...

:door:[Human Pose Estimation with Parsing Induced Learner](http://openaccess.thecvf.com/content_cvpr_2018/papers/Nie_Human_Pose_Estimation_CVPR_2018_paper.pdf) (Jun 2018)

<b>:camera:one:package:[FBI-Pose: Towards Bridging the Gap between 2D Images and 3D Human Poses using Forward-or-Backward Information](https://arxiv.org/pdf/1806.09241) (Jun 2018) </b>
###### Anderson Silva's broken bent leg

<a name="Spring 2018"/>

### Spring 2018

<b>:package:[DRPose3D: Depth Ranking in 3D Human Pose Estimation](https://arxiv.org/pdf/1805.08973.pdf) (May 2018) </b>

<b>:movie_camera::one::package: [MonoPerfCap: Human Performance Capture from Monocular Video](http://gvv.mpi-inf.mpg.de/projects/wxu/MonoPerfCap/content/monoperfcap.pdf) (Mar 2018)</b> [[Project]](http://gvv.mpi-inf.mpg.de/projects/wxu/MonoPerfCap/)
###### makes a 3d replica of you without the expensive camera sphere

:package:[Learning to Estimate 3D Human Pose and Shape from a Single Color Image](http://openaccess.thecvf.com/content_cvpr_2018/papers/Pavlakos_Learning_to_Estimate_CVPR_2018_paper.pdf) (May 2018)
###### SMPL brand Artist's Mannequin

<b>:camera::one::package: [3D Human Pose Estimation in the Wild by Adversarial Learning](https://arxiv.org/pdf/1803.09722.pdf) (Mar 2018)</b>

<b>:movie_camera::1234::package: [LCR-Net++: Multi-person 2D and 3D Pose Detection in Natural Images](https://arxiv.org/pdf/1803.00455.pdf) (Mar 2018)</b> [[Project]](https://thoth.inrialpes.fr/src/LCR-Net/)

<b>:camera::one::package: [Unsupervised Adversarial Learning of 3D Human Pose from 2D Joint Locations](https://arxiv.org/pdf/1803.08244.pdf) (Mar 2018)</b> [[Project page]](https://nico-opendata.jp/en/casestudy/3dpose_gan/index.html)

<a name="Winter 2017"/>

### Winter 2017
<b>:movie_camera::one::package: [End-to-end Recovery of Human Shape and Pose](https://arxiv.org/pdf/1712.06584.pdf) (Dec 2017)</b> [[CODE]](https://github.com/akanazawa/hmr)

<b>:camera::1234::package: [DensePose: Dense Human Pose Estimation In The Wild](https://arxiv.org/pdf/1802.00434.pdf) [[CODE]](https://github.com/facebookresearch/Densepose) (Feb 2018)</b> [[Project page]](http://densepose.org)

<a name="Fall 2018"/>

### Fall 2018

<b>:movie_camera::1234::door: [Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf) (Apr 2017)</b> [[CODE]](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

<b>:camera::one::door: [Adversarial PoseNet: A Structure-aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389.pdf) (May 2017)</b>

<b>:camera::one::package: [A simple yet effective baseline for 3d human pose estimation](https://arxiv.org/pdf/1705.03098.pdf) (Aug 2017)</b> [[CODE]](https://github.com/una-dinosauria/3d-pose-baseline)

<b>:movie_camera::one::package: [VNect: Real-time 3D Human Pose Estimation with a Single RGB Camera](http://gvv.mpi-inf.mpg.de/projects/VNect/content/VNect_SIGGRAPH2017.pdf) [[CODE]](https://github.com/timctho/VNect-tensorflow) (Jul 2017) </b> [[Project]](http://gvv.mpi-inf.mpg.de/projects/VNect/)

<b>:camera::one::package: [Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](https://arxiv.org/pdf/1701.00295.pdf) (Oct 2017)</b>

<b>:camera::one::package:[Knowledge-Guided Deep Fractal Neural Networks for Human Pose Estimation](https://arxiv.org/pdf/1705.02407.pdf) (Aug 2017) </b> [[CODE]](http://github.com/Guanghan/GNet-pose)

<a name="2016"/>

### 2016
<b>:camera::one::package: [Learning to Fuse 2D and 3D Image Cues for Monocular Body Pose Estimation](https://arxiv.org/pdf/1611.05708.pdf) (Nov 2016)</b> 

<b>:camera::one::package: [Monocular 3D Human Pose Estimation In The Wild Using Improved CNN Supervision](https://arxiv.org/pdf/1611.09813.pdf) (Nov 2016)</b> [[Project]](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)

<b>:camera::one::package: [MoCap-guided Data Augmentation for 3D Pose Estimation in the Wild](https://arxiv.org/pdf/1607.02046.pdf) (Oct 2016)</b>

<b>:camera::one::package:[3D Human Pose Estimation Using Convolutional Neural Networks with 2D Pose Information](https://arxiv.org/pdf/1608.03075.pdf) (Sep 2016) </b>

<b>:movie_camera::one::package: [Spatio-temporal Matching for Human Pose Estimation](http://www.f-zhou.com/hpe/2014_ECCV_STM.pdf) (Sep 2014)</b> [[Project]](http://www.f-zhou.com/hpe.html)

<b>:camera::one::package: [Keep it SMPL: Automatic Estimation of 3D Human Pose and Shape from a Single Image](https://arxiv.org/pdf/1607.08128.pdf) (Jul 2016)</b>

<b>:camera::one::door: [Stacked Hourglass Networks for Human Pose Estimation](https://arxiv.org/pdf/1603.06937.pdf) [[CODE]](https://github.com/umich-vl/pose-hg-demo) (Mar 2016) </b>

<b>:camera::one::door: [Convolutional Pose Machines](https://arxiv.org/pdf/1602.00134.pdf) [[CODE]](https://github.com/shihenw/convolutional-pose-machines-release) (Jan 2016) </b>

### 2014 & 2015

<b>:movie_camera::one::package: [Sparseness Meets Deepness: 3D Human Pose Estimation from Monocular Video](https://arxiv.org/pdf/1511.09439.pdf) (Nov 2015)</b> [[Project]](http://cis.upenn.edu/~xiaowz/monocap.html)

## DataSets

[Human3.6M](http://vision.imar.ro/human3.6m/description.php)
[HumanEva](http://humaneva.is.tue.mpg.de/)
[MPI-INF-3DHP](http://gvv.mpi-inf.mpg.de/3dhp-dataset/)
[Unite The People](http://files.is.tuebingen.mpg.de/classner/up/)
[Pose Guided Person Image Generation](https://arxiv.org/pdf/1705.09368.pdf) - [[CODE]](https://github.com/charliememory/Pose-Guided-Person-Image-Generation) - Ma, L., Jia, X., Sun, Q., Schiele, B., Tuytelaars, T., & Gool, L.V. (NIPS 2017)
[A Generative Model of People in Clothing](https://arxiv.org/pdf/1705.04098.pdf) - Lassner, C., Pons-Moll, G., & Gehler, P.V. (ICCV 2017)
[Deformable GANs for Pose-based Human Image Generation](http://openaccess.thecvf.com/content_cvpr_2018/papers/Siarohin_Deformable_GANs_for_CVPR_2018_paper.pdf) - [[CODE]](https://github.com/AliaksandrSiarohin/pose-gan) - Siarohin, A., Sangineto, E., Lathuilière, S., & Sebe, N. (CVPR 2018)
[Dense Pose Transfer](https://arxiv.org/pdf/1809.01995.pdf) - Neverova, N., Guler, R.A., & Kokkinos, I. (ECCV 2018)

## Guide

[Gesture and Sign Language Recognition with Deep Learning](https://biblio.ugent.be/publication/8573066/file/8573068)

[Human Pose Estimation 101](https://github.com/cbsudux/Human-Pose-Estimation-101)

[Bob](https://github.com/Bob130/Human-Pose-Estimation-Papers)

[Jessie](https://github.com/dongzhuoyao/jessiechouuu-adversarial-pose)

[Awesome](https://github.com/cbsudux/awesome-human-pose-estimation)

[HoreFice](https://github.com/horefice/Human-Pose-Estimation-from-RGB)

## My personal goals:
I'd like to find a project I can clone. 
I'd like to find a recent project. 
I'd like to find a project with the 3d work done. 
I'd like to find a project that can integrate with SteamVR. (bone locations instead of blobs/meshes)

First off, there ought to be a state of the art 2d pose detector, this is crucial. This 2d pose detector can return colored limbs corresponding to each body part, heat maps corresponding to joins, and FBI switches corresponding to limb direction in the z axis. Ideally the network, when extrapolating 2d pose to 3d information should take into account the previous frame, and an internalized GAN representation of what human poses can look like. Additionally, physics simulations of body mechanics can be used, as well as reprojection of 3d joints back to 2d geometery. Additionally, there can be two cameras in operation, and these two cameras should return the same 3d pose. Additionally the 3d pose can be iteratively refined. Additionally there should be 3d pose standarization. Additionally 2d pose should be done well.

<b>:camera::one::package: [Deep Textured 3D Reconstruction of Human Bodies](https://arxiv.org/pdf/1809.06547v1.pdf) (Sep 2018)</b>[[Project]](http://www.f-zhou.com/hpe.html)
###### I'm going to make a replica out of you from too soft clay. Not relevant since blob based.

It's all Relative: Monocular 3D Human Pose Estimation from Weakly Supervised Data (May 2018)

:package:[Unsupervised Geometry-Aware Representation for 3D Human Pose Estimation](http://openaccess.thecvf.com/content_ECCV_2018/papers/Helge_Rhodin_Unsupervised_Geometry-Aware_Representation_ECCV_2018_paper.pdf)  [[CODE]](https://github.com/hrhodin/UnsupervisedGeometryAwareRepresentationLearning)  (Apr 2018)

:package:[BodyNet: Volumetric Inference of 3D Human Body Shapes](https://arxiv.org/pdf/1804.04875v3.pdf) [[CODE]](https://github.com/gulvarol/bodynet) (Apr 2018)

:door:[Simple Baselines for Human Pose Estimation and Tracking](http://openaccess.thecvf.com/content_ECCV_2018/papers/Bin_Xiao_Simple_Baselines_for_ECCV_2018_paper.pdf) [[CODE]](https://github.com/Microsoft/human-pose-estimation.pytorch) (Apr 2018)

A generalizable approach for multi-view 3D human pose regression (Apr 2018)

Learning Monocular 3D Human Pose Estimation from Multi-view Images (Mar 2018)

Multi-Scale Structure-Aware Network for Human Pose Estimation (Mar 2018)

Mo2Cap2: Real-time Mobile 3D Motion Capture with a Cap-mounted Fisheye Camera (Mar 2018)

Image-based Synthesis for Deep 3D Human Pose Estimation (Feb 2018)

:door:[LSTM Pose Machines](https://arxiv.org/pdf/1712.06316.pdf) [[CODE]](https://github.com/lawy623/LSTM_Pose_Machines) (Dec 2017)

Single-Shot Multi-Person 3D Pose Estimation From Monocular RGB (Dec 2017)

Using a single RGB frame for real time 3D hand pose estimation in the wild (Dec 2017)

:package:[Learning 3D Human Pose from Structure and Motion](http://openaccess.thecvf.com/content_ECCV_2018/papers/Rishabh_Dabral_Learning_3D_Human_ECCV_2018_paper.pdf) (Nov 2017)

:package:[Integral Human Pose Regression](https://arxiv.org/pdf/1711.08229.pdf) [[CODE]](https://github.com/JimmySuen/integral-human-pose) (Nov 2017)

Exploiting temporal information for 3D pose estimation (Nov 2017)

:door:[Human Pose Estimation Using Global and Local Normalization](https://arxiv.org/pdf/1709.07220.pdf) (Sep 2017)

:door:[Learning Feature Pyramids for Human Pose Estimation](https://arxiv.org/pdf/1708.01101.pdf) [[CODE]](https://github.com/bearpaw/PyraNet) (Aug 2017)

:package:[Recurrent 3D Pose Sequence Machines](https://arxiv.org/pdf/1707.09695.pdf) (Jul 2017)

:door:[Self Adversarial Training for Human Pose Estimation](https://arxiv.org/pdf/1707.02439.pdf) [[CODE1]](https://github.com/dongzhuoyao/jessiechouuu-adversarial-pose)[[CODE2]](https://github.com/roytseng-tw/adversarial-pose-pytorch) (Jul 2017)

Learning Human Pose Models from Synthesized Data for Robust RGB-D Action Recognition (Jul 2017)

Faster Than Real-time Facial Alignment: A 3D Spatial Transformer Network Approach in Unconstrained Poses (Jul 2017)

:package:[Towards 3D Human Pose Estimation in the Wild: a Weakly-supervised Approach](https://arxiv.org/pdf/1704.02447.pdf) [[CODE]](https://github.com/xingyizhou/Pytorch-pose-hg-3d) (Apr 2017)

[Adversarial PoseNet: A Structure-Aware Convolutional Network for Human Pose Estimation](https://arxiv.org/pdf/1705.00389.pdf) (Apr 2017)

Forecasting Human Dynamics from Static Images (Apr 2017)

:package:[Compositional Human Pose Regression](https://arxiv.org/pdf/1704.00159.pdf) (Apr 2017)

2D-3D Pose Consistency-based Conditional Random Fields for 3D Human Pose Estimation (Apr 2017)

:door:[Multi-context Attention for Human Pose Estimation](https://arxiv.org/pdf/1702.07432.pdf) - [[CODE]](https://github.com/bearpaw/pose-attention) (Feb 2017)

:package:[Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image](https://arxiv.org/pdf/1701.00295.pdf) (Jan 2017)

:door:[Towards Accurate Multi-person Pose Estimation in the Wild](https://arxiv.org/pdf/1701.01779.pdf) [[CODE]](https://github.com/hackiey/keypoints) (Jan 2017)

Lifting from the Deep: Convolutional 3D Pose Estimation from a Single Image (Jan 2017)

Learning from Synthetic Humans (Jan 2017)

MonoCap: Monocular Human Motion Capture using a CNN Coupled with a Geometric Prior (Jan 2017)

:door:[RMPE: Regional Multi-person Pose Estimation](https://arxiv.org/pdf/1612.00137.pdf) [[CODE1]](https://github.com/Fang-Haoshu/RMPE)[[CODE2]](https://github.com/MVIG-SJTU/AlphaPose) (Dec 2016)

:package:[Coarse-to-Fine Volumetric Prediction for Single-Image 3D Human Pose](https://arxiv.org/pdf/1611.07828.pdf) [[CODE]](https://github.com/geopavlakos/c2f-vol-demo) (Nov 2016)

:door:[Realtime Multi-person 2D Pose Estimation Using Part Affinity Fields](https://arxiv.org/pdf/1611.08050.pdf) [[CODE]](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation) (Nov 2016)

3D Human Pose Estimation from a Single Image via Distance Matrix Regression (Nov 2016)

Learning camera viewpoint using CNN to improve 3D body pose estimation (Sep 2016)

EgoCap: Egocentric Marker-less Motion Capture with Two Fisheye Cameras (Sep 2016)

:package:[Structured Prediction of 3D Human Pose with Deep Neural Networks](https://arxiv.org/pdf/1605.05180.pdf) (May 2016)

:door:[DeeperCut: A Deeper, Stronger, and Faster Multi-Person Pose Estimation Model](https://arxiv.org/pdf/1605.03170.pdf) [[CODE1]](https://github.com/eldar/deepcut-cnn)[[CODE2]](https://github.com/eldar/pose-tensorflow) (May 2016)

:door:[Recurrent Human Pose Estimation](https://arxiv.org/pdf/1605.02914.pdf) [[CODE]](https://github.com/ox-vgg/keypoint_detection) (May 2016)

Synthesizing Training Images for Boosting Human 3D Pose Estimation (Apr 2016)

Seeing Invisible Poses: Estimating 3D Body Pose from Egocentric Video - Completely insane and above the scope of science (Mar 2016)

:door:[DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation](https://arxiv.org/pdf/1511.06645.pdf) [[CODE]](https://github.com/eldar/deepcut) (Nov 2015)

A Dual-Source Approach for 3D Pose Estimation from a Single Image (Sep 2015)

:door:[Human Pose Estimation with Iterative Error Feedback](https://arxiv.org/pdf/1507.06550.pdf) [[CODE]](https://github.com/pulkitag/ief) (Jul 2015)

:door:[Flowing ConvNets for Human Pose Estimation in Videos](https://arxiv.org/pdf/1506.02897.pdf) [[CODE]](https://github.com/tpfister/caffe-heatmap) (Jun 2015)

:package:[3D Human Pose Estimation from Monocular Images with Deep Convolutional Neural Network](http://visal.cs.cityu.edu.hk/static/pubs/conf/accv14-3dposecnn.pdf) (Nov 2014)

:door:[Efficient Object Localization Using Convolutional Networks](https://arxiv.org/pdf/1411.4280.pdf) (Nov 2014)

:door:[MoDeep: A Deep Learning Framework Using Motion Features for Human Pose Estimation](https://arxiv.org/pdf/1409.7963.pdf) (Sep 2014)

:door:[Joint Training of a Convolutional Network and a Graphical Model for Human Pose Estimation](https://arxiv.org/pdf/1406.2984.pdf) [[CODE]](https://github.com/max-andr/joint-cnn-mrf) (Jun 2014)

:door:[Learning Human Pose Estimation Features with Convolutional Networks](https://arxiv.org/pdf/1312.7302.pdf) (Dec 2013)

:door:[DeepPose: Human Pose Estimation via Deep Neural Networks](https://arxiv.org/pdf/1312.4659.pdf) (Dec 2013)
