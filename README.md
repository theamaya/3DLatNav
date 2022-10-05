# 3DLatNav: Navigating generative latent spaces for semantic aware 3D object manipulation

* * *

**This is the official code repository of the ECCV submission - Paper ID 3744**

### Abstract

3D generative models have been recently successful in generating realistic 3D objects in the form of point clouds. However, existing generative models fail to manipulate the semantics of one or many different parts without extensive semantic attribute labels of 3D object parts or reference target point clouds, significantly limiting practicality. Furthermore, there is a lack of understanding on how the semantics of these non-Euclidean and permutation-invariant representations of 3D shapes are encoded in their corresponding generative latent spaces, beyond the ability to perform simple latent vector arithmetic or interpolations. In this paper, we first propose a part-based unsupervised semantic attribute identification mechanism using latent representations of 3D shape reconstruction models. Then, we transfer that knowledge to latent spaces of pretrained 3D generative models to unravel that their latent spaces embed disentangled representations for the component parts of objects in the form of linear subspaces, despite the unavailability of part-level labels during the training. Finally, we utilize the identified subspaces to develop a part-aware controllable 3D object editing framework that can operate on any pretrained 3D generative model.  With multiple ablation studies and testing on state-of-the-art generative models, we show that the proposed method can implement part-level semantic editing on an input point cloud while preserving other features and the realistic nature of the object. 

* * *
Original  chairs
![alt text](https://github.com/entc-17-fyp-05/CVPR_2022_1440/blob/master/gifs/Original.png)

Editing Legs - Straight legs, Swivel legs, Cantilever legs
![me](https://github.com/entc-17-fyp-05/CVPR_2022_1440/blob/master/gifs/gif0.gif)

Editing Armrests - Simple armrests, Connected armrests, Remove armrests
![me](https://github.com/entc-17-fyp-05/CVPR_2022_1440/blob/master/gifs/gif1.gif)

Editing Backrests - Curved backrests, Stuffed backrests, Reclined backrests
![me](https://github.com/entc-17-fyp-05/CVPR_2022_1440/blob/master/gifs/gif2.gif)

Editing Seats - Narrow seats, Wide seats, Stuffed seats
![me](https://github.com/entc-17-fyp-05/CVPR_2022_1440/blob/master/gifs/gif3.gif)

* * *

### Installation

##### **Step 01** : Install the conda enviroment

    # Creates a conda environment and installs dependencies
    ./install.sh

* * *

### Datasets and Pre-Trained Models

#### **Step 01** : Download the datasets
Datasets are available [here](https://drive.google.com/drive/folders/1Xbv8XzWUkYb_-p34eAkwBYIAl2buL91n?usp=sharing)
    Download and place them at the path "Final_Repository/data"

#### **Step 02** : Download the Trained Models
Pre-Trained models are available [here](https://drive.google.com/drive/folders/10Zt2EKBBPn743-5onwiRJ_USxlrApVft?usp=sharing)

    Download and place them at the path "Final_repository/models/*--model--*/Trained_models"

    You can find pre-trained models of the following
        1. PointNet Part Segmentation
        2. 3DAAE
        3. Pointflow
        4. Diffusion Probabilitic Models
        
* **Example-1** : If you want to edit pointclouds with the **Diffusion Probabilistic Model** for the **chair class** you will need the following: 

        | Dataset           | Path to Include                         |
        |-------------------|-----------------------------------------|
        | chair_parts.npy   | Final_Repository/data/chair_parts.npy   |
        | chair_objects.npy | Final_Repository/data/chair_objects.npy |
       
       
        | Trained Model         | Path to Include                                                          |
        |-----------------------|--------------------------------------------------------------------------|
        | AAE_chair_parts/E.pth | Final_Repository/models/AAE/Trained_Models/AAE_chair_parts/E.pth         |
        | AAE_chair_parts/G.pth | Final_Repository/models/AAE/Trained_Models/AAE_chair_parts/G.pth         |
        | chair.pth             | Final_Repository/models/Part_Segment/Trained_Models/chair.pth            |
        | latent_con_model.pth  | Final_Repository/models/Part_Segment/Trained_Models/latent_con_model.pth |
        | DPM_chair_objects.pt  | Final_Repository/models/Diffusion/Trained_Models/DPM_chair_objects.pt    |
        
        

### Editing Framework

Run the notebook **"Final_Experiments2.ipynb"** from the begining. The customized entries need to be specified at the mentioned places in the notebook. (You will have to specify the object category, Root DIR, preferred generative model and the optimum number of clusters based on the Davies Bouldin score. The respective locations and instructions are specified in the notebook.

* * *

### Credits

Our Framework can be deployed in the existing Generative Models. 
Below are the three models we have used in this implementation.

#### 3DAAE

[code](https://github.com/MaciejZamorski/3d-AAE) [paper](https://arxiv.org/abs/1811.07605)

<!--     @article{Zamorski20203dAAE,
      year = {2020},
      month = {April},
      volume = {193},
      articles= {102921},
      author = {Maciej Zamorski and Maciej Zi{\k{e}}ba and Piotr Klukowski and Rafa{\l} Nowak and Karol Kurach and Wojciech Stokowiec and Tomasz Trzci{\'{n}}ski},
      title = {Adversarial autoencoders for compact representations of 3{D} point clouds},
      journal = {Computer Vision and Image Understanding}
    }
 -->
#### Pointflow

[code](https://github.com/stevenygd/PointFlow) [paper](https://arxiv.org/abs/1906.12320)

<!--     @InProceedings{guandao2020pointflow,
    author = {Yang, Guandao and Huang, Xun and Hao, Zekun and Liu, Ming-Yu and Belongie, Serge and Hariharan, Bharath},
    title = {PointFlow: 3{D} Point Cloud Generation With Continuous Normalizing Flows},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
    } -->

#### Diffusion Probabilistic Models

[code](https://github.com/luost26/diffusion-point-cloud) [paper](https://arxiv.org/abs/2103.01458)

<!--     @inproceedings{luo2021diffusion,
      author = {Luo, Shitong and Hu, Wei},
      title = {Diffusion Probabilistic Models for 3D Point Cloud Generation},
      booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
      month = {June},
      year = {2021}
    } -->








