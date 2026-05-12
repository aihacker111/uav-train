# AMOT
> Official code for multi-object tracker AMOT, AAAI, 2026
> 
![](readme/MOT.png)
> [**Tracking the Unstable: Appearance-Guided Motion Modeling for Robust Multi-Object Tracking in UAV-Captured Videos**](https://arxiv.org/abs/2508.01730),            
> Jianbo Ma, Hui Luo, Qi Chen, Yuankai Qi, Yumei Sun, Amin Beheshti, Jianlin Zhang, Ming-Hsuan Yang

## 💁 Get Started

### Environment preparation

> git clone https://github.com/ydhcg-BoBo/AMOT.git

> You can follow our [STCMOT](https://github.com/ydhcg-BoBo/STCMOT) to install the environment and prepare the datasets.


### Trained Models
| Dataset    | Link                                                                                    |
|------------|-----------------------------------------------------------------------------------------|
| Visdrone   | [BaiduDrive](https://pan.baidu.com/s/1BjGru5Xoi8tuejdG8VsnMw?pwd=2026) (password: 2026) |
| UAVDT      | [BaiduDrive](https://pan.baidu.com/s/1WX2zLrNSTRlekhWUhcJLQw?pwd=2026) (password: 2026) |
| VT-MOT-UAV | [BaiduDrive](https://pan.baidu.com/s/1FcPbXRnFAiNAc-oY2m1a3g?pwd=2026 ) (password: 2026) |
* VT-MOT-UAV is a subdata of [VTMOT](https://github.com/wqw123wqw/PFTrack)

### Test
```
cd src
Run python track_AMOT.py
```

## 📚 Citation
>If you find this code useful, please star the project and consider citing:
```bibtex
@article{ma2025tracking,
  title={Tracking the Unstable: Appearance-Guided Motion Modeling for Robust Multi-Object Tracking in UAV-Captured Videos},
  author={Ma, Jianbo and Luo, Hui and Chen, Qi and Qi, Yuankai and Sun, Yumei and Beheshti, Amin and Zhang, Jianlin and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2508.01730},
  year={2025}
}
```

## 🙏 Acknowledgement
A large part of the code is borrowed from [Fairmot](https://github.com/ifzhang/FairMOT) and [STCMOT](https://github.com/ydhcg-BoBo/STCMOT).
Thanks for their wonderful works.

>If you have any questions related to the paper and the code please contact me.
> 