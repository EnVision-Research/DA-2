# <img src="assets/badges/icon2.png" alt="lotus" style="height:1.2em; vertical-align:bottom;"/>&nbsp;DA<sup>2</sup>: Depth Anything in Any Direction

[![Page](https://img.shields.io/badge/Project-Website-pink?logo=googlechrome&logoColor=white)](https://depth-any-in-any-dir.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](http://arxiv.org/abs/2509.26618)
[![Data](https://img.shields.io/badge/📂%20HuggingFace-Data%20(Soon)-green)]()
[![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo%20-yellow)](https://huggingface.co/spaces/haodongli/DA-2)
[![Slides](https://img.shields.io/badge/Google-Slides-blue?logo=slideshare&logoColor=white)](https://docs.google.com/presentation/d/1QUonqLuYGEh0qcqY72pbTXsZimINlyN4rOogy7qX4GY/edit?usp=sharing)
[![BibTeX](https://img.shields.io/badge/BibTeX-grey?logo=googlescholar&logoColor=white)](https://github.com/EnVision-Research/DA-2?tab=readme-ov-file#-citation)

[Haodong Li](https://haodong2000.github.io/)<sup>123&sect;</sup>,
[Wangguangdong Zheng](https://wangguandongzheng.github.io/)<sup>1</sup>,
[Jing He](https://jingheya.github.io/)<sup>3</sup>,
[Yuhao Liu](https://yuhaoliu7456.github.io/)<sup>1</sup>,
[Xin Lin](https://linxin0.github.io/)<sup>2</sup>,
[Xin Yang](https://abnervictor.github.io/2023/06/12/Academic-Self-Intro.html)<sup>34</sup>,<br>
[Ying-Cong Chen](https://www.yingcong.me/)<sup>34&#9993;</sup>,
[Chunchao Guo]()<sup>1&#9993;</sup>

<span class="author-block"><sup>1</sup>Tencent Hunyuan</span>
<span class="author-block"><sup>2</sup>UC San Diego</span>
<span class="author-block"><sup>3</sup>HKUST(GZ)</span>
<span class="author-block"><sup>4</sup>HKUST</span><br>
<span class="author-block">
    <sup>&sect;</sup>Work primarily done during an internship at Tencent Hunyuan.
    <sup>&#9993;</sup>Corresponding author.
</span>

![teaser](assets/badges/teaser.jpg)

<strong>DA<sup>2</sup> predicts dense, scale-invariant distance from a single 360&deg; panorama in an end-to-end manner, with remarkable geometric fidelity and strong zero-shot generalization.</strong>

## 📢 News
- <strong>We are currently uploading the curated panoramic data, please stay tuned!😊</strong>
- 2025-10-04 The 🤗Huggingface Gradio demo ([online](https://huggingface.co/spaces/haodongli/DA-2) and [local](https://github.com/EnVision-Research/DA-2?tab=readme-ov-file#-gradio-demo)) are released!
- 2025-10-04 The inference code and the [model](https://huggingface.co/haodongli/DA-2) are released!
- 2025-10-01 [Paper](https://arxiv.org/abs/2509.26618) released on arXiv!

## 🛠️ Setup
| This installation was tested on: Ubuntu 20.04 LTS, Python 3.12, CUDA 12.2, NVIDIA GeForce RTX 3090.  

1. Clone the repository:
```
git clone https://github.com/EnVision-Research/DA-2.git
cd DA-2
```
2. Install dependencies using conda:
```
conda create -n da-2 python=3.12 -y
conda activate da-2
pip install -e src/
```

## 🤗 Gradio Demo
1. Online demo: [Hugggingface Space](https://huggingface.co/spaces/haodongli/DA-2)
2. Local demo:
```
python app.py
```

## 🕹️ Inference
| We've pre-uploaded the cases appeared in the [project page](https://depth-any-in-any-dir.github.io/). So you can proceed directly to step 3.😊

1. Images are placed in a directory, e.g., `assets/demos`.
2. (Optional) Masks (e.g., sky masks for outdoor images) in another directory, e.g., `assets/masks`. The filenames under both directories should be consistent.
3. Run the inference command:
```
sh infer.sh
```
4. The visualized distance and normal maps will be saved at `output/infer/vis_all.png`. The projected 3D point clouds will be saved at `output/infer/3dpc`.

## 🎓 Citation
If you find our work useful in your research, please consider citing our paper:
```bibtex
@article{li2025depth,
  title={DA $\^{} 2$: Depth Anything in Any Direction},
  author={Li, Haodong and Zheng, Wangguangdong and He, Jing and Liu, Yuhao and Lin, Xin and Yang, Xin and Chen, Ying-Cong and Guo, Chunchao},
  journal={arXiv preprint arXiv:2509.26618},
  year={2025}
}
```

## 🤝 Acknowledgement
This implementation is impossible without the awesome contributions of [MoGe](https://wangrc.site/MoGePage/), [UniK3D](https://lpiccinelli-eth.github.io/pub/unik3d/), [DINOv2](https://github.com/facebookresearch/dinov2), [Accelerate](https://github.com/huggingface/accelerate), [Gradio](https://github.com/gradio-app/gradio), [HuggingFace Hub](https://github.com/huggingface/huggingface_hub), and [PyTorch](https://pytorch.org/) to the open-cource community.
