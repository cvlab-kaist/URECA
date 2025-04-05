<div align="center">
<h1>URECA💡: Unique Region Caption Anything</h1>

[**Sangbeom Lim**](https://sangbeomlim.github.io/)<sup>1\*</sup> · [**Junwan Kim**](https://junwankimm.github.io/)<sup>2\*</sup> · [**Heeji Yoon**](https://github.com/yoon-heez)<sup>3</sup> · [**Jaewoo Jung**](https://crepejung00.github.io/)<sup>3</sup> · [**Seungryong Kim**](https://cvlab.korea.ac.kr)<sup>3&dagger;</sup>

<sup>1</sup>Korea University&emsp;&emsp;&emsp;&emsp;<sup>2</sup>Yonsei University&emsp;&emsp;&emsp;&emsp;<sup>3</sup>KAIST

*: Equal Contribution <br>
&dagger;: Corresponding Author

**ArXiv 2025**

[//]: # (<a href="https://arxiv.org/abs/2412.01471"><img src="https://img.shields.io/badge/arXiv-2412.01471-%23B31B1B"></a>)
<a href="https://cvlab-kaist.github.io/URECA"><img src="https://img.shields.io/badge/Project%20Page-online-brightgreen"></a>

<p float='center'><img src="assets/ureca_teaser.png" width="80%" /></p>
<span style="color: green; font-size: 1.3em; font-weight: bold;">URECA can generate <span style="color:darkolivegreen;font-weight:bold;">Unique Caption</span> for <span style="font-weight:bold;">Any Granularity Regions!</span></span>
</div>


## 📰 News
* **2025-04-06:** [URECA](https://github.com/cvlab-kaist/URECA) is released.
* **2025-04-06:** Training Code, Data collection pipeline, and URECA Model are released.

**Please stay tuned for a URECA Dataset and Evaluation Code!**

## 🔥 TODO
- [x] Train Code (Apr 6, 2024)
- [x] Pre-trained weights (Apr 6, 2024)
- [x] Code of interactive demo (Apr 6, 2024)
- [x] Demo update (Apr 6, 2024)
- [ ] Evaluation Code 
- [ ] Training and Test Dataset release

## Environment
```bash
conda create -n ureca python=3.9
conda activate ureca

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 🎮Demo
<p float='center'><img src="assets/demo_example.png" width="80%" /></p>

Please Download [SAM](https://github.com/facebookresearch/segment-anything) and place it on models folder.  
Download URECA Model by following bellow script.
```bash
mkdir models
cd models
git lfs install
git clone https://huggingface.co/SammyLim/URECA
mkdir sam
! Download SAM-H model weight manually!
python gradio_demo/app.py
```

## Data Curation Pipeline
- **[Data Curation Pipeline](./data_curation_pipeline/README.md)**

## Training
- **[URECA Training](./ureca_model/README.md)**


[//]: # (## Evaluation)

[//]: # (- **[URECA Evaluation]&#40;./evaluation/EVAL.md&#41;**)

[//]: # (## Datasets)



[//]: # (## 📚 Citing this Work)

[//]: # (Please use the following bibtex to cite our work:)

[//]: # (```)

[//]: # (@article{lim2024multi,)

[//]: # (  title={Multi-Granularity Video Object Segmentation},)

[//]: # (  author={Lim, Sangbeom and Kim, Seongchan and An, Seungjun and Cho, Seokju and Seo, Paul Hongsuck and Kim, Seungryong},)

[//]: # (  journal={arXiv preprint arXiv:2412.01471},)

[//]: # (  year={2024})

[//]: # (})

[//]: # (```)

## 🙏 Acknowledgement
This project is largely based on the [InternVL repository](https://github.com/OpenGVLab/InternVL). Thanks to the authors for their invaluable work and contributions.