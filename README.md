# vsum_dsf

## How to set it up

	git clone https://github.com/prateeksarangi.git

### Install dependency

You can install required python packages using conda:

	conda env create -f vsum_dsf/environment.yml
	
Requirements:
- numpy=1.11
- scipy
- scikit learn
- chainer=2.0

Optional:
- scikit video ( for exporting video )

This code utilizes tools provided by M. Gygli *et al.* [1].
You can set it up by:

	cd vsum_dsf
	git clone https://github.com/prateeksarangi/gm_submodular.git
	cd gm_submodular
	python3 setup.py install --user

[1] Gygli, Grabner & Van Gool. Video Summarization by Learning Submodular Mixtures of Objectives. CVPR 2015.

### Download dataset and model parameters

To test the model in the paper, download a `data.zip` [**HERE**](https://www.dropbox.com/s/zxp8dq18t0tqlk2/data.zip?dl=0) and extract it in the folder `vsum_dsf`.

The demo performs video summarization on the SumMe dataset ([project page](https://people.ee.ethz.ch/~gyglim/vsum/index.php)).

You can download the dataset as: 

	cd data/summe
	wget https://data.vision.ee.ethz.ch/cvl/SumMe/SumMe.zip
	unzip SumMe.zip

## Example

See the [notebook](https://github.com/mayu-ot/vsum_dsf/blob/master/Demo.ipynb) or:

	python3 script/summarize.py
	python3 script/evaluate.py results/summe/smt_feat
