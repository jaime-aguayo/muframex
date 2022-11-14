
## human-motion-prediction
Pytorch implementation of:

Julieta Martinez, Michael J. Black, Javier Romero.
_On human motion prediction using recurrent neural networks_. In CVPR 17.

It can be found on arxiv as well: https://arxiv.org/pdf/1705.02445.pdf

The code in the original repository was written by [Julieta Martinez](https://github.com/una-dinosauria/) and [Javier Romero](https://github.com/libicocco/) and is accessible [here](/blob/master/src/translate.py).

If you have any comment on this fork you can email me at [enriccorona93@gmail.com]

### Dependencies

* [h5py](https://github.com/h5py/h5py) -- to save samples
* [Pytorch](https://pytorch.org/)

### Get this code and the data

First things first, clone this repo and get the human3.6m dataset on exponential map format.

```bash
git clone https://github.com/enriccorona/human-motion-prediction-pytorch.git
cd human-motion-prediction-pytorch
mkdir data
cd data
wget http://www.cs.stanford.edu/people/ashesh/h3.6m.zip
unzip h3.6m.zip
rm h3.6m.zip
cd ..
```

### Quick demo and visualization

For a quick demo, you can train for a few iterations and visualize the outputs
of your model.

To train, run
```bash
python src/train.py --action walking --seq_length_out 25 --iterations 10000
```

To save some samples of the model, run
```bash
python src/test.py --action walking --seq_length_out 25 --iterations 10000 --load 10000
```

Finally, to visualize the samples run
```bash
python src/animate.py
```

This should create a visualization similar to this one

<p align="center">
  <img src="https://raw.githubusercontent.com/una-dinosauria/human-motion-prediction/master/imgs/walking.gif"><br><br>
</p>


You can substitute the `--action walking` parameter for any action in

```
["directions", "discussion", "eating", "greeting", "phoning",
 "posing", "purchases", "sitting", "sittingdown", "smoking",
 "takingphoto", "waiting", "walking", "walkingdog", "walkingtogether"]
```

or `--action all` (default) to train on all actions.

### Citing

If you use our code, please cite our work

```
@inproceedings{julieta2017motion,
  title={On human motion prediction using recurrent neural networks},
  author={Martinez, Julieta and Black, Michael J. and Romero, Javier},
  booktitle={CVPR},
  year={2017}
}
```

### Acknowledgments

The pre-processed human 3.6m dataset and some of our evaluation code (specially under `src/data_utils.py`) was ported/adapted from [SRNN](https://github.com/asheshjain399/RNNexp/tree/srnn/structural_rnn) by [@asheshjain399](https://github.com/asheshjain399).

### Licence
MIT
