This repo includes the codes for the paper of 'Membership Inference Attacks Against Recommender Systems'.

There are two .py files which are our attack models.
One is based on a clustering algorithm. And the other is based on the deep learning technique.

And there are three types of data set, i.e., "Interactions", "Recommendations" and "Vectorizations".
- "Interactions" is formatted as: ``UserID`` ``\t`` ``ItemID`` ``\t`` ``Scores`` ``\n``
- "Recommendations" is formatted as : ``UserID`` ``\t`` ``ItemID`` ``\t`` ``Scores`` ``\n``
- "Vectorization" is formatted as: ``Vector[i][1]`` ``\t`` ``Vector[i][2]`` ``\t`` ... ``\t`` ``Vector[i][m]`` ``\n`` (Here, $m$ is the dimension of the feature space, and $i$ means this feature vector corresponds to the $i^{th}$ user.)

Note that, to balance data, the first half of Interactions and Recommendations are for Members while the second half of them are for Non-Members.

To acknowledge use of the model in publications, please cite the following paper:
[Membership Inference Attacks Against Recommender Systems](https://arxiv.org/abs/2109.08045)
