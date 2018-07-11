Notes
=====

Running
-------

singularity exec --nv ~/images/tensorflow/ python scripts/train_unet_model.py

To try
------

* DONE Test data from broken images
* Dilate 0
* Use weighted losses
* Fix fitting for keras flow
* Check weighting of cell wall/not wall signal
* Measure compared to "ground truth"
* Estimate threshold classification accuracy
* Fix train vs. test
