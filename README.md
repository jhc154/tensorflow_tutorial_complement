# tensorflow_tutorial_complement
Jupyter Notebooks to complement the TensorFlow Object Detection Tutorial 

Self-study in the TensorFlow Object Detector Tutorial required more troubleshooting than I originally thought it would require. I am sharing these files to help others with workarounds to make the tutorial work. 

To use this repo, following along with the tutorial at https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html. 

General Notes:
- I recommend that you set up two environments for the object detection project, one on the local computer and the other in Google Colab. 
- Use your local computer to perform all steps up to and including the transformation of files to .record.
- Starting at the point where you want to train the model and run object detection, use Google Colab. 
- To create and work with files like the label_map.pbtxt and pipeline.config files, I used a text editor (Atom). 

My files to TensorFlow Tutorial Mapping:

- get_image.ipnyb | start with a repository of your own images or alter my notebook to scrape a website of your choosing to get started with a training set of data. https://github.com/jhc154/tensorflow_tutorial_complement/blob/master/get_images.ipynb

1. Follow as Directed on Local Machine: Preparing Workspace, Annotating Images (labelimg), Creating Label Map (.pbtxt)

2. xml_to_csv.ipnyb | Converting *.xml to *.csv: As written, the script failed to work in my environment so I re-wrote the script to run from a jupyter notebook. https://github.com/jhc154/tensorflow_tutorial_complement/blob/master/xml_to_csv.ipynb

3. csv_to_tfrecord.ipnyb | Converting from *.csv to *.record: As written, the script failed to work in my environment so I re-wrote the script to run from a jupyter notebook. https://github.com/jhc154/tensorflow_tutorial_complement/blob/master/csv_to_tfrecord.ipynb

- At some point you should create a workspace in your google drive, if you have not already, that looks similar to the workspace that the tutorial recommends. Copy all the files that you created on your local machine to the google drive; make sure to copy them to their appropriate folder. 

4. ssd_inception_v2_coco_2018_01_28.config | Configuring a Training Pipeline: Follow as directed and be mindful that you will have to download both models and config files from the TensorFlow Model Zoo. For reference specific to my project, I shared the config file that I used; change paths to your environment to make it work. See comments. 

- Models: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

- Configs: https://github.com/tensorflow/models/tree/master/research/object_detection/samples/configs

5. tf_working_public.ipynb | Training the Model: To follow the tutorial, I used a jupyter notebook in Colab to run all my commands (python and terminal). A critical deficiency that I found while following along in the tutorial is that the train.py does not work with versions higher than 1.12.0. Version 1.15 is currently the default and is headed to 2.x soon as of January 2020. On that note, train.py is to be deprecate also but you can still use it. You have two choices, downgrade for the sake of being able to perform these tasks or adapt everything to work with 2.x. I tried really hard to do the latter but found that downgrading to 1.12.0 was the only option that I could be succesful with. 

- Use tf_working_public.ipynb to set up the environment and run train.py.

- When training is completed (Exporting a Trained Inference Graph), use tf_working_public.ipynb to export the custom inference graph that you created with train.py.

6. tf_tensorboard.ipnyb | Monitor Training Job Progress using TensorBoard: The tutorial did not work for me as it is written. From a different window than tf_working_public.ipnyb, launch tf_tensorboard.ipnyb until you get a blank tensorboard. Come back to this window after it starts logging. Note: I found that TensorFlow did not start creating tfevent files until after I paused and restarted the cell that launches train.py from tf_working_public.ipnyb; once I paused and resumed that cell, I could see that tensorboard was displaying graphs. 


7. tf_object_detect_public.ipnyb | Nothing in Tutorial: The tutorial ends with the exporting of an inference graph so I felt a bit unsatisfied becuase I was not sure how to use the model and see if it would detect my image.  After some research and Googling, I found a pretty good solution from http://www.insightsbot.com/tensorflow-object-detection-tutorial-on-images/. 

- use tf_object_detect_public.ipnyb to see if pre-trained models from TensorFlow Model Zoo can pick up labels in your image. In some ways, it makes sense to do this step first to see whether any of the models can even detect that there is something in the image. I would imagine that the model that can draw a bounding box around the thing of interest (even if mis-labeled), can become a better model after you training it.  

- use tf_object_detect_public.ipnyb to see if your custom detector can detect your images succesfully. 






